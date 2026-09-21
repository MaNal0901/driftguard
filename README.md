# DriftGuard

An MLOps pipeline for energy consumption forecasting, built to demonstrate a full model lifecycle: data versioning, experiment tracking, automated drift detection, and CI/CD-triggered retraining.

## Objective

Most portfolio ML projects stop at "trained a model, got a good score." DriftGuard is built around a different question: what happens to a model after it ships? Forecasting models degrade over time as the real world diverges from the data they were trained on. This project implements the monitoring loop that catches that degradation automatically , measuring drift, deciding whether it matters, and triggering retraining without manual intervention.

The forecasting task itself (predicting hourly energy consumption) is a vehicle for this loop, not the end goal. Model accuracy is treated as good enough rather than maximized, in favor of keeping the pipeline simple enough to reason about end to end.

## How the pipeline fits together

```
data/raw (DVC) → preprocessing → train (MLflow) → serve (FastAPI / Docker)
                                                          |
                                                    drift_check (Evidently)
                                                          |
                                          drift detected → retrain (CI/CD)
```

Each stage is a standalone script, chained together by DVC (`dvc.yaml`) and, in CI, by GitHub Actions. Nothing in the pipeline assumes it runs end to end in one process : `preprocessing.py`, `train.py`, and `drift_check.py` can each be run and inspected independently.

## Stack

| Concern | Tool |
|---|---|
| Data versioning | DVC |
| Experiment tracking | MLflow |
| Drift detection | Evidently |
| Model serving | FastAPI |
| Monitoring dashboard | Streamlit |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Model | scikit-learn (RandomForestRegressor) |

## Dataset

[PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) (PJME_hourly.csv) : 145,366 hourly readings of electricity load from 2002 to 2018, in megawatts.

Exploratory analysis (see `notebooks/01_eda.ipynb`) surfaced two things that shaped the rest of the pipeline:

- **Data quality is tied to US daylight saving time.** Every duplicate timestamp and every gap in the hourly index lines up exactly with a DST transition. `preprocessing.py` deduplicates, reindexes to a complete hourly range, and linearly interpolates the resulting gaps.
- **The series has a real, measurable drift**, not a synthetic one. Seasonal decomposition shows a stable annual cycle on top of a gradual downward trend in average consumption from 2010 onward — plausibly efficiency gains in appliances and buildings. This made it possible to build the drift-monitoring stages around an actual phenomenon in the data rather than injecting artificial noise.

Given that trend, the data is split chronologically with no overlap: `train` (2002–2015) fits the model, `reference` (2016) stands in for "the world as the model last saw it," and `current` (2017) plays the role of incoming production data.

## Features

The model uses eight calendar-derived features: cyclical encodings (sine/cosine) of hour, month, and day of week, plus binary flags for weekend and US federal holidays. All of them are deterministic functions of the timestamp , none depend on the consumption history itself.

That constraint was intentional. Lag or rolling-window features would likely improve accuracy, but they derive from the target series, which would make the drift signal harder to interpret (drift in a lag feature is just an echo of drift in the target) and would require the serving API to have access to recent history rather than a single timestamp. Keeping features purely calendar-based keeps the monitoring story clean, at a known cost to accuracy. That trade-off, and what a production version would add, is documented under Roadmap.

## Model

`RandomForestRegressor`, chosen after comparing it against `HistGradientBoostingRegressor` on the same features and split. The two produced effectively identical validation performance (R² 0.7046 vs 0.7039), which indicated the ceiling was set by the feature set, not the algorithm RandomForest was kept for being the simpler of the two to reason about and explain.

Validation (on `reference`, 2016 : never seen during training):

| Metric | Train | Validation |
|---|---|---|
| MAE | 2418.5 MW | 2912.1 MW |
| RMSE | 3284.9 MW | 3682.2 MW |
| R² | 0.7418 | 0.7046 |

The gap between train and validation is moderate, indicating the model generalizes rather than memorizes. The remaining error is consistent with the feature set: temperature, the primary physical driver of electricity demand, isn't in the data.

## Drift monitoring

This is the part of the project the rest is built to support, and it required correcting an assumption partway through. Monitoring feature drift alone (`DataDriftPreset` on the eight inputs) reliably reports 0% drift between any two years, because calendar-derived features are invariant by construction : the distribution of months, hours, and weekdays in 2016 is structurally identical to 2017. That's expected, not a bug, but it also means feature drift can never detect the thing the project is actually about.

The pipeline therefore tracks two signals separately:

- **Feature drift** : on the eight model inputs. Expected to stay near zero; useful mainly as a sanity check that the feature engineering hasn't broken.
- **Target drift** : on `PJME_MW` itself, via `ColumnDriftMetric`. This is the signal that reflects concept drift: does the relationship between calendar and consumption still hold? Between 2016 and 2017, it does not (drift score 0.101, flagged as detected), consistent with the downward trend seen in the EDA.

Retraining is triggered off target drift, not feature drift, since that's the signal that actually corresponds to model degradation.

## Project structure

```
driftguard/
├── notebooks/              EDA (exploration only, not part of the pipeline)
├── src/
│   ├── data/                preprocessing.py — cleaning, feature engineering, splitting
│   ├── models/               train.py, predict.py
│   ├── monitoring/           drift_check.py
│   └── api/                  FastAPI app
├── dashboard/                 Streamlit monitoring dashboard
├── tests/                     unit tests (synthetic data, no dependency on the raw CSV)
├── dvc.yaml                   pipeline stages (preprocessing → train → drift_check)
├── Dockerfile, Dockerfile.dashboard, docker-compose.yml
└── .github/workflows/ci-cd.yml
```

## Running it

```bash
git clone <repo-url>
cd driftguard
python -m venv venv
source venv/bin/activate   # venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cp .env.example .env
```

Download `PJME_hourly.csv` from Kaggle into `data/raw/`, then:

```bash
python -m src.data.preprocessing
python -m src.models.train
python -m src.monitoring.drift_check
```

Serve the model and the dashboard:

```bash
uvicorn src.api.app:app --reload --port 8000   # docs at /docs
streamlit run dashboard/app.py
```

Or run the whole pipeline through DVC, which skips stages whose inputs haven't changed:

```bash
dvc repro
```

### Docker

```bash
docker compose up --build
```

API at `localhost:8000`, dashboard at `localhost:8501`.

### Tests

```bash
pytest tests/ -v
```

All fixtures are synthetic, so the suite doesn't require the raw dataset to be present.

## CI/CD

GitHub Actions runs on every push: linting (flake8, black), the unit test suite, and a build of both Docker images. It stops there rather than re-running the DVC pipeline on the runners — the DVC remote in this setup is local storage, which the runners can't reach. In a production setup, pointing the remote at S3 or GCS would make `dvc pull && dvc repro` a straightforward addition to the workflow.

## Roadmap

Documented here rather than implemented, since they fall outside what this project set out to demonstrate:

| Addition | Why it would help |
|---|---|
| Weather data (temperature) | The main physical driver of demand; likely the single biggest accuracy gain available |
| Lag and rolling-window features | Would capture short-term momentum in consumption, at the cost of complicating the drift signal as noted above |
| Holiday eve flags | Consumption often shifts the day before a holiday, not just on it |
| Time-series-specific models (Prophet, SARIMA) | Purpose-built for multi-seasonality (daily + weekly + annual) |
| Cloud DVC remote | Would let CI actually reproduce the pipeline, not just build and test the code |