"""
Détection de drift entre les données de référence (2016) et les données
"actuelles" (2017, simulant la production), via Evidently.

Usage:
    python -m src.monitoring.drift_check
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric

from src.models.train import FEATURE_COLUMNS, TARGET_COLUMN

PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", "data/processed")
REPORTS_DIR = "reports"
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))


def load_reference_and_current():
    """
    Objectif dans le pipeline : charger reference (2016) et current (2017)
    en gardant à la fois les features ET la target. C'est nécessaire pour
    pouvoir surveiller le concept drift (drift sur PJME_MW), pas seulement
    le data drift sur les features calendaires (qui ne peut pas driftter
    par construction).
    """
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    reference = pd.read_csv(os.path.join(PROCESSED_DIR, "reference.csv"))[columns]
    current = pd.read_csv(os.path.join(PROCESSED_DIR, "current.csv"))[columns]
    return reference, current


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    """
    Objectif dans le pipeline : deux signaux distincts et complémentaires :
    - DataDriftPreset sur les features → attendu à 0% ici (features
      calendaires invariantes), sert de vérification de cohérence.
    - ColumnDriftMetric sur PJME_MW → LE signal qui importe vraiment :
      est-ce que la consommation elle-même a une distribution différente
      entre 2016 et 2017 ? C'est ça, le concept drift qu'on veut détecter.
    """
    report = Report(
        metrics=[
            DataDriftPreset(columns=FEATURE_COLUMNS),
            ColumnDriftMetric(column_name=TARGET_COLUMN),
        ]
    )
    report.run(reference_data=reference, current_data=current)
    return report


def extract_drift_summary(report: Report) -> dict:
    """
    Objectif dans le pipeline : extraire les deux signaux séparément
    pour ne pas les confondre — un data drift sur les features et un
    drift sur la target sont deux informations différentes.
    """
    result = report.as_dict()

    dataset_drift_metric = next(
        m for m in result["metrics"] if m["metric"] == "DatasetDriftMetric"
    )
    r = dataset_drift_metric["result"]

    target_drift_metric = next(
        m for m in result["metrics"] if m["metric"] == "ColumnDriftMetric"
    )
    t = target_drift_metric["result"]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Drift sur les features (attendu à 0% — vérification de cohérence)
        "feature_dataset_drift_detected": r["dataset_drift"],
        "feature_share_of_drifted": r["share_of_drifted_columns"],
        "n_drifted_features": r["number_of_drifted_columns"],
        "n_features": r["number_of_columns"],
        # Drift sur la target — LE signal qui compte pour ce projet
        "target_drift_detected": t["drift_detected"],
        "target_drift_score": t.get("drift_score"),
        "threshold_used": DRIFT_THRESHOLD,
    }


def should_trigger_retraining(summary: dict) -> bool:
    """
    Objectif dans le pipeline : décider du ré-entraînement en se basant
    sur le VRAI signal pertinent — le drift de la target (concept drift),
    pas sur le drift des features (qui ne se produira jamais ici).
    """
    return summary["target_drift_detected"]


def run():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    reference, current = load_reference_and_current()
    report = run_drift_report(reference, current)

    html_path = os.path.join(REPORTS_DIR, "drift_report.html")
    report.save_html(html_path)

    summary = extract_drift_summary(report)
    json_path = os.path.join(REPORTS_DIR, "drift_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    trigger = should_trigger_retraining(summary)

    print(f"[drift_check] Rapport HTML -> {html_path}")
    print(f"[drift_check] Résumé JSON  -> {json_path}")
    print(
        f"[drift_check] Drift features (calendaires) : "
        f"{summary['n_drifted_features']}/{summary['n_features']} — attendu proche de 0%"
    )
    print(
        f"[drift_check] Drift TARGET (PJME_MW) détecté : "
        f"{summary['target_drift_detected']} (score: {summary['target_drift_score']:.4f})"
    )
    print(f"[drift_check] Ré-entraînement recommandé : {trigger}")
    # Plus de exit(1) ici — le signal de drift vit uniquement dans le JSON,
    # que DVC ET le CI/CD peuvent tous les deux lire sans ambiguïté.


if __name__ == "__main__":
    run()
