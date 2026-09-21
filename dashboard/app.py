"""Tableau de bord DriftGuard : suivi de la dérive et prévision de consommation.

Lancement : streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.predict import predict_batch

REPORTS_DIR = "reports"
PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", "data/processed")
TARGET_COLUMN = "PJME_MW"

REF_YEAR = "2016"
CUR_YEAR = "2017"

FONT = "Schibsted Grotesk, system-ui, sans-serif"
INK = "#16202E"
MUTED = "#5D6B7E"
RULE = "#D5DBE3"
REF_COLOR = "#9AA8BA"
CUR_COLOR = "#1746C7"

NARROW_SPACE = "\u202f"

st.set_page_config(page_title="DriftGuard", layout="wide")

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --ink: #16202E;
    --muted: #5D6B7E;
    --rule: #D5DBE3;
    --alert: #B42318;
    --warn: #9A5B00;
}

html, body, .stApp, .stApp p, .stApp label, .stApp button,
.stApp input, .stApp summary {
    font-family: 'Schibsted Grotesk', system-ui, sans-serif;
}

[data-testid="stHeader"], #MainMenu, footer { visibility: hidden; }

.block-container { max-width: 960px; padding-top: 2rem; }

.dg-bar {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--rule);
}
.dg-brand { font-weight: 700; letter-spacing: -0.01em; }
.dg-meta { color: var(--muted); font-size: 0.88rem; font-variant-numeric: tabular-nums; }

.dg-verdict {
    font-size: 2.1rem;
    font-weight: 600;
    line-height: 1.2;
    letter-spacing: -0.02em;
    max-width: 32ch;
    margin: 2.4rem 0 0.4rem 0;
}
.dg-verdict.alert { color: var(--alert); }
.dg-verdict.warn { color: var(--warn); }

.dg-h2 { font-size: 1.05rem; font-weight: 600; margin: 2.8rem 0 0.8rem 0; }

.stApp table.dg-table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
.stApp table.dg-table th,
.stApp table.dg-table td {
    border: 0;
    border-bottom: 1px solid var(--rule);
    background: transparent;
    text-align: left;
    padding: 0.7rem 0.5rem 0.7rem 0;
}
.stApp table.dg-table th { color: var(--muted); font-weight: 500; font-size: 0.85rem; }
.stApp table.dg-table td.status { font-weight: 600; }
.stApp table.dg-table td.status.alert { color: var(--alert); }
.stApp table.dg-table td.status.warn { color: var(--warn); }

.dg-note { color: var(--muted); font-size: 0.88rem; font-variant-numeric: tabular-nums; }
.dg-result { font-size: 1.25rem; margin-top: 1rem; font-variant-numeric: tabular-nums; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


def fmt_int(value: float) -> str:
    return f"{value:,.0f}".replace(",", NARROW_SPACE)


def fmt_signed(value: float) -> str:
    return f"{value:+,.0f}".replace(",", NARROW_SPACE)


def fmt_pct(value: float) -> str:
    return f"{value * 100:.0f}{NARROW_SPACE}%"


def fmt_score(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def fmt_timestamp(raw) -> str:
    try:
        return pd.to_datetime(raw).strftime("%d/%m/%Y à %H:%M")
    except (ValueError, TypeError):
        return str(raw)


def section(title: str) -> None:
    st.markdown(f'<h2 class="dg-h2">{title}</h2>', unsafe_allow_html=True)


def status_cell(drifted: bool, level: str = "alert") -> str:
    if drifted:
        return f'<td class="status {level}">Dérive</td>'
    return '<td class="status">Stable</td>'


summary_path = os.path.join(REPORTS_DIR, "drift_summary.json")

if not os.path.exists(summary_path):
    st.warning("Aucune analyse de drift disponible.")
    st.code("python -m src.monitoring.drift_check", language="bash")
    st.stop()

with open(summary_path, encoding="utf-8") as f:
    summary = json.load(f)

feature_share = summary["feature_share_of_drifted"]
feature_drift = feature_share > 0
target_drift = bool(summary["target_drift_detected"])

st.markdown(
    '<div class="dg-bar"><span class="dg-brand">DriftGuard</span>'
    f'<span class="dg-meta">Dernière analyse le {fmt_timestamp(summary["timestamp"])}</span></div>',
    unsafe_allow_html=True,
)

if target_drift:
    verdict = "La consommation réelle a dérivé par rapport à la référence."
elif feature_drift:
    verdict = "Certaines variables d'entrée ont dérivé, la consommation réelle reste stable."
else:
    verdict = "Aucune dérive détectée."

verdict_class = " alert" if target_drift else " warn" if feature_drift else ""
st.markdown(
    f'<div class="dg-verdict{verdict_class}">{verdict}</div>',
    unsafe_allow_html=True,
)

section("Indicateurs")

rows = (
    "<tr><td>Variables d'entrée</td>"
    f"<td>{fmt_pct(feature_share)} des variables ont dérivé</td>"
    f"{status_cell(feature_drift, 'warn')}</tr>"
    f"<tr><td>Consommation réelle ({TARGET_COLUMN})</td>"
    f"<td>Score de dérive {fmt_score(summary['target_drift_score'])}</td>"
    f"{status_cell(target_drift)}</tr>"
)
st.markdown(
    '<table class="dg-table"><thead><tr>'
    "<th>Indicateur</th><th>Mesure</th><th>Statut</th>"
    f"</tr></thead><tbody>{rows}</tbody></table>",
    unsafe_allow_html=True,
)

section("Distribution de la consommation")

try:
    reference = pd.read_csv(os.path.join(PROCESSED_DIR, "reference.csv"))
    current = pd.read_csv(os.path.join(PROCESSED_DIR, "current.csv"))

    ref_values = reference[TARGET_COLUMN]
    cur_values = current[TARGET_COLUMN]

    # Mêmes intervalles pour les deux séries, sinon les barres ne sont pas comparables.
    low = min(ref_values.min(), cur_values.min())
    high = max(ref_values.max(), cur_values.max())
    bins = dict(start=low, end=high, size=(high - low) / 50)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=ref_values, name=f"{REF_YEAR} (référence)", histnorm="percent",
            xbins=bins, marker=dict(color=REF_COLOR, line=dict(width=0)), opacity=0.75,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=cur_values, name=f"{CUR_YEAR} (actuelle)", histnorm="percent",
            xbins=bins, marker=dict(color=CUR_COLOR, line=dict(width=0)), opacity=0.6,
        )
    )

    ref_mean, cur_mean = ref_values.mean(), cur_values.mean()
    ref_side, cur_side = (
        ("top left", "top right") if ref_mean <= cur_mean else ("top right", "top left")
    )
    fig.add_vline(
        x=ref_mean, line_width=1.5, line_dash="dot", line_color=MUTED,
        annotation_text=f"Référence {REF_YEAR} : moyenne {fmt_int(ref_mean)} MW",
        annotation_position=ref_side, annotation_font=dict(color=MUTED, size=12),
    )
    fig.add_vline(
        x=cur_mean, line_width=1.5, line_dash="dot", line_color=CUR_COLOR,
        annotation_text=f"Actuelle {CUR_YEAR} : moyenne {fmt_int(cur_mean)} MW",
        annotation_position=cur_side, annotation_font=dict(color=CUR_COLOR, size=12),
    )

    fig.update_layout(
        barmode="overlay",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=13),
        margin=dict(t=30, b=10, l=10, r=10),
        xaxis=dict(title="Consommation (MW)", showgrid=False, linecolor=RULE,
                   ticks="outside", tickcolor=RULE),
        yaxis=dict(title="Part des observations (%)", gridcolor="#E3E7ED", zeroline=False),
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="dg-note">Écart des moyennes : {fmt_signed(cur_mean - ref_mean)} MW</div>',
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    st.warning("Données introuvables.")
    st.code("python -m src.data.preprocessing", language="bash")

section("Prévision ponctuelle")

col_date, col_hour = st.columns(2)
selected_date = col_date.date_input("Date")
selected_hour = col_hour.slider("Heure", 0, 23, 12)

if st.button("Calculer la prévision"):
    dt_str = f"{selected_date}T{selected_hour:02d}:00:00"
    try:
        result_df = predict_batch([dt_str])
        prediction = result_df["predicted_consumption_mw"].iloc[0]
        st.markdown(
            f'<p class="dg-result"><strong>{fmt_int(prediction)} MW</strong> '
            f"prévus le {selected_date:%d/%m/%Y} à {selected_hour} h.</p>",
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.warning("Modèle introuvable.")
        st.code("python -m src.models.train", language="bash")

st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)

with st.expander("Rapport Evidently complet"):
    html_path = os.path.join(REPORTS_DIR, "drift_report.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=800, scrolling=True)
    else:
        st.write("Rapport introuvable.")