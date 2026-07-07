"""
pages/dashboard.py — Dashboard de Observabilidad del Agente Starken
EV3 - ISY0101 Ingeniería de Soluciones con IA
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="Dashboard Observabilidad · Starken",
    page_icon="📊",
    layout="wide"
)

# ── Cargar datos ──────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "metricas.jsonl")


def cargar_datos() -> pd.DataFrame:
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()

    registros = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                try:
                    registros.append(json.loads(linea))
                except json.JSONDecodeError:
                    continue

    if not registros:
        return pd.DataFrame()

    df = pd.DataFrame(registros)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["error"] = df["error"].astype(bool)
    return df


# ── Header ────────────────────────────────────────────────────────
st.title("📊 Dashboard de Observabilidad — Agente Starken")

if st.button("🔄 Actualizar datos"):
    st.rerun()

df = cargar_datos()

if df.empty:
    st.warning("⚠️ No hay datos aún. Ejecuta el agente (`app.py`) y realiza algunas consultas primero.")
    st.info("Los registros se guardan automáticamente en `logs/metricas.jsonl` cada vez que el agente responde.")
    st.stop()


# ── KPIs principales ──────────────────────────────────────────────
st.markdown("## 📈 Métricas Globales")

total = len(df)
errores = df["error"].sum()
precision = round((total - errores) / total * 100, 1)
lat_prom = round(df["latencia_seg"].mean(), 3)
lat_max = round(df["latencia_seg"].max(), 3)
tasa_error = round(errores / total * 100, 1)

grupos = df.groupby(df["pregunta"].str.lower().str.strip())["respuesta"].nunique()
consistencia = round((grupos == 1).sum() / len(grupos) * 100, 1) if len(grupos) > 0 else 100.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total ejecuciones", total)
col2.metric("Precisión (sin error)", f"{precision}%")
col3.metric("Latencia promedio", f"{lat_prom}s")
col4.metric("Tasa de error", f"{tasa_error}%", delta=f"{int(errores)} errores", delta_color="inverse")
col5.metric("Consistencia", f"{consistencia}%")

st.divider()

# ── Fila 1: Latencia en el tiempo + por herramienta ──────────────
st.markdown("## ⏱️ Latencia")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### Latencia por ejecución (cronológico)")
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    df_sorted.index += 1
    st.line_chart(
        df_sorted[["latencia_seg"]].rename(columns={"latencia_seg": "Latencia (seg)"}),
        use_container_width=True
    )

with col_b:
    st.markdown("### Latencia promedio por herramienta")

    lat_tool = (
        df.groupby("herramienta")["latencia_seg"]
        .agg(["mean", "count", "max", "min"])
        .round(4)
        .reset_index()
        .rename(columns={
            "herramienta": "Herramienta",
            "mean": "Promedio (s)",
            "count": "Ejecuciones",
            "max": "Máxima (s)",
            "min": "Mínima (s)"
        })
    )

    st.bar_chart(
        lat_tool.set_index("Herramienta")[["Promedio (s)"]],
        use_container_width=True
    )

    st.dataframe(
        lat_tool,
        use_container_width=True,
        hide_index=True
    )
    
st.divider()

# ── Fila 2: Distribución de herramientas ──────────────────────────
st.markdown("## 🔧 Distribución de herramientas usadas")

conteo = df["herramienta"].value_counts().reset_index()
conteo.columns = ["Herramienta", "Llamadas"]
conteo = conteo.set_index("Herramienta")

st.bar_chart(conteo, use_container_width=True)

st.divider()

# ── Tabla de últimas ejecuciones ──────────────────────────────────
st.markdown("## 📋 Registro de ejecuciones recientes")

max_n = min(100, total)

if max_n <= 1:
    n = 1
    st.caption("Mostrando la única ejecución registrada.")
elif max_n <= 5:
    n = max_n
    st.caption(f"Mostrando las últimas {n} ejecuciones.")
else:
    n = st.slider(
        "Mostrar últimas N ejecuciones",
        min_value=5,
        max_value=max_n,
        value=min(20, max_n)
    )

df_tabla = (
    df.sort_values("timestamp", ascending=False)
    .head(n)[["timestamp", "pregunta", "herramienta", "latencia_seg", "error", "tokens_aprox", "detalle_error"]]
    .rename(columns={
        "timestamp": "Timestamp",
        "pregunta": "Pregunta",
        "herramienta": "Herramienta",
        "latencia_seg": "Latencia (s)",
        "error": "Error",
        "tokens_aprox": "Tokens aprox.",
        "detalle_error": "Detalle error",
    })
    .reset_index(drop=True)
)

st.dataframe(
    df_tabla,
    use_container_width=True,
    hide_index=True
)

# ── Análisis de anomalías ─────────────────────────────────────────
st.divider()
st.markdown("## 🔍 Detección de anomalías")

umbral_lat = df["latencia_seg"].mean() + 2 * df["latencia_seg"].std()
anomalias = df[df["latencia_seg"] > umbral_lat]

if anomalias.empty:
    st.success("✅ No se detectaron anomalías de latencia (ninguna ejecución supera media + 2σ).")
else:
    st.warning(f"⚠️ Se detectaron **{len(anomalias)}** ejecuciones con latencia anómala (>{umbral_lat:.2f}s):")
    st.dataframe(
        anomalias[["timestamp", "pregunta", "herramienta", "latencia_seg"]]
        .rename(columns={
            "timestamp": "Timestamp",
            "pregunta": "Pregunta",
            "herramienta": "Herramienta",
            "latencia_seg": "Latencia (s)"
        })
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
    f"Datos desde `logs/metricas.jsonl` · {total} registros totales"
)