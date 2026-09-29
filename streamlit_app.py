"""Aplicación Streamlit para consultar y actualizar el modelo de similitud."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from similarity_model import (
    FIELD_FEATURES,
    GOALKEEPER_FEATURES,
    IDENTITY_COLUMNS,
    find_similar,
    load_model,
    train_model,
)


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("PLAYER_MODEL_PATH", APP_DIR / "modelo_jugadores.joblib"))

st.set_page_config(page_title="Similitud de jugadores", page_icon="⚽", layout="wide")
st.title("Similitud de jugadores")
st.caption("WhoScored · Consulta perfiles y actualiza el modelo con nuevos datos")


def read_artifact():
    if not MODEL_PATH.exists():
        return None
    try:
        return load_model(MODEL_PATH)
    except Exception as exc:  # archivo corrupto o creado con versión incompatible
        st.error(f"No pude cargar el modelo en `{MODEL_PATH}`: {exc}")
        return None


artifact = read_artifact()
page = st.sidebar.radio(
    "Módulo",
    ["Consultar similitudes", "Reentrenar modelo"],
    help="La consulta usa el modelo actual. El reentrenamiento reemplaza el archivo local del modelo.",
)


if page == "Consultar similitudes":
    st.header("Buscar jugadores similares")
    if artifact is None:
        st.warning("No hay un modelo disponible. Ve a **Reentrenar modelo** y carga un CSV para generarlo.")
        st.stop()

    field_records = artifact["profiles"]["field"]["records"]
    goalkeeper_records = artifact["profiles"]["goalkeeper"]["records"]
    records = pd.concat([field_records, goalkeeper_records], ignore_index=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        league = st.selectbox("Liga de referencia", artifact["leagues"])
    league_records = records[records["league_folder"].astype(str).eq(str(league))]
    with c2:
        teams = sorted(league_records["team_name"].dropna().astype(str).unique())
        team = st.selectbox("Equipo", teams, disabled=not teams)
    team_records = league_records[league_records["team_name"].astype(str).eq(str(team))]
    with c3:
        positions = sorted(team_records["position"].dropna().astype(str).unique())
        position = st.selectbox("Posición", positions, disabled=not positions)
    player_records = team_records[team_records["position"].eq(position)]
    with c4:
        players = sorted(player_records["player_name"].dropna().astype(str).unique())
        player = st.selectbox("Jugador", players, disabled=not players)

    st.subheader("Filtros de resultados")
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        min_minutes = st.number_input("Mínimo de minutos", min_value=0, max_value=20000, value=450, step=90)
    with f2:
        age_min, age_max = st.slider("Rango de edad", min_value=15, max_value=50, value=(15, 50))
    with f3:
        result_leagues = st.multiselect(
            "Ligas donde buscar",
            options=artifact["leagues"],
            default=artifact["leagues"],
            help="Selecciona una o varias ligas. Todas vienen seleccionadas inicialmente.",
        )
    n_results = st.slider("Cantidad de resultados", min_value=1, max_value=50, value=10)

    if st.button("Buscar similitudes", type="primary", disabled=not players):
        try:
            results = find_similar(
                artifact,
                league=league,
                team=team,
                position=position,
                player=player,
                min_minutes=int(min_minutes),
                age_min=int(age_min),
                age_max=int(age_max),
                result_leagues=result_leagues,
                n_results=int(n_results),
            )
            if results.empty:
                st.info("No hay candidatos con esos filtros. Amplía las ligas o ajusta edad/minutos.")
            else:
                st.markdown(f"### Resultados para **{player}** · {team} · {league} · {position}")
                shown = results[
                    ["player_name", "team_name", "league_folder", "position", "age", "played_minutes", "distance", "similarity"]
                ].rename(
                    columns={
                        "player_name": "Jugador",
                        "team_name": "Equipo",
                        "league_folder": "Liga",
                        "position": "Posición",
                        "age": "Edad",
                        "played_minutes": "Minutos",
                        "distance": "Distancia (menor = más parecido)",
                        "similarity": "Índice de similitud",
                    }
                )
                st.dataframe(
                    shown.style.format({"Distancia (menor = más parecido)": "{:.3f}", "Índice de similitud": "{:.3f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("El índice ordena los resultados; no es una probabilidad. Se comparan jugadores de la misma posición.")
        except ValueError as exc:
            st.error(str(exc))

    with st.expander("Información del modelo"):
        st.write(f"Apariciones incluidas: {artifact['source_rows']:,}")
        st.write(f"Métricas de campo: {len(artifact['profiles']['field']['features'])}")
        st.write(f"Métricas de portero: {len(artifact['profiles']['goalkeeper']['features'])}")
        st.write("Los filtros de resultados no requieren reentrenar el modelo.")


else:
    st.header("Cargar datos y reentrenar")
    st.write(
        "Carga el CSV actualizado. La aplicación ajustará de nuevo los perfiles de campo y portero, "
        "guardará el artefacto y te permitirá descargarlo."
    )
    uploaded = st.file_uploader("CSV de jugadores", type=["csv"], help="Debe conservar las columnas del dataset WhoScored.")

    if uploaded is not None:
        try:
            uploaded.seek(0)
            preview = pd.read_csv(uploaded, nrows=8)
            uploaded.seek(0)
            st.caption(f"Muestra de datos · {len(preview.columns)} columnas")
            st.dataframe(preview, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"No pude leer el CSV: {exc}")

        with st.expander("Columnas requeridas por el flujo"):
            st.write("**Identificación y filtros:** " + ", ".join(IDENTITY_COLUMNS))
            st.write("**Métricas de campo:** " + ", ".join(FIELD_FEATURES))
            st.write("**Métricas de portero:** " + ", ".join(GOALKEEPER_FEATURES))

    if uploaded is not None and st.button("Reentrenar y exportar modelo", type="primary"):
        try:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                csv_path = Path(tmp_dir) / "jugadores.csv"
                staged_model = Path(tmp_dir) / "modelo_jugadores.joblib"
                csv_path.write_bytes(uploaded.getvalue())
                new_artifact = train_model(csv_path, staged_model)
                os.replace(staged_model, MODEL_PATH)

            st.session_state["trained_model_bytes"] = MODEL_PATH.read_bytes()
            st.session_state["trained_model_name"] = MODEL_PATH.name
            st.success(
                f"Modelo actualizado: {new_artifact['source_rows']:,} apariciones, "
                f"{len(new_artifact['leagues'])} ligas y {len(new_artifact['positions'])} posiciones."
            )
        except Exception as exc:
            st.error(f"No se pudo reentrenar el modelo: {exc}")

    if "trained_model_bytes" in st.session_state:
        st.download_button(
            "Descargar modelo entrenado (.joblib)",
            data=st.session_state["trained_model_bytes"],
            file_name=st.session_state["trained_model_name"],
            mime="application/octet-stream",
            type="secondary",
        )
        st.caption(f"Artefacto guardado en `{MODEL_PATH}` para que lo use el módulo de consulta.")

    if artifact is not None:
        st.info(f"Modelo actualmente cargado: {artifact['source_rows']:,} apariciones · {len(artifact['leagues'])} ligas.")

    st.warning(
        "En alojamientos efímeros, como Streamlit Community Cloud, los archivos locales pueden perderse al reiniciar o desplegar. "
        "Descarga y conserva el `.joblib` generado; para usarlo en otro despliegue, reemplaza el artefacto del repositorio."
    )
