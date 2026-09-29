"""Aplicación Streamlit para consultar y actualizar el modelo de similitud."""

from __future__ import annotations

import base64
import hmac
import io
import os
import tempfile
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import joblib

from similarity_model import (
    FIELD_FEATURES,
    GOALKEEPER_FEATURES,
    IDENTITY_COLUMNS,
    find_similar,
    train_model,
)


GITHUB_API = "https://api.github.com"
GITHUB_TIMEOUT = 60


def github_settings():
    """Load repository coordinates and credentials from Streamlit secrets."""
    try:
        config = st.secrets.get("github", {})
        admin = st.secrets.get("admin", {})
    except Exception:
        config, admin = {}, {}
    return {
        "repository": config.get("repository", "adlihs/whoscored_similitud"),
        "model_branch": config.get("model_branch", "model-artifacts"),
        "token": config.get("token") or os.environ.get("GITHUB_TOKEN"),
        "admin_password": admin.get("password") or os.environ.get("STREAMLIT_ADMIN_PASSWORD"),
    }


def github_request(method, endpoint, *, token=None, **kwargs):
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.request(
        method,
        f"{GITHUB_API}{endpoint}",
        headers=headers,
        timeout=GITHUB_TIMEOUT,
        **kwargs,
    )
    if not response.ok:
        try:
            detail = response.json().get("message", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"GitHub respondió {response.status_code}: {detail}")
    return response


def latest_model_commit(repository, branch, token):
    ref = github_request(
        "GET", f"/repos/{repository}/git/ref/heads/{branch}", token=token
    ).json()
    return ref["object"]["sha"]


@st.cache_resource(show_spinner="Cargando el modelo más reciente desde GitHub…")
def load_remote_artifact(repository, commit_sha, token):
    """Load one immutable artifact revision; commit SHA keys the Streamlit cache."""
    headers = {"Accept": "application/vnd.github.raw+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(
        f"{GITHUB_API}/repos/{repository}/contents/modelo_jugadores.joblib",
        params={"ref": commit_sha},
        headers=headers,
        timeout=GITHUB_TIMEOUT,
    )
    if not response.ok:
        try:
            detail = response.json().get("message", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"No se pudo descargar el artefacto ({response.status_code}): {detail}")
    artifact = joblib.load(io.BytesIO(response.content))
    if artifact.get("version") != 1:
        raise RuntimeError("La versión del artefacto no es compatible con esta app.")
    return artifact


def publish_model_to_github(model_path, repository, branch, token):
    """Commit the new binary artifact to the dedicated model branch."""
    if not token:
        raise RuntimeError("Falta el token de GitHub. Configura [github].token en Streamlit Secrets.")
    ref = github_request(
        "GET", f"/repos/{repository}/git/ref/heads/{branch}", token=token
    ).json()
    parent_sha = ref["object"]["sha"]
    parent = github_request(
        "GET", f"/repos/{repository}/git/commits/{parent_sha}", token=token
    ).json()

    encoded_model = base64.b64encode(Path(model_path).read_bytes()).decode("ascii")
    blob = github_request(
        "POST",
        f"/repos/{repository}/git/blobs",
        token=token,
        json={"content": encoded_model, "encoding": "base64"},
    ).json()
    tree = github_request(
        "POST",
        f"/repos/{repository}/git/trees",
        token=token,
        json={
            "base_tree": parent["tree"]["sha"],
            "tree": [
                {
                    "path": "modelo_jugadores.joblib",
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob["sha"],
                }
            ],
        },
    ).json()
    commit = github_request(
        "POST",
        f"/repos/{repository}/git/commits",
        token=token,
        json={
            "message": "Update trained player similarity model",
            "tree": tree["sha"],
            "parents": [parent_sha],
        },
    ).json()
    github_request(
        "PATCH",
        f"/repos/{repository}/git/refs/heads/{branch}",
        token=token,
        json={"sha": commit["sha"], "force": False},
    )
    return commit["sha"]

st.set_page_config(page_title="Similitud de jugadores", page_icon="⚽", layout="wide")
st.title("Similitud de jugadores")
st.caption("WhoScored · Consulta perfiles y actualiza el modelo con nuevos datos")


def read_artifact(settings):
    try:
        commit_sha = latest_model_commit(
            settings["repository"], settings["model_branch"], settings["token"]
        )
        model = load_remote_artifact(settings["repository"], commit_sha, settings["token"])
        return model, commit_sha, None
    except Exception as exc:
        return None, None, str(exc)


settings = github_settings()
artifact, artifact_sha, artifact_error = read_artifact(settings)
page = st.sidebar.radio(
    "Módulo",
    ["Consultar similitudes", "Reentrenar modelo"],
    help="La consulta usa el último artefacto de GitHub. El reentrenamiento publica una nueva versión en la rama model-artifacts.",
)


if page == "Consultar similitudes":
    st.header("Buscar jugadores similares")
    if artifact is None:
        st.error(f"No pude cargar el modelo desde GitHub: {artifact_error}")
        st.warning("Comprueba que exista la rama `model-artifacts` y que contenga `modelo_jugadores.joblib`.")
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

    search_col, help_col = st.columns([0.92, 0.08], vertical_alignment="bottom")
    with search_col:
        run_search = st.button("Buscar similitudes", type="primary", disabled=not players)
    with help_col:
        st.markdown(
            "<style>.st-key-results_similarity_help [data-testid='stIconMaterial'] "
            "{font-size: 1.35rem !important;}</style>",
            unsafe_allow_html=True,
        )
        st.button(
            " ",
            key="results_similarity_help",
            icon=":material/info:",
            type="tertiary",
            help=(
                "**Distancia:** mide qué tan diferentes son las estadísticas del jugador y las del candidato. "
                "Un valor menor significa un perfil más cercano; por ejemplo, 1.2 es más cercano que 2.7. "
                "\n\n**Índice de similitud:** resume esa cercanía en una escala de 0 a 1. Un valor mayor indica "
                "más similitud. No es un porcentaje ni una probabilidad de que el jugador rinda igual o sea mejor."
            ),
        )

    if run_search:
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
        st.write(f"Versión GitHub: `{artifact_sha[:12]}` · rama `{settings['model_branch']}`")
        st.write(f"Métricas de campo: {len(artifact['profiles']['field']['features'])}")
        st.write(f"Métricas de portero: {len(artifact['profiles']['goalkeeper']['features'])}")
        st.write("Los filtros de resultados no requieren reentrenar el modelo.")


else:
    st.header("Cargar datos y reentrenar")
    admin_password = settings["admin_password"]
    if not admin_password:
        st.error("El módulo de reentrenamiento está bloqueado. Configura [admin].password en Streamlit Secrets.")
        st.stop()
    if not st.session_state.get("admin_authenticated", False):
        entered_password = st.text_input("Contraseña de administrador", type="password")
        if st.button("Desbloquear módulo"):
            if hmac.compare_digest(entered_password, admin_password):
                st.session_state["admin_authenticated"] = True
                st.rerun()
            st.error("Contraseña incorrecta.")
        st.stop()

    st.write(
        "Carga el CSV actualizado. La aplicación ajustará de nuevo los perfiles de campo y portero, "
        "publicará el artefacto en GitHub y te permitirá descargar una copia."
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

    if uploaded is not None and st.button("Reentrenar y publicar modelo", type="primary"):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                csv_path = Path(tmp_dir) / "jugadores.csv"
                staged_model = Path(tmp_dir) / "modelo_jugadores.joblib"
                csv_path.write_bytes(uploaded.getvalue())
                new_artifact = train_model(csv_path, staged_model)
                commit_sha = publish_model_to_github(
                    staged_model,
                    settings["repository"],
                    settings["model_branch"],
                    settings["token"],
                )
                st.session_state["trained_model_bytes"] = staged_model.read_bytes()
            st.session_state["trained_model_name"] = "modelo_jugadores.joblib"
            st.session_state["latest_model_sha"] = commit_sha
            st.session_state["latest_model_artifact"] = new_artifact
            st.success(
                f"Modelo actualizado: {new_artifact['source_rows']:,} apariciones, "
                f"{len(new_artifact['leagues'])} ligas y {len(new_artifact['positions'])} posiciones. "
                f"Publicado en `{settings['model_branch']}` ({commit_sha[:12]})."
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
        st.caption("El módulo de consulta carga automáticamente el artefacto más reciente desde GitHub.")

    current_artifact = st.session_state.get("latest_model_artifact", artifact)
    if current_artifact is not None:
        st.info(f"Modelo actual: {current_artifact['source_rows']:,} apariciones · {len(current_artifact['leagues'])} ligas.")

    st.caption(f"Repositorio: `{settings['repository']}` · rama de artefactos: `{settings['model_branch']}`")
