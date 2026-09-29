"""Entrena y consulta un buscador de jugadores similares con scikit-learn.

Uso:
    python similarity_model.py train --csv jugadores.csv --output modelo_jugadores.joblib

El artefacto exportado contiene los escaladores, las matrices de perfiles y
los registros necesarios para consultar desde una aplicación Streamlit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


VERSION = 1
IDENTITY_COLUMNS = [
    "league_folder", "team_id", "team_name", "player_id", "player_name",
    "position", "age", "played_minutes",
]

# Conteos por 90 minutos para perfiles de jugadores de campo. Las variables
# derivadas de estas (porcentajes/ratios) se omiten para no contar dos veces
# la misma señal. Agrega o quita métricas según el objetivo del producto.
FIELD_FEATURES = [
    "aerialsTotal_p90", "aerialsWon_p90", "clearances_p90",
    "defensiveAerials_p90", "dispossessed_p90", "dribbledPast_p90",
    "dribblesAttempted_p90", "dribblesLost_p90", "dribblesWon_p90",
    "errors_p90", "foulsCommited_p90", "interceptions_p90",
    "offensiveAerials_p90", "offsidesCaught_p90", "passesAccurate_p90",
    "passesKey_p90", "passesTotal_p90", "possession_p90",
    "shotsBlocked_p90", "shotsOffTarget_p90", "shotsOnPost_p90",
    "shotsOnTarget_p90", "shotsTotal_p90", "tackleSuccessful_p90",
    "tackleUnsuccesful_p90", "tacklesTotal_p90", "touches_p90",
]

# Perfil específico para porteros; incluye distribución con pies y acciones
# aéreas además de acciones de atajada/retención.
GOALKEEPER_FEATURES = [
    "totalSaves_p90", "collected_p90", "claimsHigh_p90",
    "parriedDanger_p90", "parriedSafe_p90", "clearances_p90",
    "aerialsTotal_p90", "aerialsWon_p90", "passesAccurate_p90",
    "passesTotal_p90", "touches_p90", "errors_p90",
]


def _valid_features(columns: pd.Index, features: list[str], profile: str) -> list[str]:
    absent = sorted(set(features) - set(columns))
    if absent:
        raise ValueError(f"Faltan columnas para el perfil {profile}: {absent}")
    return features


def train_model(csv_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Lee una nueva extracción CSV, ajusta perfiles y guarda el artefacto."""
    df = pd.read_csv(csv_path)
    required = set(IDENTITY_COLUMNS)
    absent = sorted(required - set(df.columns))
    if absent:
        raise ValueError(f"Faltan columnas obligatorias: {absent}")

    features = {
        "field": _valid_features(df.columns, FIELD_FEATURES, "field"),
        "goalkeeper": _valid_features(df.columns, GOALKEEPER_FEATURES, "goalkeeper"),
    }
    # Una fila representa un jugador en una competición. Evita perfiles
    # duplicados accidentales para la misma liga, equipo y jugador.
    df = df.drop_duplicates(["league_folder", "team_id", "player_id"]).copy()
    df["position"] = df["position"].astype(str).str.strip().str.upper()
    for cols in features.values():
        df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
        df[cols] = df[cols].fillna(df[cols].median()).fillna(0)

    profiles: dict[str, dict[str, Any]] = {}
    for name, cols in features.items():
        mask = df["position"].eq("GK") if name == "goalkeeper" else df["position"].ne("GK")
        records = df.loc[mask, IDENTITY_COLUMNS].reset_index(drop=True)
        scaler = StandardScaler()
        matrix = scaler.fit_transform(df.loc[mask, cols].astype(float))
        profiles[name] = {"features": cols, "scaler": scaler, "matrix": matrix, "records": records}

    artifact = {
        "version": VERSION,
        "profiles": profiles,
        "positions": sorted(df["position"].dropna().unique().tolist()),
        "leagues": sorted(df["league_folder"].dropna().astype(str).unique().tolist()),
        "source_rows": int(len(df)),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    return artifact


def find_similar(
    artifact: dict[str, Any], *, league: str, team: str, position: str,
    player: str, min_minutes: int = 0, age_min: int | None = None,
    age_max: int | None = None, result_leagues: list[str] | None = None,
    n_results: int = 10,
) -> pd.DataFrame:
    """Devuelve vecinos del jugador seleccionado tras aplicar filtros."""
    position = position.strip().upper()
    profile_name = "goalkeeper" if position == "GK" else "field"
    profile = artifact["profiles"][profile_name]
    records: pd.DataFrame = profile["records"]
    target_mask = (
        records["league_folder"].astype(str).eq(str(league))
        & records["team_name"].astype(str).eq(str(team))
        & records["position"].eq(position)
        & records["player_name"].astype(str).eq(str(player))
    )
    target_indices = np.flatnonzero(target_mask.to_numpy())
    if len(target_indices) == 0:
        raise ValueError("No se encontró el jugador con esa liga, equipo y posición.")
    target_i = int(target_indices[0])

    candidates = records.copy()
    keep = candidates["position"].eq(position)
    keep &= candidates["played_minutes"].ge(min_minutes)
    if age_min is not None:
        keep &= candidates["age"].ge(age_min)
    if age_max is not None:
        keep &= candidates["age"].le(age_max)
    if result_leagues:
        keep &= candidates["league_folder"].astype(str).isin([str(x) for x in result_leagues])
    keep.iloc[target_i] = False
    candidate_indices = np.flatnonzero(keep.to_numpy())
    if len(candidate_indices) == 0:
        return pd.DataFrame(columns=IDENTITY_COLUMNS + ["distance", "similarity"])

    distances = pairwise_distances(
        profile["matrix"][candidate_indices],
        profile["matrix"][target_i:target_i + 1],
        metric="euclidean",
    ).ravel()
    result = candidates.iloc[candidate_indices].copy()
    result["distance"] = distances
    # Escala monotónica de lectura; no es una probabilidad.
    result["similarity"] = 1 / (1 + result["distance"])
    return result.sort_values("distance").head(n_results).reset_index(drop=True)


def load_model(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if artifact.get("version") != VERSION:
        raise ValueError("La versión del artefacto no es compatible con este código.")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train", help="entrena/exporta desde un CSV actualizado")
    train_parser.add_argument("--csv", required=True, help="ruta al CSV semanal")
    train_parser.add_argument("--output", default="modelo_jugadores.joblib")
    args = parser.parse_args()
    if args.command == "train":
        artifact = train_model(args.csv, args.output)
        print(
            f"Modelo guardado: {args.output}\n"
            f"Filas: {artifact['source_rows']} | ligas: {len(artifact['leagues'])} "
            f"| posiciones: {len(artifact['positions'])}"
        )


if __name__ == "__main__":
    main()
