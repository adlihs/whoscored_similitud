# Buscador de jugadores similares

Este primer flujo lee el CSV semanal, ajusta un escalador `StandardScaler` de
scikit-learn para cada perfil y exporta los datos procesados a un artefacto
`.joblib`. La búsqueda usa distancia euclídea sobre métricas por 90 minutos.

## 1. Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

## 2. Entrenar o reentrenar

Cada semana, sustituye el CSV por la extracción más reciente y ejecuta:

```bash
python similarity_model.py train \
  --csv /ruta/al/CSV_semanal.csv \
  --output modelo_jugadores.joblib
```

El comando vuelve a ajustar ambos perfiles con el conjunto actualizado y
reemplaza el artefacto indicado. Conserva el CSV de cada corte si necesitas
reproducir una versión anterior.

## 3. Consultar desde Python o Streamlit

```python
from similarity_model import load_model, find_similar

modelo = load_model("modelo_jugadores.joblib")
resultados = find_similar(
    modelo,
    league="BEL_JupiterLeague_2627",
    team="Kortrijk",
    position="ML",
    player="Thierry Ambrose",
    min_minutes=450,
    age_min=20,
    age_max=30,
    result_leagues=None,  # None significa todas las competiciones
    n_results=10,
)
```

Para poblar los controles de Streamlit, usa `modelo["leagues"]` y
`modelo["positions"]`. El control de equipo y jugador puede poblarse desde
`modelo["profiles"]["field"]["records"]` o
`modelo["profiles"]["goalkeeper"]["records"]`, filtrando sucesivamente por
liga, equipo y posición.

## Decisiones del primer modelo

- `GK` usa su perfil específico (atajadas, balones retenidos/despejados,
  juego aéreo, pases, toques y errores). Las otras posiciones usan métricas
  generales de campo.
- Los candidatos deben tener la misma etiqueta de posición que el jugador de
  referencia; por ejemplo, `FW` con `FW`. Los filtros de edad, minutos y ligas
  se aplican a candidatos en el momento de la consulta, sin reentrenar.
- Se comparan apariciones jugador-liga-equipo; un jugador presente en varias
  competiciones puede aparecer más de una vez.
- Se usan métricas `_p90` para reducir el efecto del tiempo jugado. No se
  incluyen edad, liga, equipo, nombre ni identificadores en la distancia.
- `similarity` es una transformación legible de la distancia (`1/(1+d)`), no
  una probabilidad ni una calificación de calidad.

La selección de métricas y el corte mínimo de minutos son decisiones de
producto que se pueden ajustar tras revisar los resultados con usuarios de
fútbol. El entrenamiento no aplica normalización por liga; las comparaciones
son entre competiciones tal como están sus métricas por 90 minutos.
