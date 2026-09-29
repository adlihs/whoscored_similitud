# App Streamlit: similitud de jugadores

La app ofrece dos módulos: buscar jugadores similares y reentrenar/exportar el modelo desde un CSV actualizado.

## Ejecutar localmente

Desde la carpeta que contiene `streamlit_app.py`, `similarity_model.py` y `modelo_jugadores.joblib`:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Módulo de consulta

Selecciona la liga, el equipo, la posición y el jugador de referencia. Después define minutos mínimos, rango de edad, ligas de resultados y cantidad de candidatos. Los resultados se buscan entre todas las competiciones seleccionadas, usando la misma etiqueta de posición. Los porteros usan su propio conjunto de métricas.

La columna de distancia sirve para ordenar (menor indica más cercanía); el índice de similitud es una transformación de esa distancia y no representa una probabilidad.

## Módulo de actualización

Carga el CSV semanal en **Reentrenar modelo** y pulsa **Reentrenar y exportar modelo**. La app reconstruye ambos perfiles, guarda `modelo_jugadores.joblib` junto a la app y ofrece el archivo para descargar. Ese artefacto alimenta inmediatamente el módulo de consulta de la instancia actual.

Para actualizar otro despliegue, descarga el artefacto y reemplaza `modelo_jugadores.joblib` allí. En servicios con almacenamiento efímero, los cambios locales pueden perderse al reiniciar.
