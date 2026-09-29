# WhoScored: similitud de jugadores

App de Streamlit para consultar jugadores similares y reentrenar/exportar el modelo desde un CSV actualizado.

## Iniciar

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Consulta [README_streamlit.md](README_streamlit.md) para el uso de los dos módulos y [README_modelo_similitud.md](README_modelo_similitud.md) para el flujo de entrenamiento y la interfaz Python.

El archivo `modelo_jugadores.joblib` es el artefacto de muestra. La app permite sustituirlo mediante el CSV actualizado.
