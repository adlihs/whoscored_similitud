# WhoScored: similitud de jugadores

App de Streamlit para consultar jugadores similares y reentrenar el modelo desde un CSV actualizado.

## Iniciar la app

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Configura un token GitHub (Contents: read/write) y una contraseña de administrador en Streamlit Secrets. La app guarda artefactos nuevos en la rama `model-artifacts` y carga la versión más reciente desde allí. Nunca agregues secretos al repositorio.

Consulta [README_streamlit.md](README_streamlit.md) para la configuración y [README_modelo_similitud.md](README_modelo_similitud.md) para los perfiles y la interfaz Python.
