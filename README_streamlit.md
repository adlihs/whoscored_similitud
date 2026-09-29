# App Streamlit: similitud de jugadores

La app ofrece dos módulos: consultar jugadores similares y reentrenar/exportar el modelo desde un CSV actualizado. El artefacto vigente vive en la rama `model-artifacts` del repositorio; la app consulta esa rama al iniciar cada interacción y cachea el modelo por SHA de commit.

## Ejecutar localmente

Desde la carpeta que contiene `streamlit_app.py` y `similarity_model.py`:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Configurar secretos

La app necesita un token de GitHub con permiso **Contents: Read and write**, limitado a este repositorio. El módulo de reentrenamiento requiere además una contraseña de administrador. En Streamlit Community Cloud, configura ambos valores en **App settings → Secrets**. Para desarrollo local, guárdalos en `.streamlit/secrets.toml` (ese archivo no se debe subir a GitHub):

```toml
[github]
repository = "adlihs/whoscored_similitud"
model_branch = "model-artifacts"
token = "TU_TOKEN_FINE_GRAINED"

[admin]
password = "UNA_CONTRASENA_LARGA"
```

No guardes el token ni la contraseña en el código o en el repositorio. El token solo se usa desde el servidor de Streamlit para leer y publicar el artefacto.

## Módulo de consulta

Selecciona liga, equipo, posición y jugador de referencia. Después define minutos mínimos, rango de edad, ligas de resultados y cantidad de candidatos. La búsqueda usa jugadores de la misma etiqueta de posición en las competiciones seleccionadas. Los porteros usan su perfil específico.

La distancia ordena los resultados (menor indica mayor cercanía); el índice de similitud es una transformación de la distancia y no representa una probabilidad.

## Módulo de actualización

Desbloquea el módulo con la contraseña de administrador, carga el CSV semanal y pulsa **Reentrenar y publicar modelo**. La app reconstruye ambos perfiles, crea un commit en `model-artifacts` y ofrece descargar una copia `.joblib`. El módulo de consulta detecta y carga esa versión desde GitHub.

La app desplegada debe tener el token en sus secretos. Protege también el acceso general a la app si no quieres que otras personas consulten los datos; solo la publicación del artefacto requiere la contraseña administrativa.

## Ayuda para interpretar resultados

El menú **Ayuda** explica en lenguaje sencillo qué significa cada columna: una distancia más baja representa estadísticas más cercanas; un índice más alto indica mayor similitud. El índice no es un porcentaje ni una probabilidad de rendimiento.
