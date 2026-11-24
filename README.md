# Recomendador de álbumes

Aplicación web que recomienda álbumes a partir de los gustos de una persona.

La persona selecciona entre uno y cuatro álbumes conocidos y el sistema devuelve diez recomendaciones similares. Utiliza una red neuronal entrenada con calificaciones reales de usuarios de Amazon sobre CDs y vinilos.

## Qué hace

- Permite buscar álbumes por nombre.
- Recibe de uno a cuatro álbumes favoritos.
- Genera diez recomendaciones ordenadas por similitud.
- Evita recomendar álbumes repetidos o ediciones duplicadas.
- Ofrece una interfaz web local y comandos desde la terminal.

## Cómo funciona

El modelo convierte usuarios y álbumes en vectores de 32 dimensiones llamados *embeddings*. Durante el entrenamiento aprende a asignar puntajes altos a pares usuario-álbum con calificaciones positivas y puntajes bajos a pares negativos.

Para una persona nueva, el sistema:

1. Obtiene los embeddings de los álbumes seleccionados.
2. Calcula el promedio de esos vectores.
3. Compara el promedio con el catálogo disponible.
4. Elimina álbumes elegidos y ediciones duplicadas.
5. Devuelve los diez resultados más similares.

No necesita crear una cuenta ni usar un historial previo del usuario.

## Tecnologías

- Python
- TensorFlow y Keras
- NumPy
- SQLite
- HTTP con la biblioteca estándar de Python
- HTML, CSS y JavaScript
- Matplotlib

## Resultados

El modelo fue evaluado con usuarios que no participaron en el entrenamiento y fue comparado con una recomendación sencilla basada en los álbumes más populares.

| Métrica | Red neuronal | Álbumes populares |
| --- | ---: | --- |
| Hit rate@10 | 3.62% | 0.69% |
| Precision@10 | 1.38% | 0.46% |
| Recall@10 | 3.65% | 0.66% |
| F1@10 | 1.59% | 0.38% |

El modelo supera al baseline en todas las métricas registradas. Al ser una prueba con decenas de miles de álbumes, los porcentajes absolutos son reducidos.

Los resultados están disponibles en `artifacts/metrics.json` y en la base `data/album_recommender.db`.

## Ejecución local

### 1. Crear el entorno

Desde la raíz del proyecto:

```bash
python -m venv venv
```

Activar el entorno en Windows:

```bash
venv\Scripts\activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

### 2. Preparar el catálogo

El repositorio incluye el caché de calificaciones, el modelo y la base SQLite. Si se necesitan regenerar los datos:

```bash
cd src
python main.py preprocess
python main.py names
```

Los archivos fuente deben encontrarse en:

- `data/reviews/CDs_and_Vinyl_5.json.gz`
- `data/metadata/meta_CDs_and_Vinyl.json.gz`

### 3. Iniciar la interfaz web

```bash
python main.py server
```

Abrir `http://127.0.0.1:8000` en el navegador.

## Uso desde la terminal

Desde `src/` también se pueden obtener recomendaciones directamente:

```bash
python main.py recommend B000002IQ1 B000002UB4 -k 10
```

Comandos disponibles:

| Comando | Uso |
| --- | --- |
| `preprocess` | Construir el caché de calificaciones |
| `baseline` | Evaluar el ranking de álbumes populares |
| `train` | Entrenar y evaluar el modelo |
| `evaluate` | Recalcular las métricas de ranking |
| `experiments` | Comparar configuraciones de entrenamiento |
| `names` | Importar títulos y popularidad a SQLite |
| `recommend` | Recomendar desde la terminal |
| `server` | Iniciar la interfaz web |
| `import-legacy` | Importar artefactos de una versión anterior |

Ejemplo de entrenamiento:

```bash
python main.py train
```

## Pruebas

Ejecutar las pruebas unitarias desde la raíz del proyecto:

```bash
python -m unittest tests/test_core.py
```

## Estructura del proyecto

```text
src/
  controllers/  manejo de terminal y servidor web
  services/     lógica de datos, entrenamiento y recomendaciones
  ml/           arquitectura de la red neuronal
  daos/         consultas y persistencia en SQLite
  domain/       entidades del proyecto
  database/     conexión y tablas de SQLite
  views/        interfaz web
data/           datos y caché
artifacts/      modelo entrenado y gráficos
tests/          pruebas básicas
```

## Datos y artefactos

Los datos de calificaciones provienen del conjunto público *Amazon Review Data* para la categoría CDs and Vinyl. No se incluye el contenido de las reseñas en el proceso de recomendación: solo se usan identificadores de usuario, álbum y calificación.

Los artefactos incluidos permiten ejecutar el recomendador sin volver a entrenar el modelo.
