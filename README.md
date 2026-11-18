# Recomendador de álbumes Vynl

Proyecto educativo de sistemas de recomendación que usa una red neuronal con embeddings para sugerir álbumes a partir de 1 a 20 códigos ASIN.

Incluye:

- Entrenamiento de embeddings de usuarios y álbumes con Keras.
- Recomendaciones por consola y servidor web local.
- Búsqueda de álbumes por nombre.
- Evaluación con `hit rate`, precision, recall y F1.
- Experimentos automatizados de hiperparámetros.
- Cobertura entre el catálogo de Vynl y el conjunto de datos.

## Vista rápida

```text
Usuario nuevo
    |
    +-- Embeddings de sus álbumes conocidos
    |       |
    |       +-- promedio del vector
    |
    +-- similitud coseno con todos los álbumes
            |
            +-- excluir álbumes conocidos
            +-- filtrar ediciones y productos no válidos
            +-- devolver los 10 resultados más cercanos
```

La aplicación usa **dos interfaces**:

- CLI: `python src/main.py recommend B000...`
- Web: `python src/main.py server` y luego `http://127.0.0.1:8000`

## Datos

Se utiliza **Amazon Review Data 2018**, categoría **CDs and Vinyl**, versión 5-core:

- [Dataset oficial](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2)
- 1.402.148 interacciones únicas.
- 112.395 usuarios.
- 73.713 álbumes.
- Solo se usan `reviewerID`, `asin` y `overall`.

Una reseña de 4 o 5 estrellas se considera positiva. Las reseñas de 1 a 3 estrellas se consideran negativas.

Los datos deben colocarse en:

```text
data/
  reviews/
    CDs_and_Vinyl_5.json.gz
  metadata/
    meta_CDs_and_Vinyl.json.gz
```

El archivo de metadata es opcional para entrenar y evaluar, pero es necesario para buscar álbumes por nombre.

## Instalación

Se recomienda Python 3.11 o 3.12. Desde PowerShell, en la raíz del proyecto:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación del entorno:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

El proyecto usa TensorFlow, NumPy y Matplotlib:

```text
tensorflow>=2.12
numpy
matplotlib
```

## Uso

Ejecuta los comandos desde la carpeta `src/`:

```powershell
cd src
```

### Preparar los datos

```powershell
python main.py preprocess
```

Lee el archivo comprimido de reseñas y crea `data/ratings_cache.npz`.

### Entrenar y evaluar

```powershell
python main.py baseline
python main.py train
```

`train` genera el modelo y los gráficos en `artifacts/` y guarda el historial, las métricas y los resultados en `data/album_recommender.db`.

Para volver a calcular la evaluación sin entrenar:

```powershell
python main.py evaluate
```

### Obtener recomendaciones

Pasa uno o más códigos ASIN separados por espacios:

```powershell
python main.py recommend B000002IQ1 B000002UB4
```

También puede cambiarse el número de resultados:

```powershell
python main.py recommend B000002IQ1 B000002UB4 -k 10
```

Antes de usar nombres, procesa el catálogo de metadata:

```powershell
python main.py names
```

### Abrir la interfaz web

```powershell
python main.py server
```

Abre `http://127.0.0.1:8000`. La interfaz permite buscar por nombre, seleccionar álbumes y solicitar recomendaciones.

### Probar configuraciones

```powershell
python main.py experiments
python main.py experiments --solo base dim64
python main.py experiments --max-epochs 3
```

Las pruebas guardan una fila por configuración en la tabla `experiment`. La selección se hace con validación, no con test.

### Medir cobertura de Vynl

Si tienes una copia local de `database.sqlite` del catálogo de Vynl:

```powershell
python main.py coverage ../data/vynl/database.sqlite
```

El resultado indica cuántos álbumes del catálogo están presentes en el modelo entrenado.

### Importar artefactos anteriores

```powershell
python main.py import-legacy
```

Importa una sola vez los archivos `.json` o `.csv` de una versión anterior del proyecto.

## Cómo funciona

### Arquitectura del modelo

El modelo aprende un vector de 32 números para cada usuario y otro para cada álbum:

```text
usuario -> embedding (32) --\
                             +-> producto punto -> sigmoide -> probabilidad
álbum   -> embedding (32) --/
```

| Componente | Descripción | Dimensiones |
| --- | --- | --- |
| Embedding de usuarios | Representación aprendida de cada usuario | 89.917 × 32 |
| Embedding de álbumes | Representación aprendida de cada álbum | 73.700 × 32 |
| Producto punto | Mide la similitud entre usuario y álbum | 1 |
| Sigmoide | Convierte el producto en una probabilidad | 1 |

El código se encuentra en `src/ml/album_embedding_network.py`. El modelo usa binary cross-entropy y Adam con learning rate `0.003`.

Como la aplicación recomienda a usuarios que no forman parte del entrenamiento, no usa su embedding. En su lugar:

1. Toma los embeddings de los álbumes seleccionados.
2. Calcula su promedio.
3. Busca los álbumes con mayor similitud coseno.
4. Elimina los álbumes ya conocidos y duplica ediciones.

### Entrenamiento

- Positivos: calificaciones de 4 o 5 estrellas.
- Negativos: calificaciones de 1 a 3 estrellas y álbumes muestreados aleatoriamente.
- Negativos por positivo: 4.
- Épocas máximas: 20.
- Early stopping: detiene el entrenamiento tras 3 épocas sin mejora en validación.
- Tamaño de lote: 4096.

Los negativos se muestrean nuevamente en cada época. `NEG_POWER = 0.75` hace que los álbumes populares tengan más probabilidades de aparecer como negativos.

### División de datos

La división se realiza por usuario:

- 80% entrenamiento.
- 10% validación.
- 10% test.

Los usuarios de validación y test no participan en el entrenamiento.

## Evaluación

Para cada usuario de test se esconden parte de sus álbumes positivos: se usan 4 como entrada y se pide un ranking de 10 álbumes.

| Métrica | Interpretación |
| --- | --- |
| Hit rate@10 | Porcentaje de usuarios para los que aparece al menos un álbum relevante |
| Precision@10 | Proporción de aciertos entre las 10 recomendaciones |
| Recall@10 | Proporción de álbumes relevantes recuperados |
| F1@10 | Media armónica de precision y recall |

También se calcula un baseline que siempre recomienda los 10 álbumes más populares.

## Resultados registrados

Los valores siguientes proceden de `artifacts/metrics.json` y `data/album_recommender.db`.

| Métrica | Modelo test | Populares test |
| --- | ---: | ---: |
| Hit rate@10 | 3.615% | 0.691% |
| Precision@10 | 1.380% | 0.461% |
| Recall@10 | 3.652% | 0.659% |
| F1@10 | 1.591% | 0.378% |

| Métrica | Modelo validación | Populares validación |
| --- | ---: | ---: |
| Hit rate@10 | 3.467% | 0.497% |
| Precision@10 | 1.333% | 0.457% |
| Recall@10 | 3.428% | 0.562% |
| F1@10 | 1.490% | 0.350% |

El modelo obtuvo aproximadamente **5.2 veces más hit rate** que el baseline en test.

- Usuarios evaluados en test: 9.266.
- Usuarios evaluados en validación: 9.260.
- Mejor época: 8.
- Épocas ejecutadas: 11.

Las gráficas disponibles son:

- `artifacts/training_curves.png`
- `artifacts/hit_rate_comparison.png`
- `artifacts/ranking_metrics.png`

## Experimentos disponibles

| Ajuste | Valor base | Valores probados |
| --- | ---: | --- |
| `DIM` | 32 | 16, 32, 64 |
| `NEG_RATIO` | 4 | 2, 4, 8 |
| `NEG_POWER` | 0.75 | 0, 0.75, 1 |
| `BATCH` | 4096 | 1024, 4096 |

Cada experimento cambia un ajuste respecto de la configuración base.

## Filtro de resultados

Amazon puede representar una misma edición como varios productos. Para reducir duplicados y resultados de baja calidad, el servicio solicita 30 candidatos y conserva los 10 mejores que cumplen estas reglas:

- No son sencillos cuyo título contiene `single` o usa `/` como separador de temas.
- Tienen título.
- No contienen fragmentos sospechosos como HTML, enlaces o entidades.
- No repiten otra edición de un álbum seleccionado.
- No repiten álbumes dentro del ranking.

Este filtrado depende de haber ejecutado `python main.py names`. Sin el catálogo de títulos solo se aplican las comprobaciones básicas disponibles.

## Estructura

```text
src/
  main.py                         Punto de entrada CLI
  config.py                       Rutas e hiperparámetros
  controllers/                    Entrada de consola y servidor web
  services/                       Reglas de negocio y evaluación
  daos/                           Acceso a SQLite
  domain/                         Entidades del proyecto
  database/                       Conexión y creación de tablas
  ml/                             Red neuronal con Keras
  views/index.html                Interfaz web

data/
  reviews/                        Dataset descargado
  metadata/                       Metadata descargada
  ratings_cache.npz               Caché generado
  album_recommender.db            Base SQLite generada

artifacts/
  model.keras                     Modelo generado
  album_to_idx.json               Índices ASIN
  album_titles.json               Catálogo de títulos
  metrics.json                    Últimos resultados
  *.png                           Gráficas
```

La separación sigue una arquitectura por capas:

```text
CLI / HTTP
  -> controllers
  -> services
  -> daos
  -> SQLite

domain/ contiene las entidades compartidas.
```

## Tablas SQLite

`data/album_recommender.db` contiene:

- `album`
- `training_run`
- `epoch_record`
- `metric`
- `experiment`
- `vynl_coverage`

## Limitaciones

- Los álbumes nuevos o con pocas reseñas no están presentes en el modelo.
- Solo se evalúan usuarios con suficientes álbumes relevantes.
- Recall@10 tiene un límite práctico de 10 recomendaciones.
- El filtro de títulos usa reglas simples y puede descartar álbumes válidos.
- No se usa el texto de las reseñas, género, artista ni metadatos audiovisuales.
- Los resultados dependen del muestreo y de la partición definida por la semilla.

## Referencia

Ni, J., Li, J. y McAuley, J. *Justifying recommendations using distantly-labeled reviews and fined-grained aspects*. EMNLP, 2019.
