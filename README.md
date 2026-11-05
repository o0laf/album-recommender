
# Recomendador de álbumes (Vynl)

Red neuronal (Keras) que, dados de 1 a 4 álbumes, recomienda 10 parecidos.
Hecho para **Vynl**, una app social tipo Letterboxd pero de música. Proyecto de la materia Redes Neuronales.

> Otros archivos del repo: [`AGENT.md`](AGENT.md) (qué IA se usó y cómo) y [`memoria.md`](memoria.md) (estado y decisiones, para retomar el trabajo).

## 1. Objetivo

Aprender, a partir de lo que le gustó a mucha gente, qué álbumes "van juntos".
Si te gustaron los álbumes A, B y C, la red busca álbumes que le suelen gustar a la gente que también ama A, B y C.

## 2. Dataset (la base de datos)

**Amazon Review Data (2018)**, Jianmo Ni, Jiacheng Li, Julian McAuley (UCSD):
https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2

- Categoría **CDs and Vinyl**, versión **5-core** (cada usuario y cada álbum tienen al menos 5 reseñas).
- 1.402.148 calificaciones únicas, 112.395 usuarios y 73.713 álbumes.
- Se usan solo 3 columnas: `reviewerID` (usuario), `asin` (código del álbum) y `overall` (estrellas de 1 a 5). **El texto de las reseñas no se usa.**
- 4 o 5 estrellas = "le gustó". 1 a 3 estrellas = "no le gustó".
- Los archivos no se suben a GitHub (son pesados). Se ponen en:
  - `data/reviews/CDs_and_Vinyl_5.json.gz`
  - `data/metadata/meta_CDs_and_Vinyl.json.gz` (opcional: sirve para los nombres de los álbumes)
- Además, `coverage.py` lee la base `database.sqlite` de Vynl (solo lectura) para medir cuántos álbumes de la app existen en el modelo.

## 3. Arquitectura de la red

    usuario > Embedding (32 números) ─┐
                                      ├> producto punto > sigmoide > probabilidad de "le gusta"
    álbum   > Embedding (32 números) ─┘

| Capa                 | Qué hace                                                | Tamaño              |
| -------------------- | -------------------------------------------------------- | -------------------- |
| Embedding de usuario | Convierte cada usuario en un vector de 32 números       | 89.917 usuarios x 32 |
| Embedding de álbum  | Convierte cada álbum en un vector de 32 números        | 73.700 álbumes x 32 |
| Producto punto       | Si los dos vectores apuntan parecido, el número es alto | 1 número            |
| Sigmoide             | Pasa ese número a un valor entre 0 y 1                  | 1 número            |

Código: `training/model.py`. Pérdida: `binary_crossentropy`. Optimizador: Adam (learning rate 0,003).

**Para recomendar a alguien nuevo** no se usa el embedding de usuario (esa persona no existe en el entrenamiento).
Se promedian los vectores de sus álbumes, se buscan los álbumes más cercanos (similitud coseno) y se sacan los que ya puso.

## 4. Entrenamiento

- **Positivos:** las calificaciones de 4 o 5 estrellas.
- **Negativos:** las de 1 a 3 estrellas, más 4 álbumes sorteados al azar por cada positivo. Se sortean de nuevo en cada época. Los populares salen más seguido (`NEG_POWER = 0.75`).
- **Ajustes:** `DIM = 32`, `BATCH = 4096`, `NEG_RATIO = 4`, `NEG_POWER = 0.75`, `LR = 0.003`, hasta 20 épocas.
- **Early stopping:** si la validación no mejora en 3 épocas, se corta y se queda la mejor época.
- **División:** los **usuarios** se dividen en 80 % train, 10 % validación y 10 % test. Así, el test son personas que la red nunca vio.

## 5. Evaluación

A cada usuario de test se le dan 4 álbumes que le gustaron y se piden 10 recomendaciones (sin repetir esos 4).
**Acierto** = una recomendación que también le gustó a esa persona.

| Métrica               | Qué responde                                                                      |
| ---------------------- | ---------------------------------------------------------------------------------- |
| **hit rate@10**  | ¿Apareció entre los 10 un álbum escondido que le gustó? (sí / no)             |
| **precision@10** | De los 10 recomendados, ¿qué parte le gustó? (aciertos / 10)                    |
| **recall@10**    | De todo lo que le gustó, ¿qué parte encontré? (aciertos / total que le gustó) |
| **F1@10**        | Promedio parejo de precision y recall: 2·P·R / (P+R)                             |

Se promedia entre todos los usuarios de test y se compara con un **baseline**: recomendar siempre los 10 álbumes más populares.

> Por qué precision, recall y F1 se miden sobre el ranking y no con "umbral 0,5": la red tiene un embedding por usuario,
> así que no puede opinar sobre usuarios que nunca vio. Medir el ranking es medir lo que hace la app de verdad.

## 6. Feedback loop (probar, medir, ajustar)

Si el resultado no mejora, se cambia un ajuste y se reentrena. `training/experiments.py` automatiza eso:
prueba varias configuraciones (cada una cambia **un** ajuste respecto de la base) y guarda una fila por prueba en
`models/experiments.csv`. La mejor se elige con **validación**, nunca con test.

| Ajuste        | Qué cambia                                   | Valores probados |
| ------------- | --------------------------------------------- | ---------------- |
| `DIM`       | Largo de cada embedding                       | 16, 32, 64       |
| `NEG_RATIO` | Negativos por cada "me gusta"                 | 2, 4, 8          |
| `NEG_POWER` | Qué tanto salen los populares como negativos | 0, 0,75, 1       |
| `BATCH`     | Ejemplos por paso de entrenamiento            | 1024, 4096       |

## 7. Cómo correrlo (PowerShell, desde la carpeta `training/`)

    pip install -r ../requirements.txt
    python preprocessing.py       # lee los datos (la 1ª vez tarda unos minutos y arma un caché)
    python baseline.py            # resultado de los populares
    python train.py               # entrena, evalúa y guarda todo en models/
    python evaluate.py            # (opcional) re-mide precision/recall/F1 sin reentrenar
    python experiments.py         # (opcional) prueba configuraciones > models/experiments.csv
    python names.py               # (opcional) títulos para buscar álbumes por nombre
    python recommend.py CODIGO1 CODIGO2
    python server.py              # abrir http://127.0.0.1:8000
    python coverage.py ../data/vynl/database.sqlite   # (opcional) cobertura con el catálogo de Vynl

**Entorno de ejecución en Codelab:** COMPLETAR (link o pasos; son los mismos comandos de arriba, desde `training/`).

## 8. Resultados

Medido en test (usuarios que la red nunca vio). Fuente: `models/metrics.json`.

|                           | Red              | Populares |
| ------------------------- | ---------------- | --------- |
| hit rate@10 (test)        | **0,0362** | 0,0069    |
| hit rate@10 (validación) | 0,0347           | 0,0050    |
| precision@10 (test)       | COMPLETAR        | COMPLETAR |
| recall@10 (test)          | COMPLETAR        | COMPLETAR |
| F1@10 (test)              | COMPLETAR        | COMPLETAR |

- La red acierta unas **5 veces más** que recomendar siempre los populares.
- Mejor época: la 8. La pérdida de entrenamiento sigue bajando después, pero la validación deja de mejorar: es sobreajuste, y por eso se usa early stopping.
- Gráficos en `models/`: `training_curves.png`, `hit_rate_comparison.png` y `ranking_metrics.png`.
- Los números de precision / recall / F1 salen al correr `train.py` o `evaluate.py` (completar la tabla con `models/metrics.json`).

## 9. Filtro de recomendaciones

En Amazon cada edición es un producto distinto, y hay singles y títulos con basura (HTML pegado, títulos vacíos).
Por eso `recommend.py` pide 30 candidatos y se queda con los primeros 10 que **no** sean:

- singles (título con "single" o con "/" doble tema),
- álbumes sin título,
- títulos con texto raro (HTML, links, casi todo símbolos),
- otra edición de un álbum que ya puso el usuario o que ya está en la lista.

Esto necesita los títulos (`python names.py`). Sin títulos, no se filtra.

## 10. Limitaciones

- Un hit rate de 3,6 % parece bajo: hay 73.700 álbumes y se mira un solo álbum escondido entre 10. Lo importante es la comparación con populares.
- Los álbumes nuevos o con pocas reseñas no están en el modelo.
- Solo se evalúan usuarios con al menos 5 álbumes que les gustaron.
- Recall@10 tiene un tope: con 10 recomendaciones no se pueden encontrar más de 10 álbumes.
- El filtro es por reglas simples y puede dejar pasar algún título raro o tirar uno bueno.
- No se usa el texto de las reseñas ni datos como género o artista.

## 11. Estructura

    training/   preprocessing.py  model.py  baseline.py  train.py  evaluate.py  experiments.py
                recommend.py  server.py  index.html  names.py  coverage.py
    models/     modelo, métricas, gráficos y experiments.csv (los .keras no se suben a GitHub)
    data/       datos descargados (no se suben a GitHub)

## Cita

Ni, Li, McAuley. "Justifying recommendations using distantly-labeled reviews and
fined-grained aspects". EMNLP 2019.
