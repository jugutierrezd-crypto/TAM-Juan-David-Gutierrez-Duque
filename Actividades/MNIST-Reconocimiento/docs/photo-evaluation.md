# Diagnóstico reproducible de fotografías

El modo adaptativo reconoce las tres fotografías de desarrollo en esta ejecución, sin cambiar pesos ni usar sus etiquetas para preprocesar o inferir. Esto **no demuestra una mejora general en fotografías reales**. T4 sigue pendiente de un conjunto fotográfico independiente etiquetado.

## Uso

Desde la raíz, con el entorno y el modelo local existentes:

```bash
.venv/bin/python diagnose.py --help
.venv/bin/python diagnose.py --manifest resultados/evaluacion-desarrollo.csv --mode compare --output resultados/diagnostico-adaptativo
```

La segunda orden ya fue ejecutada: ahora rechaza ese directorio para proteger la evidencia. Para un diagnóstico posterior, elegí un directorio inexistente. También se aceptan rutas posicionales sin etiquetas; `--mode legacy` es el valor predeterminado y `--mode adaptive` ejecuta solamente la extracción nueva. El manifiesto contiene `path,label` (0–9), con rutas relativas al CSV o absolutas. Las etiquetas solo se añaden a las tablas **después** de inferir.

No se entrena ni descarga nada. `--mnist` señala una caché local NPZ; por defecto es `resultados/.keras/datasets/mnist.npz`. Si no existe, las similitudes quedan en `null` y los metadatos dicen `unavailable`.

## Evidencia y coordenadas

Cada combinación de archivo y modo tiene su propio directorio `NNN_nombre_modo`:

| Archivo | Significado |
|---|---|
| `original.jpeg` / extensión original | Copia exacta de los bytes de entrada, sin sobrescribirlos. |
| `original_stage.png` | Original de PhotoResult corregido por EXIF y con transparencia sobre blanco. |
| Etapas `*.png` | Todas las matrices de imagen devueltas por el preprocesador, incluyendo `seeds` y `hysteresis_mask`. |
| `comparison.png` | Figura de etapas; las leyendas indican operaciones no aplicadas y evidencia parcial. |
| `tensor.npy` | Tensor exacto enviado al modelo: float32, forma 1×784, rango 0–255, sin normalizar. |
| `report.json` | Diez probabilidades, predicción real del modelo, estadísticas, similitudes, calidad heurística, tiempos y coordenadas. |
| `mnist_references.png` | Primer ejemplo de **entrenamiento** por clase, no ejemplos del conjunto de prueba sintético. |

`bbox` usa coordenadas de trabajo `[izquierda, arriba, derecha, abajo]`, con extremo derecho/inferior exclusivo. `coordinates` conserva tamaño original corregido por EXIF, recorte fuente, tamaño de trabajo y escala fuente/píxel de trabajo. Para pasar al original: sumar el origen de `source_crop` después de multiplicar cada coordenada por su escala. Legacy trabaja a tamaño original; adaptativo reduce a un máximo de 1200 píxeles.

En legacy, contraste y eliminación de cuadrícula son copias de gris: JSON y figura indican **sin aplicar**, no mejoras reales. En las ablaciones se marca del mismo modo la operación desactivada. `adaptive_mask` es evidencia diagnóstica de umbralización; el candidato final se selecciona por semillas e histéresis de intensidad, no exclusivamente con esa máscara.

Si no se aísla un dígito, se conserva la copia original, el gris seguro disponible y una figura parcial. PhotoResult no expone sus etapas internas al lanzar una excepción: no se reconstruyen ni inventan etapas faltantes. Queda un JSON y una fila `rejected` con advertencia y error; predicción, confianza, estadísticas y tiempo de inferencia ausentes se representan como `null` en JSON y celdas vacías en CSV. No se envía una imagen rechazada a la red. El CLI termina con código 0 si completó el lote documentando rechazos; los rechazos deben revisarse en la tabla y stderr. Un directorio de salida existente causa código 2 sin sobrescribirlo.

## Tabla anterior/nuevo y ablaciones

`comparison.csv` contiene una fila por entrada: etiqueta real, predicciones anterior/nueva, confianza Softmax de cada una, calidad y advertencias, error y tiempos separados de preprocesamiento e inferencia. `quality_ground_truth=False` deja explícito que la calidad **no es una anotación humana ni una medida de exactitud**. `candidate` solo significa que pasó las comprobaciones heurísticas; `uncertain` requiere inspección; `unassessed` identifica el recorte global antiguo.

Resultados observados en `resultados/diagnostico-adaptativo/comparison.csv`:

| Foto / real | Anterior → nuevo | Softmax anterior / nuevo | Calidad anterior / nueva | Preprocesamiento ms anterior / nuevo | Inferencia ms anterior / nuevo |
|---|---|---|---|---|---|
| 4-2 / 4 | 2 → 4 | 1.000000 / 0.993175 | unassessed / candidate | 133.15 / 137.39 | 3.19 / 2.99 |
| 1-2 / 1 | 2 → 1 | 0.988300 / 0.985382 | unassessed / candidate | 38.08 / 109.68 | 2.79 / 3.13 |
| 2-2 / 2 | 2 → 2 | 0.998836 / 1.000000 | unassessed / candidate | 24.90 / 105.72 | 2.72 / 2.88 |

El 1 sí se reconoce en adaptive en esta ejecución; **sin eliminación de cuadrícula sigue prediciendo 2** (0.997772). La extracción visual preservada no garantiza reconocimiento correcto. No se sustituyeron predicciones manualmente.

`evaluation.csv` conserva la tabla larga de cuatro modos: `legacy`, `adaptive`, `no-grid` y `no-clahe`. Sin cuadrícula significa que **no se elimina** la cuadrícula, no que la foto carezca de ella. En las tres fotos esa ablación advierte líneas residuales. Sin CLAHE conserva la corrección de iluminación y el resto del proceso; el 1 baja a 0.712540 de Softmax. Estas comparaciones describen sensibilidad local, no atribuyen causalidad ni justifican ajustar parámetros con tres etiquetas.

Los tiempos usan reloj monotónico por imagen, tras cargar/calentar el modelo. Separan preprocesamiento de inferencia y excluyen exportación de PNG/JSON, carga inicial del modelo y preparación inicial del original. Son una medición local, no un benchmark repetido. Softmax no está calibrada: el sistema anterior se equivoca en el 4 con confianza 1.

## Estadísticas y referencias

Cada imagen final aceptada tiene media, varianza poblacional, desviación estándar y proporción de píxeles mayores de 15. Se calculan Pearson, covarianza muestral y coseno contra un primer ejemplo MNIST **de entrenamiento** de cada clase. Pearson/coseno degenerados son `null`. Los índices de referencia por clase 0–9 son `1,3,5,7,2,0,13,15,17,4`, registrados en `metadata.json`. Estas medidas describen semejanzas, nunca cambian la clase inferida; un ejemplo por clase no constituye un clasificador validado.

El modelo conservado es `models/mnist.keras`, SHA256 `ab1d64d10ac6b5d2b1f39948c643d53197ed89179a128a262ed9a63039477fa3`. Los metadatos registran ruta absoluta, mtime, dimensiones y escala. No se modificaron `photo.py`, `train.py`, los originales ni los pesos en este trabajo diagnóstico.

## Chequeo SINTÉTICO de dominio, no prueba de fotografía real

`resultados/mnist-independent-check/` contiene una única pasada sobre índices **0–99 de x_test**, nunca utilizados aquí para elegir umbrales. Cada ejemplo se renderizó como tinta oscura en papel claro con la misma cuadrícula y sombra, sin rotaciones, aleatoriedad ni reglas por clase. El código de una sola ejecución es `evaluate.py`; `config.json` fija índices, render, interpolación y hashes de caché/modelo. Rechaza reejecutarse sobre los resultados existentes.

- Anterior: **95/100**, nuevo: **96/100**; ningún rechazo. Todavía hay **4 errores** con el nuevo proceso.
- Tiempo medio anterior: preprocesamiento 1.30 ms, inferencia 3.15 ms; nuevo: 6.88 ms y 3.27 ms.
- `comparison.csv` conserva las 100 comparaciones; `summary.json` resume resultados; `contact-sheet.png` muestra los primeros diez ejemplos con ambas vistas finales.
- Cada subdirectorio conserva etapas, tensor, probabilidades, estadísticas y similitudes contra las referencias de **train**, separadas de estos ejemplos de **test**.

La diferencia de un acierto en una muestra sintética pequeña no prueba superioridad general. MNIST renderizado no reproduce todas las condiciones de cámara, papel, lápiz y escritura. **T4 no está cumplida**: falta un conjunto de fotografías reales independiente de estas tres imágenes de desarrollo y del ajuste del extractor.

## Próximo paso: solo con autorización

Si persisten errores en fotografías independientes, se propone recopilar fotos reales etiquetadas de dígitos sobre papel cuadriculado y separar entrenamiento, validación y prueba por autor/sesión/original para evitar fugas. Las tres fotos actuales seguirán siendo desarrollo, no prueba independiente. Antes de entrenar debe solicitarse autorización explícita; se conservará el modelo original y cualquier nuevo entrenamiento producirá otra versión y una evaluación ciega sobre la partición de prueba. No se ha autorizado ni ejecutado entrenamiento adicional.

## Fuente técnica y límite de investigación

Se consultó la documentación oficial de operaciones morfológicas: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html . Las otras dos páginas oficiales intentadas sobre CLAHE y umbralización no estuvieron disponibles (403/error de acceso); no se les atribuye evidencia recuperada. Las ablaciones y artefactos locales son la evidencia ejecutada, no un reemplazo de esa investigación pendiente.
