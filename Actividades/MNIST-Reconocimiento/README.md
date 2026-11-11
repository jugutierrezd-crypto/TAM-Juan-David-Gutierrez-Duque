# MNIST: entrenamiento y reconocimiento local de fotografías
# Reconocimiento de Dígitos Manuscritos mediante Machine Learning

**Asignatura:** Teoría de Aprendizaje de Máquinas

**Integrantes:**
- Juan David Gutiérrez Duque
- Geronimo Maya Diaz

**Proyecto:** Clasificación de dígitos manuscritos utilizando redes neuronales convolucionales (CNN), procesamiento digital de imágenes y análisis estadístico.

**Herramientas:** Python, TensorFlow, Keras, OpenCV y Gradio.

---

## Descripción del proyecto

Este proyecto implementa un sistema de inteligencia artificial capaz de clasificar dígitos manuscritos del 0 al 9 utilizando el conjunto de datos MNIST.

También incorpora herramientas de preprocesamiento para analizar imágenes externas, visualización de predicciones y evaluación del desempeño del modelo.
El proyecto conserva `scratchpad.ipynb` y el modelo existente `models/mnist.keras` sin modificar. Ya hay historial en `resultados/training.json` (exactitud de test registrada: 96,51 % en MNIST, no en fotografías). La interfaz diagnostica fotos localmente; nunca inicia entrenamiento ni crea enlaces públicos.

## 1. Preparar WSL y VS Code

Desde una terminal **WSL**, no desde PowerShell:

```bash
cd /home/guti/proyectos-ia/tam/mnist-interactivo
# Ya existe .venv en esta entrega. Solo si falta, crealo:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q
```

Se verificó Python 3.12 en Linux/WSL, TensorFlow CPU 2.20.0 y Gradio 5.50.0. No necesitás CUDA. Las versiones directas están fijadas; no es un lock de dependencias transitivas ni una garantía para otros sistemas operativos.

En VS Code abrí esta carpeta mediante **WSL**, instalá la extensión Python en WSL y elegí **Python: Select Interpreter → .venv/bin/python**. Comprobá el intérprete en la terminal:

```bash
python -c "import sys; print(sys.executable)"
```

Debe terminar en `mnist-interactivo/.venv/bin/python`. No se utilizó ni modificó el entorno activo de otro proyecto.

## 2. Entrenar — requiere tu aprobación previa

**El modelo y el historial ya existen. No se autoriza reentrenar por iniciar la aplicación.** Cualquier entrenamiento adicional requiere aprobación explícita y una nueva versión que conserve los pesos actuales. Para consultar el comando existente:

```bash
cd /home/guti/proyectos-ia/tam/mnist-interactivo
.venv/bin/python train.py --help
.venv/bin/python train.py
```

Este comando descarga MNIST desde la fuente de Keras si no está en la caché local `resultados/.keras/`, entrena **30 épocas con batch_size=32**, evalúa las 10.000 muestras de test y guarda:

- `models/mnist.keras`: arquitectura y pesos en formato Keras.
- `resultados/training.json`: historial y métricas de test.

Si el modelo ya existe, se rechaza el entrenamiento sin sobrescribirlo. El informe existente también se conserva. Importar `train.py` o iniciar `app.py` no descarga MNIST ni entrena. La precisión histórica del cuaderno (aproximadamente 97,97 %) es de entrenamiento, no de test y no es una medición de esta implementación.

## 3. Reconocer una fotografía

```bash
cd /home/guti/proyectos-ia/tam/mnist-interactivo
.venv/bin/python app.py
```

Abrí la URL local que imprime Gradio, normalmente `http://127.0.0.1:7860`. Si ese puerto está ocupado, puede elegir otro. El servidor usa exclusivamente **127.0.0.1**, `share=False`, sin cámara y con analítica desactivada. Desde Windows/WSL el acceso a localhost depende de la configuración de red de WSL; no se abre el servidor a `0.0.0.0` como solución alternativa.

Arrastrá una fotografía de **un solo dígito**. La página analiza y recorta automáticamente en modo **Fotografía** con `photo.preprocess_photo`; no necesitás configurar el diagnóstico ni ingresar coordenadas. Si no detecta dudas de aislamiento, muestra la predicción sin preguntar. Esto **no garantiza acierto**: la calidad del recorte es heurística y Softmax no detecta todos los errores.

Si encuentra varios trazos plausibles, **compara candidatos automáticamente** antes de pedir ayuda: combina geometría y evidencia del modelo, no una supuesta «exactitud» de cada trazo. Solo elige si hay una ventaja clara; si empatan (incluso con la misma clase), hay trazos incompletos o falta evidencia, conserva la advertencia sin clase final y pide otra foto o un recorte. Los controles de ayuda aparecen solo entonces. Una foto sin trazo reconocible también pide otra imagen o recorte. Un modelo ausente muestra un error, no una pregunta sobre la foto. Al resolver la duda o vaciar la imagen se oculta la ayuda. Cada nueva foto reinicia las opciones y coordenadas anteriores.

Si el aislamiento automático queda sin resolver (también por dispersión con poca ambigüedad, demasiadas regiones o comparación sin ganador), **solo la ruta automática** prueba como máximo el procesamiento original y un respaldo con `robust=True`: suavizado gaussiano de textura a escala de la fuente, apertura gris 5×5 para suprimir líneas finas y semillas fuertes sin el techo histórico de intensidad. El crecimiento recupera bordes más suaves después de esa supresión; no se rebajan los filtros compartidos de geometría, ambigüedad ni estabilidad del modelo. Incluso un único candidato recuperado debe pasar la comparación y los cinco desplazamientos antes de mostrar una clase. Los errores del modelo se muestran, no se convierten en predicciones.

Cada comparación mantiene el límite de 8 regiones plausibles; no se amplía para resolver fotos ruidosas. Con `allow_tentative=True` o `allow_alternatives=True`, la geometría barata se comprueba antes de aplicar ese límite: marcas geométricamente inadmisibles no consumen cupos y siguen registradas como rechazadas. Nueve rivales plausibles siguen obligando a abstenerse. Si el respaldo también falla, se conserva la abstención. `candidate_comparison.previous_attempts` registra la comparación original cuando hubo dos intentos. Los errores del modelo y las diferencias en la revalidación final no disparan otro intento.

Si las semillas robustas globales fallan por marcas exteriores, la comparación automática puede reutilizar las regiones originales únicamente como propuestas para extracción **local robusta** (`proposal_context: original_regions`). Mantiene las cajas EXIF, el margen, los filtros, los cinco ensayos y la revalidación exacta del contexto local; no reutiliza el tensor original incierto. Es parte del único intento robusto, sin ampliar el presupuesto de regiones ni buscar recortes adicionales.

El respaldo es optativo en `preprocess_photo`, `compare_candidates` y la función Python `diagnose_image` (palabra clave `robust`, por defecto `False`). Las rutas compatibles, el modo Original y los recortes indicados por el usuario no lo activan automáticamente. La galería añade únicamente las etapas reales `smoothed` y `robust_ink`; el JSON identifica `fallback_used` y `method: smoothed_uncapped_hysteresis` en calidad o en la comparación si se rechaza. Las coordenadas siguen siendo del original orientado por EXIF.

Cuando la página robusta contiene **exactamente un componente** y calidad `candidate`, se evalúa su tensor de página sin volver a estimar iluminación dentro de un recorte. `candidate_comparison.evaluation_context: page` identifica esta vía: `crop`/`selected_crop` y `coordinates.source_crop` son el contexto completo de procesamiento, mientras `source_bbox` es la caja de tinta seleccionada. El diagnóstico final vuelve a usar la página y debe reproducir exactamente tensor, vista, clase, calidad y caja fuente; cualquier diferencia conserva la advertencia. No evita los filtros de geometría ni los cinco ensayos del modelo. Con varios componentes se mantiene la comparación local y la abstención ante empates.

**Resultado tentativo:** si el candidato tiene geometría válida, mantiene la misma clase en las cinco vistas y una ventaja de ranking ≥0,15 frente a **todos** los rivales evaluados, la interfaz puede mostrar **«Posible N»** con **«resultado incierto»**. La vía de evidencia moderada exige probabilidad mínima ≥0,5 y margen mínimo ≥0,2, sin cumplir el piso fuerte de 0,8. No convierte el candidato en aceptado: `eligible` sigue siendo falso. `compare_candidates` conserva el criterio conservador por defecto; solo permite esta vía con `allow_tentative=True`, que activa la interfaz automática. `diagnose_image` y `classify` no cambian su política.

**Alternativas inciertas (opción elegida por el usuario):** `compare_candidates(..., allow_alternatives=True)` permite una salida separada cuando hay **exactamente una región local válida**, con geometría admisible, trazo completo y contexto consistente, pero las cinco vistas alternan entre **2 o 3 clases distintas**. Exige probabilidad máxima ≥0,3 y margen entre las dos primeras ≥0,1 **en cada vista**. Son nuevos umbrales heurísticos, no calibrados ni entrenados; no modifican los mínimos fuertes/tentativos ni convierten ninguna clase en ganadora. Dos regiones localmente válidas impiden esta salida, aunque una tenga mejor ranking; los rivales espaciales no se resuelven enumerando clases. Más de ocho regiones plausibles, recortes inciertos/incompletos, evidencia uniforme/inválida o más de tres clases mantienen la abstención.

La interfaz activa ambas opciones; `allow_alternatives=False` conserva la política anterior del comparador. Las alternativas se ordenan por frecuencia entre vistas; en empate va primero la clase de la vista original y después el orden de aparición. Ejemplos: **«Resultado incierto: podría ser 4, 1 o 9»** o **«Resultado incierto: podría ser 2 o 7»**. Se conserva cada vista en JSON y se revalidan exactamente tensor, vista, calidad, coordenadas y caja; la clase original debe repetirse al diagnosticar el contexto final. **El gráfico de probabilidades se vacía** para no presentar una distribución como certificación de la lista.

**Toda recuperación de clase única de un aislamiento incierto, por comparación original o respaldo robusto, se muestra como «Posible N»**, incluso con Softmax fuerte: haber resuelto un problema de calidad no certifica el dígito. En ese caso el informe superior dice `prediction_status: tentative`, mientras la comparación conserva su evidencia original (`accepted` si superó el piso fuerte). Las fotos claras ya admitidas conservan su flujo sin aviso.

Al mostrar un resultado tentativo **o alternativas** aparece un **aviso emergente de Gradio durante 12 segundos**: «Resultado incierto. Si es posible, subí una imagen más nítida, tomada más cerca y con mejor iluminación; esto puede ayudar al reconocimiento, pero no garantiza una respuesta correcta.» El consejo y la incertidumbre permanecen en el estado después de cerrar el aviso; también queda disponible la ayuda para subir otra foto o recortar. No hay aviso para resultados claros, imágenes vacías, empates o candidatos rechazados. Geometría inválida, evidencia demasiado baja, errores del modelo o diferencias en la revalidación siguen impidiendo mostrar un resultado. La inestabilidad de clase nunca produce una clase ganadora; solo permite alternativas bajo las condiciones anteriores.

**Observaciones de desarrollo:** la foto que motivó inicialmente el respaldo conserva **«Posible 2»** (probabilidad mínima entre vistas 0,741; margen mínimo 0,597). La nueva foto de trazo curvo muestra **«Posible 3»** (mínimo 0,734; margen 0,601). La foto con borde de cuaderno y marcas externas muestra **«Resultado incierto: podría ser 4, 1 o 9»**: un único candidato local completo produce clases [4, 1, 4, 1, 9] entre desplazamientos (probabilidad mínima 0,377; margen mínimo 0,117). No hay clase ganadora ni gráfico de probabilidades; se muestran el consejo persistente y el aviso para mejorar la imagen. Son evidencia de desarrollo, **no una medición independiente de mejora de exactitud**, ni probabilidades calibradas de acertar.

La comparación automática está acotada a **8 regiones geométricamente plausibles** recuperadas de la máscara de histéresis. El comparador conservador por defecto mantiene su límite histórico de 8 regiones totales. En la comparación local, cada caja se transforma al original orientado por EXIF, se amplía un 10 % (mínimo 3 píxeles por lado) y se reprocesa con el extractor existente. Se descartan recortes inciertos, fragmentados, de borde o con geometría inadecuada: densidad de tinta entre 0,2 y 0,85 y ancho/alto entre 0,08 y 2,5; una línea recta fina no basta. La pérdida excesiva de señal o regiones no recuperables también obliga a pedir ayuda.

Para cada región admisible se compara la predicción original con cuatro desplazamientos de un píxel. Para la aceptación fuerte se exige la misma clase en los cinco casos, probabilidad máxima mínima ≥0,8 y margen entre primera y segunda clase ≥0,2. El ranking combina 30 % geometría (densidad próxima a 0,35) y 70 % promedio de probabilidad mínima y margen mínimo; necesita una diferencia ≥0,15 respecto al siguiente candidato evaluado. **Son umbrales heurísticos, no confianza calibrada ni evidencia de mayor exactitud**. El contexto elegido se diagnostica de nuevo y solo se muestra si sigue siendo `candidate` y reproduce tensor, vista, calidad, coordenadas y clase. Esto también se exige a los resultados tentativos. Este trabajo adicional no se aplica a fotos claras, imágenes uniformes sin contenido, recortes manuales ni modo Original.

Las etapas, superposición y JSON están en **Detalles del diagnóstico (opcionales)**, cerrado inicialmente. En **Opciones avanzadas (opcionales)** podés comparar **Original**, desactivar la eliminación de cuadrícula o CLAHE; después pulsá **Reintentar diagnóstico**. Cambiar opciones o coordenadas por sí solo no recalcula los resultados.

| Salida | Qué comprobar |
|---|---|
| Galería | `original`, `gray`, `illumination`, `contrast`, `lines`, `grid_removed`, `adaptive_mask`, `seeds`, `hysteresis_mask`, `mask`, `crop`, `final`. Solo etapas reales; operaciones desactivadas dicen «no aplicado». |
| Superposición | Original orientado por EXIF; rojo: caja del candidato, azul: recorte usado, manual o automático. |
| JSON | Dimensiones EXIF corregidas, coordenadas fuente/trabajo, calidad heurística, media, varianza, desviación, fracción activa (>15), tensor exacto y tipo/forma. `candidate_comparison` conserva cajas, elegibilidad, motivos, clase, estabilidad y puntuaciones, incluso en empate/rechazo; `scores_calibrated` es falso. `crop_selection: automatic` identifica la selección automática (usa `manual_crop: true` para recortes locales y `false` para contexto de página). La comparación de la interfaz conserva el tensor, la calidad y las coordenadas evaluados para revalidar. `prediction_status: tentative` distingue el resultado tentativo de `accepted` en selecciones fuertes; `digit` contiene la clase numérica estimada en resultados de clase única, y `min_probability`/`min_margin` la evidencia mínima entre vistas, no exactitud. Con `prediction_status: alternatives` no hay `digit` ganador: `alternative_digits` contiene la lista, `base_digit` y `raw_probabilities` conservan la inferencia original, y `views` registra desplazamiento, clase, probabilidad máxima, margen y probabilidades de cada vista; `scores_calibrated` es falso. La calidad del aislamiento (`candidate`) no certifica la predicción. |
| Vista y probabilidades | Vista 28×28 idéntica al tensor `float32 (1,784)` sin dividir por 255; Softmax de diez clases, **no confianza calibrada**. |

Si el aislamiento es incierto aparece **ADVERTENCIA**, sin clase ni probabilidades. Un estado `candidate` tampoco garantiza aislamiento correcto. Activá **Usar recorte manual**, indicá `x0,y0,x1,y1` enteros sobre el original corregido por EXIF (origen superior izquierdo, extremos derecho/inferior exclusivos) y pulsá **Reintentar diagnóstico**. Las dimensiones están en `coordinates.source_size`; incluí margen, el pie del 1 y los cruces del 4. Son controles numéricos, **no una herramienta de dibujo**. El recorte se valida en `preprocess_photo`; solo funciona en modo Fotografía.

Sin modelo se conservan las etapas disponibles y se avisa sin inventar predicciones. Sin imagen, con rechazo o con fallo se limpian resultados anteriores (incluidos JSON y galería); el extractor no expone etapas parciales cuando rechaza la segmentación. Una advertencia de incertidumbre conserva únicamente la evidencia actual para inspección.

`app.classify(image)` y **`/classify`** conservan sus cuatro resultados `(clase, probabilidades, vista, estado)`. **`/diagnose_image`** y **`/retry_diagnosis`** conservan sus nueve entradas y siete resultados: clase, probabilidades, vista, estado, galería, superposición y JSON. Estas tres rutas se mantienen mediante disparadores de compatibilidad ocultos.

La interfaz usa **`/analyze_image`** al subir una foto: recibe solo la imagen. El evento actualiza los siete resultados, la visibilidad de la ayuda, la pregunta (vacía cuando no hace falta) y ocho valores que reinician opciones/recorte. **`/resolve_image`** recibe las nueve entradas del diagnóstico para reintentar y actualiza los siete resultados más visibilidad y pregunta. La función Python `analyze_image` devuelve esos nueve resultados; `analyze_upload` añade los ocho valores de reinicio. **`gradio_client` omite el grupo de visibilidad**: devuelve 16 valores en `/analyze_image` (pregunta en índice 7; reinicio desde índice 8) y 8 en `/resolve_image` (pregunta en índice 7). La API Python `predict_digit` / `preprocess_image` sigue usando el sistema original; no se alteran los pesos ni el comportamiento por defecto del extractor.

Ejemplo con `gradio_client` y un archivo aportado por vos, sin cuadro de rutas arbitrarias del servidor:

```python
from gradio_client import Client, handle_file
client = Client('http://127.0.0.1:7860')
result = client.predict(handle_file('mi_digito.png'), 'Fotografía', True, True,
                        False, 0, 0, 0, 0, api_name='/diagnose_image')
legacy = client.predict(handle_file('mi_digito.png'), api_name='/classify')
```

Detalles: [extracción y coordenadas](docs/photo-preprocessing.md), [tabla anterior/nuevo, ablaciones y límites](docs/photo-evaluation.md). **T4 sigue pendiente**: falta un conjunto independiente de fotografías etiquetadas; tres fotos de desarrollo y MNIST renderizado no demuestran mejora general.

Detené el servidor con `Ctrl+C`. No se guardan fotografías automáticamente. Gradio puede crear archivos temporales locales al recibirlas; no es un sistema de almacenamiento privado endurecido.

## Arquitectura y dimensiones

Para un lote de tamaño N:

| Etapa | Dimensión | Parámetros |
|---|---|---:|
| MNIST gris y aplanado | `(N,28,28)` → `(N,784)` | 0 |
| Dense + ReLU | `(N,64)` | 784×64 + 64 = 50.240 |
| Dense + ReLU | `(N,64)` | 64×64 + 64 = 4.160 |
| Dense + Softmax | `(N,10)` | 64×10 + 10 = 650 |
| Total | | **55.050** |

Cada capa calcula `h = ReLU(xW + b)`; la salida calcula `pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)`. Las etiquetas son enteros 0–9: se usa `sparse_categorical_crossentropy`, Adam y métrica `accuracy`, exactamente como en el cuaderno. No se agregó CNN, normalización, aumento de datos ni validación durante `fit`.

El cuaderno aplana los bytes originales y Keras los convierte implícitamente a float32. Los scripts hacen explícita esa conversión: **NO dividen por 255**. El entrenamiento usa `(60000,784)` y la evaluación `(10000,784)`.

## De fotografía a tensor

`predict.preprocess_image(image)` devuelve `(tensor, preview)`:

1. Corrige orientación EXIF, compone transparencia sobre blanco y convierte a gris.
2. Estima el fondo por la mediana del borde; invierte si el fondo es claro para obtener trazo claro sobre fondo oscuro.
3. Resta el nivel de fondo y detecta contraste. Rechaza imágenes uniformes o con contraste insuficiente.
4. Recorta la caja del trazo, mantiene proporción y ajusta dentro de **20×20**.
5. Coloca en un lienzo negro de **28×28**, centra por masa sin cortar el trazo.
6. Devuelve vista `uint8 (28,28)` y lote **`float32 (1,784)` en 0..255**.

```python
from PIL import Image
from predict import load_model, preprocess_image, predict_digit

image = Image.open('mi_digito.png')  # Archivo local aportado por vos.
tensor, preview = preprocess_image(image)
print(tensor.shape, tensor.dtype, tensor.min(), tensor.max())
Image.fromarray(preview).save('resultados/preview.png')  # Guardado solo si lo pedís.
model = load_model()
digit, probabilities, _ = predict_digit(image, model=model)
print(digit, probabilities)
```

Las rutas por defecto de modelo y resultados parten de la ubicación de los scripts, no de la carpeta actual. Por ejemplo, desde cualquier directorio:

```bash
/home/guti/proyectos-ia/tam/mnist-interactivo/.venv/bin/python /home/guti/proyectos-ia/tam/mnist-interactivo/app.py
```

## Pruebas y límites

```bash
.venv/bin/python -m pytest -q tests/
.venv/bin/python -m pip check
```

Indicá `tests/` para no recolectar también las copias de pruebas incluidas en `entregables/`.

Las pruebas usan imágenes sintéticas generadas localmente. Verifican forma, tipo, escala sin normalización, polaridad, centrado, transparencia, imágenes vacías, modelo ausente y distribución Softmax. Una prueba crea la arquitectura con **pesos aleatorios**, la guarda en el directorio temporal de pytest y la vuelve a cargar. No entrena ni descarga imágenes.

Un tensor correcto no demuestra buena exactitud fotográfica. MNIST y las fotos tienen distribuciones diferentes: sombras, papel texturado, fondos complejos, varios dígitos y trazos que toquen el borde pueden fallar. Los umbrales son heurísticos; una foto con ruido puede parecer contenido. Softmax suma uno pero **no es confianza calibrada** ni detecta imágenes fuera de distribución. La exactitud registrada sobre test MNIST es histórica y no valida el reconocimiento de fotografías reales. La evaluación fotográfica independiente sigue pendiente; no se reentrena sin autorización.

## Diagnóstico reproducible de fotografías

El modelo existente no se reentrena ni se sobrescribe. Para registrar el sistema original:

```bash
.venv/bin/python diagnose.py /ruta/foto.jpeg --output resultados/diagnostico-nuevo --mode legacy
.venv/bin/python diagnose.py --help
```

La salida debe ser nueva. Cada fotografía genera copia original, etapas PNG, figura
`comparison.png`, tensor exacto `float32 (1,784)` en `tensor.npy` y `report.json` con
las diez probabilidades, estadísticas y tiempos separados de preprocesamiento e inferencia.
El modelo se carga y precalienta fuera de esos tiempos. El informe raíz registra ruta,
fecha y SHA-256 del modelo. En modo original, contraste y eliminación de cuadrícula
se muestran como **no aplicados**, no como transformaciones existentes.

La proporción activa cuenta intensidades **>15** sobre 784 píxeles; la varianza es
poblacional y la covarianza usa `N-1`. Pearson y coseno indefinidos se guardan como
`null`. Las referencias son el primer ejemplo **de entrenamiento** por clase de la
caché MNIST local, sin descargas. No seleccionan ni cambian la predicción.

Para evaluación etiquetada, `--manifest /ruta/test.csv` acepta columnas `path,label`
(rutas relativas al CSV). Las etiquetas se añaden a la tabla **después** de inferir.
Las fotos de desarrollo no constituyen una evaluación independiente ni permiten
concluir mejora general. Softmax no es una medida calibrada de confianza.

## Evaluación automática de fotografías reales

Las carpetas etiquetadas están en `datos/fotos_reales/{desarrollo,evaluacion}/0` a `9`.
Copiá fotografías **nuevas** a la carpeta de su etiqueta real; las tres fotos históricas
permanecen en desarrollo y no constituyen prueba independiente.

```bash
.venv/bin/python evaluate_photos.py datos/fotos_reales/evaluacion --output resultados/fotos-independientes-001
```

El comando compara ambos preprocesamientos con los mismos pesos, conserva rechazos
y genera `comparison.csv` y `summary.json` con métricas, matriz de confusión y huellas.
No entrena ni sobrescribe resultados. [Carga](datos/fotos_reales/README.md) y
[procedimiento, métricas y límites](docs/real-photo-evaluation.md).

## Archivos

- `train.py`: construcción, entrenamiento explícito y evaluación.
- `predict.py`: preprocesamiento inspeccionable, carga y predicción.
- `app.py`: interfaz local, sin entrenamiento automático.
- `tests/test_predict.py`: pruebas deterministas y persistencia Keras.
- `photo.py`: extracción fotográfica optativa con etapas y advertencias.
- `candidates.py`: comparación conservadora acotada, sin entrenamiento ni cambio de pesos.
- `diagnose.py`: exportación explícita de evidencia y tablas comparativas.
- `tests/test_app.py`: rutas compatibles, diagnóstico, recorte y configuración local.
- `models/`, `resultados/`: modelo, historial y evidencia local existentes; nunca sobrescribirlos.
- `odd/tasks/mnist-grid-diagnostics.md`: requisitos y evidencia de diagnóstico.
