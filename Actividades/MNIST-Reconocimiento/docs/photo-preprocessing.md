# Extracción fotográfica inspeccionable

`photo.preprocess_photo` es una alternativa **optativa** al preprocesamiento
original. Devuelve un candidato y evidencia visual; no certifica que el dígito
esté aislado ni mejora por sí sola la precisión del modelo. `predict.py`, los
pesos y las fotografías originales permanecen intactos.

## Uso y revisión

```python
from PIL import Image
from photo import preprocess_photo

with Image.open('foto.jpeg') as image:
    result = preprocess_photo(image)
print(result.quality)
# Inspeccionar result.stages antes de aceptar el tensor para inferencia.
# Si hay ambigüedad, volver a llamar con crop=(izquierda, arriba, derecha, abajo).
```

- `tensor`: `float32`, forma `(1, 784)`, rango `[0,255]`, sin normalización.
- `preview`: `uint8`, forma `(28,28)`, idéntica al tensor reordenado.
- `quality.status`: `candidate` o `uncertain`. **Candidate no significa certeza**.
  `warning` explica riesgos; `manual_crop` identifica la intervención humana.
- Una imagen vacía o sin componente aislable produce `ValueError` en español.
  Varios componentes plausibles producen una vista candidata con advertencia,
  nunca una selección basada en la clase predicha.

`crop` contiene cuatro enteros en píxeles, con derecha/abajo exclusivos, dentro
del original **después de corregir EXIF**. Se rechazan coordenadas invertidas,
externas, booleanas o no enteras. El recorte explícito conserva todos sus
componentes sustanciales, incluidos los que tocan bordes; estos últimos generan
una advertencia porque podrían estar incompletos. No se promete recuperar
trazos que ya quedaron fuera del recorte o que no se distinguen del fondo.

## Etapas y coordenadas

`stages` conserva `original` orientado y compuesto sobre blanco, `gray`,
`illumination`, `contrast`, `lines`, `grid_removed`, `adaptive_mask`, `seeds`,
`hysteresis_mask`, `mask`, `crop` y `final`. `seeds` muestra núcleos fuertes;
`hysteresis_mask` muestra todas las regiones reconstruidas admisibles y `mask`
solamente la selección final (o su unión en un recorte manual).
Los valores son arrays `uint8`; el tensor mantiene intensidades,
no convierte el trazo final en una silueta binaria.

`bbox=[x0,y0,x1,y1]` usa coordenadas de trabajo y límites finales exclusivos.
`coordinates` contiene `source_size`, `source_crop`, `working_size`,
`source_per_working_pixel` y `exif_corrected`. Para regresar al original:
`x_original = source_crop[0] + x_trabajo * source_per_working_pixel[0]`,
análogamente para `y`. El trabajo se limita a 1200 píxeles por lado mayor;
el original orientado sigue disponible a resolución completa. No se escriben
archivos ni se modifica la imagen recibida.

## Técnicas, comparación y límites

| Etapa | Decisión y contrapartida |
|-------|--------------------------|
| Iluminación | Cierre en gris estima el papel; división reduce sombras. Puede atenuar trazos muy anchos. Kernel elíptico de 6% del lado menor, mínimo 31. |
| CLAHE | Contraste local, límite 2 y mosaico 8×8; también amplifica ruido. Afecta las intensidades finales, no decide la selección. `use_clahe=False` permite comparar sin él. |
| Líneas | Apertura en gris con longitudes de 45% de la dimensión fuente escalada, mayores que un dígito pequeño. Se resta intensidad de evidencia larga, no toda línea horizontal/vertical binaria. `remove_grid=False` permite inspeccionar la alternativa no destructiva. |
| Umbral adaptativo | Gaussiano, ventana impar hasta 51 y constante 4. Se conserva como comparación visual: puede unir cuadrícula inclinada y ruido, por eso no decide el aislamiento. |
| Semillas e histéresis | Usa tinta del gris corregido por iluminación antes de CLAHE, con resta larga en gris si está habilitada. Umbral fuerte: 70% del percentil 99.8, limitado a [6,80]. Cada semilla crece desde su píxel más fuerte hacia tinta superior al máximo entre 3, 80% del umbral fuerte y 55% de su mediana. Esto impide que ruido amplificado por CLAHE conecte toda la página. |
| Componentes | Área mínima: máximo entre 10 y 0.004% del área. En automático rechaza regiones al borde (margen de 0.5% del lado menor, mínimo 1), mayores al 65% de un lado, relación de lados ≥20 u ocupación menor al 7%. Ordena por energía de tinta cruda al cuadrado × raíz de ocupación. Marcas oscuras compactas ajenas todavía pueden ganar. |
| Centrado | Ajusta proporciones dentro de 20×20 y centra por masa en 28×28 sin cortar el resultado. No usa la salida del modelo. |

La resta en gris permite conservar exceso de intensidad del lápiz en cruces;
las pruebas sintéticas comprueban el pie horizontal del 1 y las intersecciones
del 4. No garantiza conservar un trazo del mismo tono que la cuadrícula,
un dígito del tamaño de la página o líneas oblicuas. Los recortes manuales
mantienen la escala fuente de los kernels para evitar tratar su pie como una
línea larga solamente por haber reducido el campo visual. Los núcleos que
reconstruyen regiones solapadas en más del 50% de la menor se fusionan: no se
cuentan como varios dígitos. Un recorte manual permite bordes y proporciones
que el automático rechaza; sigue necesitando una semilla distinguible del papel.
Un dígito legítimo al borde puede ser rechazado automáticamente: ampliar la foto
o usar un recorte manual es preferible a seleccionar silenciosamente el fondo.

Los scores de calidad son **heurísticos, no probabilidades calibradas**:
`ambiguity_ratio` compara los dos primeros componentes (advertencia desde 0.35),
`removed_ink_fraction` mide señal restada (advertencia sobre 0.7),
`active_fraction` mide ocupación y `component_count` cuenta candidatos.
También se advierte sobre ocupación menor al 20% de la caja, contacto con el
borde y líneas largas remanentes sin eliminación. Una cuadrícula sola perfecta
se rechaza en la prueba; una cuadrícula irregular o texturada puede generar
falsos candidatos. Ningún umbral demuestra aislamiento fiable.

**Antes de aceptar una transformación destructiva**, comparar original,
contraste, líneas, máscara y recorte. Si falta un trazo, desactivar la
eliminación de líneas o ampliar/ajustar el recorte. No aceptar silenciosamente
un resultado `uncertain`. `diagnose.contact_sheet` puede presentar las etapas;
`diagnose.py --mode compare` compara legado, adaptativo, sin cuadrícula y sin
CLAHE en directorios nuevos, sin sobrescribir evidencia previa.

## Evidencia y próximos pasos

La primera selección por área e intensidad después de CLAHE favorecía el
fondo oscuro fuera del papel y unía tinta con cuadrícula. Una regresión pública
genera papel con sombra, cuadrícula tenue, tinta compacta y una región oscura
exterior; comprueba que gana la tinta y se conserva el pie horizontal.
Las pruebas previas de cruces del 4, ambigüedad, bajo contraste y recorte manual
se mantienen sin debilitar sus comprobaciones.

La corrección por semillas e histéresis selecciona regiones compactas en las
tres fotos originales, sin usar etiquetas, predicciones ni coordenadas por
archivo. Los paneles PNG completos, inspeccionados gráficamente, están en:

- `resultados/adaptive-inspection-v2/4-2.png`: bbox de trabajo `[289,530,350,625]`.
- `resultados/adaptive-inspection-v2/1-2.png`: bbox de trabajo `[320,460,394,549]`.
- `resultados/adaptive-inspection-v2/2-2.png`: bbox de trabajo `[286,505,370,592]`.

Cada figura contiene original, todas las etapas y superposición de bbox.
Se observan el pie horizontal y los cruces conservados, sin el fondo exterior
ni la cuadrícula en las máscaras seleccionadas. Los tres estados son
`candidate`, sin advertencia; esto no equivale a una garantía de segmentación
ni de reconocimiento. En la primera foto quedan otros dos candidatos débiles
que no ganan. Estas fotos son ejemplos de desarrollo, no un conjunto
independiente ni una prueba de mejora general.

Las estadísticas y similitudes MNIST (media, varianza, desviación, fracción
activa, Pearson, covarianza y coseno) ya se calculan en `diagnose.py`; sirven
para diagnóstico, nunca para escoger transformaciones o forzar predicciones.
La comparación cuantitativa y las ablaciones quedan a cargo del flujo de
diagnóstico, y la interfaz debe mostrar advertencias y permitir recortar.

Si la extracción no basta, proponer una nueva versión del modelo entrenada
con fotos etiquetadas reales, **solo con autorización explícita**. Conservar
originales y modelo actual. Separar entrenamiento, validación y prueba por
escritor, página y grupo de captura para evitar fuga; reservar prueba intacta
hasta la evaluación final. Falta un corpus fotográfico independiente para
medir generalización. No se entrena ni se cambian pesos en esta unidad.

## Dependencia y fuente

Dependencia directa verificada: `opencv-python-headless==4.13.0.92` (sin GUI).
Referencia oficial consultada:
[operaciones morfológicas de OpenCV](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html).
La apertura combina erosión y dilatación y puede borrar trazos finos; por eso
no se aplica una apertura binaria indiscriminada al dígito. Las páginas de
CLAHE y umbralización no estuvieron disponibles durante la investigación;
no se atribuyen a ellas garantías de calidad ni valores óptimos de parámetros.
