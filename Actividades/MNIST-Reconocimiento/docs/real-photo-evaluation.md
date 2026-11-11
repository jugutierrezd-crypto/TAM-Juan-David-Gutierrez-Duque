# Comparación fotográfica sin cambiar pesos

`evaluate_photos.py` compara el preprocesamiento original y el fotográfico con **el mismo `models/mnist.keras`**. No hay una segunda red entrenada; no se entrenan, descargan ni modifican pesos. [Cómo cargar fotos](../datos/fotos_reales/README.md).

## Ejecutar

```bash
.venv/bin/python evaluate_photos.py --help
.venv/bin/python evaluate_photos.py datos/fotos_reales/evaluacion --output resultados/fotos-independientes-001
```

`root` contiene fotos directamente dentro de las carpetas `0`–`9`. Se aceptan JPEG, PNG, BMP, WebP y TIFF (también extensiones mayúsculas). `--partition` admite `evaluation` (predeterminado) o `development`. `--development-root` permite señalar fotos adicionales usadas durante desarrollo; por defecto usa `datos/fotos_reales/desarrollo`. Las huellas históricas de `development.json` se comprueban siempre. Un directorio de salida existente produce código 2 y no se sobrescribe. Los errores de validación no crean salida ni cargan el modelo. El lote completado, incluso con rechazos individuales documentados, termina con código 0.

## Resultados

| Archivo | Contenido |
|---|---|
| `comparison.csv` | Archivo y etiqueta real; predicción, confianza, acierto/error (`correct`), estado, error y tiempos separados de cada sistema. |
| `summary.json` | Accuracy, precision, recall, F1 por clase, macro y ponderados; matriz 10×10, soporte, rechazos, cobertura; hashes de fotos/modelo/código, HEAD y versiones. |

Filas de la matriz: etiqueta real; columnas: predicción. Rechazos e incertidumbre no llaman a la red, no reciben clase y **cuentan como errores** en accuracy y recall. No entran en una columna inventada de la matriz: `rejected_by_label` conserva esos casos y permite reconciliar el total. `coverage` es la fracción con predicción. No se excluyen errores difíciles para inflar métricas.

Sin soporte real, recall es `null`. Sin predicciones positivas pero con soporte, precision y F1 son cero. Macro promedia clases con soporte real; ponderado usa ese soporte. Las clases ausentes y el soporte quedan explícitos: una muestra de solo tres clases no da una evaluación de diez clases. Las métricas se calculan descriptivamente para cualquier corpus válido no vacío. Menos de veinte fotos en alguna clase o cualquier corpus de desarrollo se marca `descriptive_only`; superar ese umbral orientativo **no demuestra superioridad estadística**.

Softmax no es confianza calibrada. Los tiempos son una medición por imagen/sistema después del calentamiento, alternando el orden de los sistemas. Separan preprocesamiento e inferencia y excluyen decodificación, carga del modelo y exportación. No son un benchmark repetido ni incluyen el tiempo humano de recorte. El evaluador automático no aplica recortes manuales; cualquier evaluación asistida debe registrarse por separado, sin escoger recortes hasta acertar la etiqueta.

## Independencia y reproducibilidad

Antes de ejecutar, fijá qué fotos pertenecen a desarrollo y cuáles a prueba; separá por autor/sesión/original, no solo por archivo. El evaluador detecta duplicados por píxeles orientados, y reutilización conocida por bytes o píxeles, pero **no puede demostrar que una foto sea nueva** ni detectar todas las variantes recortadas, recomprimidas o tomas similares. `independent_claim=false` impide atribuir esa garantía automáticamente. Confirmar procedencia sigue siendo responsabilidad humana.

Conservá corpus, CSV/JSON, código y entorno. El registro de código incluye hashes de los cuatro módulos usados, además de HEAD (que por sí solo no identifica cambios sin commit). Si modificás el preprocesamiento después de mirar estos resultados, esas fotos pasan a desarrollo y necesitás un nuevo conjunto de prueba. No uses las tres fotos históricas ni los controles sintéticos como evaluación independiente.

No se hace una prueba estadística de superioridad ni se declara terminado el proyecto por tener tres aciertos de desarrollo. La evaluación independiente permanece pendiente hasta recibir fotos nuevas. [Resultados de desarrollo, advertencias Keras y verificación Gradio](real-photo-verification.md).
