# Verificación local de la comparación fotográfica

Se comparó **el mismo modelo con dos preprocesamientos**, sin reentrenar ni modificar pesos. Las tres fotos son de **desarrollo**; no hay todavía fotografías nuevas de evaluación independiente.

## Resultado observado

Evidencia: `resultados/fotos-desarrollo-001/comparison.csv` y `summary.json`. Tiempos en milisegundos, una medición local tras calentamiento; no son un benchmark.

| Foto / real | Original: predicción / Softmax / acierto / inferencia ms | Fotográfico: predicción / Softmax / acierto / inferencia ms |
|---|---|---|
| 1-2 / 1 | 2 / 0,988300 / error / 6,06 | 1 / 0,985382 / acierto / 9,51 |
| 2-2 / 2 | 2 / 0,998836 / acierto / 9,16 | 2 / 1,000000 / acierto / 6,24 |
| 4-2 / 4 | 2 / 1,000000 / error / 11,11 | 4 / 0,993175 / acierto / 7,22 |

| Métrica descriptiva | Original | Fotográfico |
|---|---|---|
| Accuracy | 1/3 (33,33 %) | 3/3 (100 %) |
| Macro precision (clases presentes 1,2,4) | 0,111111 | 1 |
| Macro recall | 0,333333 | 1 |
| Macro F1 | 0,166667 | 1 |
| Cobertura | 3/3 | 3/3 |

Soporte: una foto de cada clase 1,2,4; cero de las demás. El JSON conserva la matriz 10×10 y las métricas por clase; no se interpreta este 100 % como exactitud independiente ni como mejora general. Las tres copias tienen las mismas huellas que los originales conservados en el diagnóstico previo.

## Las seis advertencias de Keras

La suite completa produjo **45 passed, 6 warnings**. Las seis son repeticiones del mismo `DeprecationWarning` en `test_real_keras_roundtrip`, no seis fallos distintos:

```text
keras/src/backend/tensorflow/core.py:172:
__array__ implementation doesn't accept a copy keyword, so passing copy=False failed.
__array__ must implement 'dtype' and 'copy' keyword arguments.
```

Keras convierte variables a NumPy con `return np.array(x)`. El protocolo de NumPy 2 espera el argumento `copy`; la firma local de `ResourceVariable.__array__` de TensorFlow es `(self, dtype=None)`. Hay seis tensores entrenables (kernel y bias de tres capas Dense), compatibles con seis avisos durante la persistencia de la red aleatoria de prueba. La prueba guarda/carga ese modelo temporal y comprueba igualdad de salidas: pasó. No guarda el modelo del proyecto ni entrena.

Entorno observado: NumPy 2.5.4, TensorFlow CPU 2.20.0, Keras 3.15.1. `pip check` pasó. **No requieren corregir el preprocesamiento ni los pesos para esta evaluación**, pero sí registrar una deuda de compatibilidad de dependencias antes de actualizarlas. No se ocultaron avisos, no se parchearon bibliotecas y no se cambiaron versiones: una actualización compatible debe probarse aparte con persistencia y comparación de salidas. Esto no garantiza compatibilidad con futuras versiones de NumPy. El mensaje AVX2/FMA de TensorFlow es informativo sobre optimización CPU, no otro error de inferencia.

Evidencia local: `resultados/verificacion-fotos-reales/pytest.log`, `keras-warning-details.json` y `evaluacion.log`.

## Gradio y recorte manual

`resultados/verificacion-fotos-reales/check_gradio.py` arrancó un servidor propio en `127.0.0.1`, sin enlaces públicos, verificó HTTP 200 y lo cerró al terminar. `gradio.json` registra:

- Dos candidatos: advertencia `uncertain`, sin clase ni probabilidades.
- Recorte manual `[170,240,245,340]`: inferencia real, predicción 1, tensor 1×784 y coordenadas fuente correctas.
- Recorte inválido: rechazo, sin resultados previos retenidos.
- Imagen vacía: limpieza de todas las salidas.
- Tres fotos de desarrollo por la API nueva: 1,2,4.
- API `/classify`: conserva cuatro salidas y el comportamiento original (foto 4 → 2).

**No se comprobó la apariencia visual en navegador**: no hay navegador ni automatización disponibles. La comprobación HTTP/API y las pruebas de componentes no sustituyen esa inspección humana. Para completarla: iniciá `.venv/bin/python app.py`, abrí la URL local, subí una foto, activá **Usar recorte manual**, introducí coordenadas válidas y pulsá **Reintentar diagnóstico**; verificá la caja azul y que el trazo no se corte. Los controles son numéricos, no una herramienta de dibujo.

## Conservación y límites

El SHA-256 de `models/mnist.keras` permanece `ab1d64d10ac6b5d2b1f39948c643d53197ed89179a128a262ed9a63039477fa3`. Los commits y los resultados anteriores se conservaron. `/tmp/mnist-final-verify.c67Fkr` no existe en esta sesión: no se eliminó nada ni se afirma haber inspeccionado su contenido histórico.

El nuevo evaluador no modifica comandos ni helpers existentes: no hay un comportamiento compartido cambiado que requiera diferencial histórico. Se ejecutaron pruebas funcionales nuevas y la suite existente. El ejecutable `gentle-ai` no está en PATH. La fachada nativa ASSESS devolvió `nativeRisk=unassessable` por requerir una declaración explícita de archivos sin seguimiento; su plan mantiene autoverificación y no pide verificador independiente. No se inició revisión ni se atribuye aprobación nativa.

Pendiente para finalizar la validación del producto: fotografías realmente nuevas y revisión visual del navegador. No se publicó el proyecto.
