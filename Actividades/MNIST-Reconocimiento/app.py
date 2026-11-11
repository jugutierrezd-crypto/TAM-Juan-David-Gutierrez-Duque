"""Interfaz exclusivamente local: importar este módulo no entrena ni lanza servidor."""
import os
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

from functools import lru_cache
import math

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from candidates import compare_candidates
from diagnose import infer, statistics
from photo import preprocess_photo
from predict import load_model, predict_digit, preprocess_image


TENTATIVE_ADVICE = ('Resultado incierto. Si es posible, subí una imagen más nítida, tomada más cerca '
                    'y con mejor iluminación; esto puede ayudar al reconocimiento, pero no garantiza '
                    'una respuesta correcta.')


@lru_cache(maxsize=1)
def get_model():
    return load_model()


def classify(image):
    if image is None:
        return None, None, None, 'Arrastrá una fotografía con un solo dígito.'
    preview = None
    try:
        # Mostrar el preprocesado incluso cuando todavía no hay modelo guardado.
        _, preview = preprocess_image(image)
        digit, scores, preview = predict_digit(image, model=get_model())
        return str(digit), scores, preview, 'Predicción lista. Softmax no garantiza certeza.'
    except (ValueError, FileNotFoundError, OSError) as error:
        return None, None, preview, str(error)


def diagnose_image(image, mode='Fotografía', remove_grid=True, use_clahe=True,
                   manual=False, x0=0, y0=0, x1=0, y1=0, *, robust=False):
    """Seven diagnostic outputs; the four-output compatibility route stays separate.

    Coordinates are EXIF-oriented source pixels, with exclusive right/bottom.
    No stages, tensors or uploads are written to project files here.
    """
    if image is None:
        return None, None, None, 'Arrastrá una fotografía con un solo dígito.', [], None, {}
    source_description = ''
    try:
        source = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
        oriented = ImageOps.exif_transpose(source)
        source_description = f'Original con orientación EXIF corregida: {oriented.width}×{oriented.height} píxeles. '
        if mode not in ('Fotografía', 'Original'):
            raise ValueError('Elegí Fotografía u Original.')
        if manual and mode == 'Original':
            raise ValueError('El recorte manual requiere el modo Fotografía.')
        if mode == 'Fotografía':
            crop = (x0, y0, x1, y1) if manual else None
            result = preprocess_photo(image, crop=crop, remove_grid=remove_grid, use_clahe=use_clahe, robust=robust)
            tensor, preview, stages, quality = result.tensor, result.preview, result.stages, result.quality
        else:
            stages = {}
            tensor, preview = preprocess_image(image, trace=stages)
            # Original-mode trace is source-sized, already EXIF corrected.
            rgba = oriented.convert('RGBA')
            original = np.asarray(Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB'))
            stages = dict(original=original, **stages)
            height, width = stages['gray'].shape
            stages['coordinates'] = dict(source_size=[width, height], source_crop=[0, 0, width, height],
                                         working_size=[width, height], source_per_working_pixel=[1, 1],
                                         exif_corrected=True)
            quality = dict(status='unassessed', warning='Recorte global sin aislamiento validado.',
                           scores_calibrated=False, manual_crop=False)
        captions = {
            'original': 'original: orientación EXIF corregida; transparencia sobre blanco',
            'gray': 'gray: escala de grises de trabajo',
            'smoothed': 'smoothed: suavizado gaussiano de textura antes de estimar iluminación',
            'illumination': 'illumination: corrección de iluminación',
            'contrast': 'contrast: CLAHE',
            'lines': 'lines: evidencia de líneas largas (no máscara del dígito)',
            'grid_removed': 'grid_removed: resta conservadora de líneas',
            'adaptive_mask': 'adaptive_mask: umbral adaptativo; evidencia, no selección final',
            'robust_ink': 'robust_ink: apertura gris 5×5 para suprimir líneas finas antes de histéresis',
            'seeds': 'seeds: semillas de tinta antes de CLAHE',
            'hysteresis_mask': 'hysteresis_mask: crecimiento de componentes plausibles',
            'mask': 'mask: candidato seleccionado heurísticamente',
            'crop': 'crop: región de tinta seleccionada',
            'final': 'final: vista exacta del tensor 28×28 sin normalizar',
        }
        if mode == 'Original' or not use_clahe:
            captions['contrast'] = 'contrast: CLAHE no aplicado'
        if mode == 'Original' or not remove_grid:
            captions['grid_removed'] = 'grid_removed: eliminación de cuadrícula no aplicado'
        gallery = [(stages[key], caption) for key, caption in captions.items() if key in stages]
        coordinates = stages['coordinates']
        ox, oy, _, _ = coordinates['source_crop']
        sx, sy = coordinates['source_per_working_pixel']
        bx0, by0, bx1, by1 = stages['bbox']
        source_bbox = [math.floor(ox + bx0 * sx), math.floor(oy + by0 * sy),
                       math.ceil(ox + bx1 * sx), math.ceil(oy + by1 * sy)]
        overlay = Image.fromarray(stages['original']).convert('RGB')
        draw = ImageDraw.Draw(overlay)
        if manual:
            cx0, cy0, cx1, cy1 = coordinates['source_crop']
            draw.rectangle((cx0, cy0, cx1 - 1, cy1 - 1), outline='blue', width=2)
        left, top, right, bottom = source_bbox
        draw.rectangle((left, top, right - 1, bottom - 1), outline='red', width=2)
        report = dict(mode=mode, coordinates=coordinates, source_bbox=source_bbox, quality=quality,
                      statistics=statistics(preview), tensor=tensor.tolist(),
                      tensor_shape=list(tensor.shape), tensor_dtype=str(tensor.dtype),
                      stages={key: caption for key, caption in captions.items() if key in stages})
        warning = quality.get('warning', '')
        if quality['status'] == 'uncertain':
            return (None, None, preview,
                    '⚠ ADVERTENCIA: aislamiento incierto. ' + source_description + warning +
                    ' Usá un recorte manual y pulsá Reintentar diagnóstico.', gallery, overlay, report)
        status = ('Candidato heurístico; inspeccioná las etapas. Softmax no es confianza calibrada. '
                  + source_description + warning)
        try:
            model = get_model()
        except FileNotFoundError as error:
            return None, None, preview, '⚠ Sin predicción: ' + source_description + str(error), gallery, overlay, report
        digit, scores = infer(model, tensor)
        return str(digit), scores, preview, status, gallery, overlay, report
    except Exception as error:
        # An event always replaces every output, including after a model failure.
        return (None, None, None, '⚠ ADVERTENCIA: ' + source_description + str(error) +
                ' Revisá la imagen y el recorte manual; pulsá Reintentar diagnóstico.', [], None, {})


def analyze_image(image, mode='Fotografía', remove_grid=True, use_clahe=True,
                  manual=False, x0=0, y0=0, x1=0, y1=0):
    """Automatic UI route; preserve the existing seven-output diagnostic API."""
    result = diagnose_image(image, mode, remove_grid, use_clahe, manual, x0, y0, x1, y1)
    quality = result[6].get('quality', {})
    robust = (mode == 'Fotografía' and not manual and not result[6]
              and 'No se pudo aislar un dígito' in result[3])
    if robust or (mode == 'Fotografía' and not manual and quality.get('status') == 'uncertain'):
        comparison = dict(selected_crop=None, candidates=[], reason='model_error', scores_calibrated=False)
        try:
            comparison = compare_candidates(image, get_model(), remove_grid=remove_grid, use_clahe=use_clahe,
                                            robust=robust, allow_tentative=True, allow_alternatives=True)
            # At most the original comparison plus one robust comparison. A
            # model error or failed final revalidation is never a retry signal.
            if (not robust and comparison['selected_crop'] is None
                    and comparison['reason'] in ('no_candidates', 'too_many_candidates',
                                                  'unrecoverable_regions', 'no_clear_winner')):
                previous = comparison
                robust = True
                comparison = compare_candidates(image, get_model(), remove_grid=remove_grid,
                                                use_clahe=use_clahe, robust=True, allow_tentative=True,
                                                allow_alternatives=True)
                comparison['previous_attempts'] = [previous]
            crop = comparison['selected_crop']
            if crop is not None:
                chosen = next(c for c in comparison['candidates'] if c['crop'] == crop)
                alternatives = comparison['prediction_status'] == 'alternatives'
                base_digit = chosen['base_digit'] if alternatives else chosen['digit']
                page_context = robust and comparison.get('evaluation_context') == 'page'
                if page_context:
                    final = diagnose_image(image, mode, remove_grid, use_clahe, manual=False, robust=True)
                else:
                    final = diagnose_image(image, mode, remove_grid, use_clahe, True, *crop, robust=robust)
                # Class agreement alone cannot validate changed ink or tensors,
                # whether the selected context is a page or a local crop.
                context_matches = (
                    final[6].get('quality') == chosen['quality']
                    and final[6].get('source_bbox') == chosen['source_bbox']
                    and final[6].get('coordinates') == chosen['coordinates']
                    and final[6].get('coordinates', {}).get('source_crop') == crop
                    and np.array_equal(final[6].get('tensor'), chosen['tensor'])
                    and np.array_equal(final[2], np.asarray(chosen['tensor']).reshape(28, 28)))
                if (context_matches and final[6].get('quality', {}).get('status') == 'candidate'
                        and final[0] is not None and final[0] == str(base_digit)):
                    selection_status = ('Aislamiento automático en contexto de página. ' if page_context
                                        else 'Recorte automático por comparación heurística. ')
                    result = (*final[:3], selection_status + final[3], *final[4:])
                    result[6]['crop_selection'] = 'automatic'
                    result[6].update(prediction_status='alternatives' if alternatives else 'tentative',
                                     min_probability=chosen['min_probability'], min_margin=chosen['min_margin'])
                    # Strong Softmax after any quality recovery is still uncertain;
                    # alternatives do not designate any class as the winner.
                    if alternatives:
                        digits = comparison['alternative_digits']
                        names = ', '.join(str(digit) for digit in digits[:-1]) + ' o ' + str(digits[-1])
                        result[6].update(base_digit=base_digit, alternative_digits=digits,
                                         raw_probabilities=chosen['raw_probabilities'], views=chosen['views'],
                                         scores_calibrated=False)
                        result = ('Resultado incierto: podría ser ' + names, None, result[2],
                                  '⚠ ADVERTENCIA: resultado incierto. ' + TENTATIVE_ADVICE + ' ' + result[3],
                                  *result[4:])
                    else:
                        result[6]['digit'] = chosen['digit']
                        result = (f'Posible {final[0]}', result[1], result[2],
                                  '⚠ ADVERTENCIA: resultado incierto. ' + TENTATIVE_ADVICE + ' ' + result[3],
                                  *result[4:])
                else:
                    comparison.update(selected_crop=None, reason='final_diagnostic_mismatch',
                                      final_status=final[3])
                    result = (*result[:3], result[3] + ' Revalidación del recorte: ' + final[3], *result[4:])
        except Exception as error:
            comparison.update(selected_crop=None, reason='model_error', error=str(error))
            result = (*result[:3], result[3] + ' Error al comparar candidatos: ' + str(error), *result[4:])
        if robust:
            comparison.update(fallback_used=True, method='smoothed_uncapped_hysteresis')
        result[6]['candidate_comparison'] = comparison
        quality = result[6].get('quality', {})
    question = ''
    if result[6].get('prediction_status') in ('tentative', 'alternatives'):
        question = ('¿Podés subir una imagen más nítida, más cercana y mejor iluminada '
                    'para revisar este resultado incierto?')
        gr.Warning(TENTATIVE_ADVICE, duration=12, title='Resultado incierto')
    elif quality.get('status') == 'uncertain':
        question = ('¿Podés indicar un recorte que contenga solo el dígito completo, '
                    'con un pequeño margen, o subir una foto más cercana?')
        if quality.get('ambiguity_ratio', 0) >= .35:
            question = ('Veo varios trazos posibles. ¿Cuál querés reconocer? '
                        'Indicá un recorte alrededor de ese único dígito, con margen.')
    elif image is not None and any(message in result[3] for message in (
            'No se pudo aislar un dígito', 'No se distingue un dígito', 'El recorte debe')):
        question = ('No pude aislar el dígito. ¿Podés subir una foto más cercana y nítida '
                    'o indicar un recorte manual que incluya el trazo completo?')
    return (*result, gr.update(visible=bool(question)), question)


def analyze_upload(image):
    """Forget the previous photograph's options and manual coordinates."""
    return (*analyze_image(image), 'Fotografía', True, True, False, 0, 0, 0, 0)


def build_app():
    with gr.Blocks(title='MNIST local', analytics_enabled=False) as demo:
        gr.Markdown('# Reconocimiento local de dígitos\nArrastrá una fotografía de un solo dígito (0–9). No se entrena desde esta interfaz.')
        image = gr.Image(type='pil', sources=['upload'], image_mode='RGBA', label='Arrastrá tu fotografía aquí')
        gr.Markdown('## Diagnóstico de imagen automático\n'
                    'Subí la foto: analizo y recorto el dígito automáticamente. '
                    'Si el aislamiento queda sin resolver, reintento una sola vez con suavizado '
                    'de textura y semillas fuertes. Toda recuperación de calidad se muestra como incierta. '
                    'Si hay varios trazos, comparo su geometría y la estabilidad de la predicción; '
                    'elijo solo si hay una ventaja clara. Con evidencia moderada muestro Posible N '
                    'como resultado incierto y un aviso para mejorar la imagen. '
                    'Si un único trazo válido alterna entre dos o tres clases con evidencia moderada, '
                    'muestro esas alternativas sin certificar ninguna y con el mismo aviso. '
                    'Si hay empate entre regiones o falta evidencia, te pido ayuda. '
                    'Sin dudas detectadas muestro el resultado directamente; eso no garantiza acierto. '
                    'Cada foto nueva se analiza desde cero en modo Fotografía.')
        with gr.Group(visible=False) as assistance:
            question = gr.Markdown('')
            gr.Markdown('### Ayuda con el recorte\n'
                        'Podés subir otra foto o activar el recorte en modo Fotografía. '
                        'Usá las dimensiones del original orientado que muestra el estado. '
                        'Origen arriba a la izquierda; x1 e y1 exclusivos. Incluí el dígito completo '
                        'y un margen. Ingresá las coordenadas y pulsá **Reintentar diagnóstico**. '
                        'Azul: recorte usado (manual o automático); rojo: candidato. '
                        'El JSON distingue la selección automática. Los detalles están debajo.')
            manual = gr.Checkbox(value=False, label='Usar recorte manual')
            with gr.Row():
                x0 = gr.Number(value=0, label='x0')
                y0 = gr.Number(value=0, label='y0')
                x1 = gr.Number(value=0, label='x1')
                y1 = gr.Number(value=0, label='y1')
        with gr.Accordion('Opciones avanzadas (opcionales)', open=False):
            gr.Markdown('No necesitás configurar estas opciones para subir una foto. '
                        'Para comparar un procesamiento, elegí opciones y pulsá Reintentar diagnóstico. '
                        'Los cambios de opciones no recalculan solos.')
            with gr.Row():
                mode = gr.Radio(['Fotografía', 'Original'], value='Fotografía', label='Preprocesamiento')
                remove_grid = gr.Checkbox(value=True, label='Eliminar cuadrícula (compará visualmente)')
                use_clahe = gr.Checkbox(value=True, label='Aplicar CLAHE')
        retry = gr.Button('Reintentar diagnóstico')
        status = gr.Textbox(label='Estado y advertencias', interactive=False)
        with gr.Row():
            digit = gr.Textbox(label='Dígito estimado (no certificado)', interactive=False)
            probabilities = gr.Label(num_top_classes=10, label='Probabilidades para 0–9 (no calibradas)')
            preview = gr.Image(type='numpy', image_mode='L', label='Tensor procesado: 28×28, intensidades 0..255', interactive=False)
        with gr.Accordion('Detalles del diagnóstico (opcionales)', open=False):
            overlay = gr.Image(type='pil', label='Original orientado: cajas en píxeles de origen', interactive=False)
            gallery = gr.Gallery(label='Etapas reales de procesamiento', columns=3, type='numpy')
            report = gr.JSON(label='Estadísticas, coordenadas, calidad heurística y tensor exacto')
        inputs = [image, mode, remove_grid, use_clahe, manual, x0, y0, x1, y1]
        outputs = [digit, probabilities, preview, status, gallery, overlay, report]
        automatic_outputs = [*outputs, assistance, question]
        # Uploads deliberately ignore old crop/options; a new photo starts fresh.
        image.change(analyze_upload, inputs=image, outputs=[*automatic_outputs, *inputs[1:]], api_name='analyze_image',
                     trigger_mode='always_last', concurrency_id='diagnosis')
        retry.click(analyze_image, inputs=inputs, outputs=automatic_outputs, api_name='resolve_image',
                    concurrency_id='diagnosis')
        # Preserve all consumed API signatures behind hidden compatibility triggers.
        diagnostic = gr.Button('Diagnóstico compatible', visible=False)
        diagnostic.click(diagnose_image, inputs=inputs, outputs=outputs, api_name='diagnose_image',
                         concurrency_id='diagnosis')
        retry_legacy = gr.Button('Reintento compatible', visible=False)
        retry_legacy.click(diagnose_image, inputs=inputs, outputs=outputs, api_name='retry_diagnosis',
                           concurrency_id='diagnosis')
        legacy = gr.Button('Compatibilidad original', visible=False)
        legacy.click(classify, inputs=image, outputs=[digit, probabilities, preview, status], api_name='classify')
    return demo


if __name__ == '__main__':
    build_app().launch(server_name='127.0.0.1', share=False, inbrowser=False)
