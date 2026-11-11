"""Public UI contracts, using generated strokes rather than photographic labels."""
import ast
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

import app
from photo import preprocess_photo
from predict import preprocess_image


def scene():
    image = Image.new('RGB', (400, 600), 'white')
    draw = ImageDraw.Draw(image)
    draw.line((205, 250, 205, 325), fill='black', width=7)
    draw.line((180, 325, 230, 325), fill='black', width=7)
    return image


@pytest.fixture
def model(monkeypatch):
    calls = []

    def predict(tensor, training=False):
        calls.append(tensor.copy())
        return np.eye(10, dtype=np.float32)[[1]]

    monkeypatch.setattr(app, 'get_model', lambda: predict)
    return calls


def test_diagnostic_stages_exact_tensor_and_source_overlay(model):
    image = scene()
    digit, scores, preview, status, gallery, overlay, report = app.diagnose_image(image)
    expected = preprocess_photo(image)
    assert digit == '1' and scores == {str(i): float(i == 1) for i in range(10)}
    assert 'heurístico' in status and 'calibrada' in status
    np.testing.assert_array_equal(model[0], expected.tensor)
    np.testing.assert_array_equal(preview, expected.preview)
    np.testing.assert_array_equal(report['tensor'], model[0])
    assert report['statistics']['active_fraction'] == float((preview > 15).mean())
    assert report['tensor_dtype'] == 'float32' and report['tensor_shape'] == [1, 784]
    assert len(gallery) == 12
    for (array, caption), name in zip(gallery, expected.stages):
        if name == 'bbox':
            break
        np.testing.assert_array_equal(array, expected.stages[name])
        assert name in caption
    assert overlay.size == image.size
    assert report['coordinates']['source_size'] == [400, 600]
    x0, y0, x1, y1 = report['source_bbox']
    assert 175 <= x0 <= 205 and x1 >= 230 and y0 <= 250 and y1 >= 325
    assert overlay.getpixel((x0, y0)) == (255, 0, 0)


def test_disabled_operations_and_legacy_mode_are_honest(model):
    result = app.diagnose_image(scene(), remove_grid=False, use_clahe=False)
    captions = dict((caption.split(':')[0], caption) for _, caption in result[4])
    assert 'no aplicado' in captions['grid_removed']
    assert 'no aplicado' in captions['contrast']
    legacy = app.diagnose_image(scene(), mode='Original')
    tensor, preview = preprocess_image(scene())
    np.testing.assert_array_equal(model[-1], tensor)
    np.testing.assert_array_equal(legacy[2], preview)
    assert 'no aplicado' in next(c for _, c in legacy[4] if c.startswith('contrast:'))
    assert legacy[6]['quality']['status'] == 'unassessed'
    colored = scene()
    ImageDraw.Draw(colored).rectangle((0, 0, 5, 5), fill='red')
    original = app.diagnose_image(colored, mode='Original')[4][0][0]
    np.testing.assert_array_equal(original, np.asarray(colored))


def test_uncertain_manual_rejected_and_empty_clear_previous_outputs(model):
    image = scene()
    image.paste(image.crop((170, 240, 245, 340)), (55, 240))
    uncertain = app.diagnose_image(image)
    assert uncertain[0:2] == (None, None)
    assert 'ADVERTENCIA' in uncertain[3] and 'recorte manual' in uncertain[3]
    assert uncertain[4] and uncertain[6]['quality']['status'] == 'uncertain'
    manual = app.diagnose_image(image, manual=True, x0=170, y0=240, x1=245, y1=340)
    assert manual[6]['quality']['manual_crop'] is True
    assert manual[6]['coordinates']['source_crop'] == [170, 240, 245, 340]
    assert manual[0] == '1'
    for source, options in [(None, {}), (Image.new('RGB', (400, 600), 'white'), {}),
                            (image, dict(manual=True, x0=-1, y0=0, x1=50, y1=50)),
                            (image, dict(manual=True, x0=0.5, y0=0, x1=50, y1=50))]:
        result = app.diagnose_image(source, **options)
        assert result[:3] == (None, None, None)
        assert result[4:] == ([], None, {})
        assert 'recorte manual' in result[3] or source is None
        if source is not None:
            assert '400×600' in result[3] and 'EXIF' in result[3]
    assert len(model) == 1  # uncertain/rejected inputs never call the model


@pytest.mark.parametrize('digit', ['1', '4'])
def test_manual_crop_preserves_horizontal_foot_and_crossing(model, digit):
    image = scene()
    if digit == '4':
        image = Image.new('RGB', (400, 600), 'white')
        draw = ImageDraw.Draw(image)
        draw.line((180, 250, 180, 290, 230, 290), fill='black', width=7)
        draw.line((230, 245, 230, 325), fill='black', width=7)
    result = app.diagnose_image(image, manual=True, x0=170, y0=240, x1=245, y1=340)
    mask = next(a for a, caption in result[4] if caption.startswith('mask:'))
    assert mask[85 if digit == '1' else 50, 20:55].mean() > 200
    image.getexif()[274] = 6
    oriented = app.diagnose_image(image, manual=True, x0=240, y0=170, x1=340, y1=245)
    assert oriented[6]['coordinates']['source_size'] == [600, 400]
    assert oriented[5].size == (600, 400)


def test_missing_and_invalid_model_clear_predictions_keep_only_valid_evidence(monkeypatch):
    def missing():
        raise FileNotFoundError('modelo ausente')
    monkeypatch.setattr(app, 'get_model', missing)
    result = app.diagnose_image(scene())
    assert result[:2] == (None, None)
    assert result[2] is not None and result[4] and result[6]['statistics']
    assert 'modelo ausente' in result[3]
    monkeypatch.setattr(app, 'get_model', lambda: lambda *a, **kw: np.ones((1, 10)))
    invalid = app.diagnose_image(scene())
    assert invalid[:3] == (None, None, None) and invalid[4:] == ([], None, {})
    assert 'ADVERTENCIA' in invalid[3]


def test_legacy_classify_still_has_four_results(model):
    result = app.classify(scene())
    assert len(result) == 4 and result[0] == '1'
    np.testing.assert_array_equal(result[2], preprocess_image(scene())[1])
    assert result[3] == 'Predicción lista. Softmax no garantiza certeza.'
    assert app.classify(None) == (None, None, None, 'Arrastrá una fotografía con un solo dígito.')
    rejected = app.classify(Image.new('RGB', (30, 30), 'white'))
    assert rejected == (None, None, None, 'Imagen sin contenido: no se distingue un dígito del fondo.')


def test_automatic_analysis_does_not_ask_for_clear_image(model):
    upload = app.analyze_upload(scene())
    assert upload[9:] == ('Fotografía', True, True, False, 0, 0, 0, 0)
    result = upload[:9]
    assert len(result) == 9 and result[0] == '1'
    assert result[6]['quality']['status'] == 'candidate'
    assert result[7]['visible'] is False and result[8] == ''
    assert len(model) == 1


def test_automatic_analysis_asks_only_when_image_needs_help(model):
    image = scene()
    image.paste(image.crop((170, 240, 245, 340)), (55, 240))
    result = app.analyze_image(image)
    assert result[:2] == (None, None)
    assert result[6]['quality']['status'] == 'uncertain'
    assert result[7]['visible'] is True
    assert '¿' in result[8] and 'recorte' in result[8]
    assert len(model) == 20  # two bounded comparisons; equal rivals still abstain
    assert result[6]['candidate_comparison']['reason'] == 'no_clear_winner'
    assert result[6]['candidate_comparison']['previous_attempts'][0]['reason'] == 'no_clear_winner'
    resolved = app.analyze_image(image, manual=True, x0=170, y0=240, x1=245, y1=340)
    assert resolved[0] == '1' and resolved[6]['quality']['manual_crop']
    assert resolved[7]['visible'] is False and resolved[8] == ''
    assert len(model) == 21
    rejected = app.analyze_image(Image.new('RGB', (400, 600), 'white'))
    assert rejected[:3] == (None, None, None)
    assert rejected[7]['visible'] is True and '¿' in rejected[8]
    assert len(model) == 21


def test_automatic_analysis_clears_help_and_does_not_ask_about_model(monkeypatch, model):
    empty = app.analyze_image(None)
    assert empty[:3] == (None, None, None) and empty[4:7] == ([], None, {})
    assert empty[7]['visible'] is False and empty[8] == ''
    def missing():
        raise FileNotFoundError('modelo ausente')
    monkeypatch.setattr(app, 'get_model', missing)
    result = app.analyze_image(scene())
    assert result[:2] == (None, None) and 'modelo ausente' in result[3]
    assert result[7]['visible'] is False and result[8] == ''


def ambiguous_scene():
    image = scene()
    image.paste(image.crop((170, 240, 245, 340)), (55, 240))
    return image


def test_automatic_comparison_selects_and_rebuilds_consistent_diagnostic(monkeypatch):
    calls, warnings = [], []
    monkeypatch.setattr(app.gr, 'Warning', lambda message, **kw: warnings.append((message, kw)))
    def predict(tensor, training=False):
        calls.append(tensor.copy())
        peak = .99 if len(calls) <= 5 or len(calls) > 10 else .4
        output = np.full((1, 10), (1 - peak) / 9, dtype=np.float32)
        output[0, 1] = peak
        return output
    monkeypatch.setattr(app, 'get_model', lambda: predict)
    result = app.analyze_image(ambiguous_scene())
    report = result[6]
    comparison = report['candidate_comparison']
    assert result[0] == 'Posible 1' and result[7]['visible'] is True
    assert app.TENTATIVE_ADVICE in result[3]
    assert warnings == [(app.TENTATIVE_ADVICE, {'duration': 12, 'title': 'Resultado incierto'})]
    assert report['prediction_status'] == 'tentative' and comparison['prediction_status'] == 'accepted'
    assert report['crop_selection'] == 'automatic'
    assert report['coordinates']['source_crop'] == comparison['selected_crop']
    assert report['quality']['status'] == 'candidate'
    assert report['quality']['manual_crop'] is True  # internal geometry, not user action
    assert 'automático' in result[3] and 'calibrada' in result[3]
    np.testing.assert_array_equal(report['tensor'], calls[-1])
    expected = preprocess_photo(ambiguous_scene(), crop=comparison['selected_crop'])
    np.testing.assert_array_equal(result[2], expected.preview)
    assert len(calls) == 11


@pytest.mark.parametrize('failure', ['mismatch', 'missing', 'invalid', 'uncertain'])
def test_comparison_failure_retains_warning_and_help(monkeypatch, failure):
    from candidates import compare_candidates as real_compare
    def compare(image, model, **kwargs):
        evidence = real_compare(image, model, **kwargs)
        if failure == 'mismatch':
            next(c for c in evidence['candidates'] if c['eligible'])['digit'] = 9
        if failure == 'uncertain':
            winner = next(c for c in evidence['candidates'] if c['eligible'])
            winner['crop'] = [0, 0, 400, 600]
            evidence['selected_crop'] = winner['crop']
        return evidence
    monkeypatch.setattr(app, 'compare_candidates', compare)
    calls = []
    def predict(tensor, training=False):
        calls.append(tensor)
        if failure == 'invalid':
            return np.ones((1, 10))
        peak = .99 if len(calls) <= 5 or len(calls) > 10 else .4
        output = np.full((1, 10), (1 - peak) / 9, dtype=np.float32)
        output[0, 1] = peak
        return output
    def get_model():
        if failure == 'missing':
            raise FileNotFoundError('modelo ausente')
        return predict
    monkeypatch.setattr(app, 'get_model', get_model)
    result = app.analyze_image(ambiguous_scene())
    assert result[:2] == (None, None) and result[7]['visible'] is True
    assert 'ADVERTENCIA' in result[3]
    assert result[6]['quality']['status'] == 'uncertain'
    comparison = result[6]['candidate_comparison']
    assert comparison['selected_crop'] is None
    if failure == 'missing':
        assert 'modelo ausente' in result[3]
    elif failure == 'invalid':
        assert 'Softmax inválida' in result[3]
    else:
        assert comparison['reason'] == 'final_diagnostic_mismatch'


def test_manual_and_original_routes_never_compare(monkeypatch, model):
    def forbidden(*args, **kwargs):
        pytest.fail('comparison is only for automatic uncertain multi-component photos')
    monkeypatch.setattr(app, 'compare_candidates', forbidden)
    manual = app.analyze_image(ambiguous_scene(), manual=True, x0=170, y0=240, x1=245, y1=340)
    assert manual[0] == '1' and 'candidate_comparison' not in manual[6]
    original = app.analyze_image(ambiguous_scene(), mode='Original')
    assert original[0] == '1' and 'candidate_comparison' not in original[6]
    assert app.analyze_image(scene())[0] == '1'
    assert app.analyze_image(Image.new('RGB', (400, 600), 'white'))[0] is None


def test_automatic_extraction_failure_retries_robust_with_consistent_evidence(model):
    import json
    from test_photo import textured_scene
    image = textured_scene()
    legacy = app.diagnose_image(image)
    assert len(legacy) == 7 and legacy[0] is None and not legacy[6]
    result = app.analyze_image(image)
    assert len(result) == 9 and result[0] == 'Posible 1' and result[7]['visible']
    assert app.TENTATIVE_ADVICE in result[3]
    report = result[6]
    assert report['quality']['fallback_used'] is True
    assert report['quality']['method'] == 'smoothed_uncapped_hysteresis'
    assert report['crop_selection'] == 'automatic'
    assert report['candidate_comparison']['reason'] == 'clear_winner'
    assert report['candidate_comparison']['evaluation_context'] == 'page'
    assert report['coordinates']['source_crop'] == [0, 0, image.width, image.height]
    assert report['quality']['manual_crop'] is False
    page = preprocess_photo(image, robust=True)
    np.testing.assert_array_equal(report['tensor'], page.tensor)
    np.testing.assert_array_equal(model[0], page.tensor)
    assert report['source_bbox'] == page.stages['bbox']
    assert 'smoothed' in report['stages']
    assert 132 <= report['source_bbox'][0] <= 140
    assert 330 <= report['source_bbox'][3] <= 338
    np.testing.assert_array_equal(report['tensor'], model[-1])
    np.testing.assert_array_equal(result[2].reshape(1, 784), report['tensor'])
    assert result[2].dtype == np.uint8 and len(model) == 6
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('policy', ['strong', 'tentative'])
@pytest.mark.parametrize('changed', ['tensor', 'preview', 'class', 'quality', 'bbox', 'context'])
def test_page_context_final_diagnostic_mismatch_abstains(monkeypatch, model, changed, policy):
    from test_photo import textured_scene
    warnings = []
    monkeypatch.setattr(app.gr, 'Warning', lambda *args, **kwargs: warnings.append(args))
    if policy == 'tentative':
        def predict(tensor, training=False):
            output = np.full((1, 10), .26 / 9, dtype=np.float32)
            output[0, 1] = .74
            return output
        monkeypatch.setattr(app, 'get_model', lambda: predict)
    real_diagnose = app.diagnose_image
    def diagnose(*args, **kwargs):
        final = real_diagnose(*args, **kwargs)
        if kwargs.get('robust'):
            if changed == 'tensor':
                final[6]['tensor'][0][0] += 1
            elif changed == 'preview':
                final[2][0, 0] += 1
            elif changed == 'class':
                final = ('9', *final[1:])
            elif changed == 'quality':
                final[6]['quality']['component_count'] = 2
            elif changed == 'bbox':
                final[6]['source_bbox'][0] += 1
            else:
                final[6]['coordinates']['source_crop'][0] += 1
        return final
    monkeypatch.setattr(app, 'diagnose_image', diagnose)
    result = app.analyze_image(textured_scene())
    assert result[:3] == (None, None, None) and result[7]['visible'] is True
    comparison = result[6]['candidate_comparison']
    assert comparison['selected_crop'] is None
    assert comparison['reason'] == 'final_diagnostic_mismatch'
    assert 'ADVERTENCIA' in result[3]
    assert warnings == []


def test_tentative_page_result_has_exact_advice_and_consistent_evidence(monkeypatch):
    import json
    from test_photo import textured_scene
    image = textured_scene()
    calls, warnings = [], []
    def predict(tensor, training=False):
        calls.append(tensor.copy())
        peak = .741 if len(calls) == 3 else .99
        output = np.full((1, 10), (1 - peak) / 9, dtype=np.float32)
        output[0, 1] = peak
        return output
    monkeypatch.setattr(app, 'get_model', lambda: predict)
    monkeypatch.setattr(app.gr, 'Warning', lambda message, **kwargs: warnings.append((message, kwargs)))
    result = app.analyze_image(image)
    assert len(result) == 9 and result[0] == 'Posible 1'
    assert 'resultado incierto' in result[3] and 'no es confianza calibrada' in result[3]
    advice = ('Resultado incierto. Si es posible, subí una imagen más nítida, tomada más cerca '
              'y con mejor iluminación; esto puede ayudar al reconocimiento, pero no garantiza '
              'una respuesta correcta.')
    assert warnings == [(advice, {'duration': 12, 'title': 'Resultado incierto'})]
    assert advice in result[3]
    report = result[6]
    assert report['digit'] == 1 and report['prediction_status'] == 'tentative'
    assert report['min_probability'] == pytest.approx(.741)
    assert report['quality']['status'] == 'candidate' and report['quality']['scores_calibrated'] is False
    comparison = report['candidate_comparison']
    assert comparison['reason'] == 'tentative_winner' and comparison['evaluation_context'] == 'page'
    assert not comparison['candidates'][0]['eligible']
    page = preprocess_photo(image, robust=True)
    np.testing.assert_array_equal(report['tensor'], page.tensor)
    np.testing.assert_array_equal(result[2], page.preview)
    np.testing.assert_array_equal(calls[-1], page.tensor)
    assert report['source_bbox'] == page.stages['bbox']
    assert report['coordinates'] == page.stages['coordinates']
    assert len(calls) == 6 and result[1]['1'] > .98
    assert result[7]['visible'] is True and '¿' in result[8]
    json.dumps(report, allow_nan=False)


def test_popup_is_absent_for_clear_empty_tied_and_unresolved_inputs(monkeypatch, model):
    from test_photo import textured_scene
    warnings = []
    monkeypatch.setattr(app.gr, 'Warning', lambda *args, **kwargs: warnings.append(args))
    assert app.analyze_image(scene())[0] == '1'
    for image in (None, Image.new('RGB', (400, 600), 'white'), ambiguous_scene(), textured_scene(0)):
        result = app.analyze_image(image)
        assert result[0] is None
    assert warnings == []


def test_tentative_local_context_revalidates_tensor_before_popup(monkeypatch):
    calls, warnings = [], []
    def predict(tensor, training=False):
        calls.append(tensor.copy())
        peak = .3 if 5 < len(calls) <= 10 else .74
        output = np.full((1, 10), (1 - peak) / 9, dtype=np.float32)
        output[0, 1] = peak
        return output
    monkeypatch.setattr(app, 'get_model', lambda: predict)
    monkeypatch.setattr(app.gr, 'Warning', lambda *args, **kwargs: warnings.append(args))
    result = app.analyze_image(ambiguous_scene())
    assert result[0] == 'Posible 1' and len(warnings) == 1
    np.testing.assert_array_equal(result[6]['tensor'], calls[-1])
    assert result[6]['quality']['manual_crop'] is True
    real_diagnose = app.diagnose_image
    def inconsistent(*args, **kwargs):
        final = real_diagnose(*args, **kwargs)
        if len(args) > 4 and args[4]:
            final[6]['tensor'][0][0] += 1
        return final
    monkeypatch.setattr(app, 'diagnose_image', inconsistent)
    calls.clear()
    warnings.clear()
    rejected = app.analyze_image(ambiguous_scene())
    assert rejected[:2] == (None, None) and rejected[7]['visible'] is True
    assert rejected[6]['candidate_comparison']['reason'] == 'final_diagnostic_mismatch'
    assert warnings == []


def test_page_context_unstable_prediction_with_changed_base_keeps_help(monkeypatch):
    from test_photo import textured_scene
    calls = []
    def predict(tensor, training=False):
        calls.append(tensor)
        return np.eye(10, dtype=np.float32)[[len(calls) % 2]]
    monkeypatch.setattr(app, 'get_model', lambda: predict)
    result = app.analyze_image(textured_scene())
    assert result[:2] == (None, None) and result[7]['visible'] is True
    assert result[6]['candidate_comparison']['evaluation_context'] == 'page'
    assert result[6]['candidate_comparison']['selected_crop'] is None
    assert result[6]['candidate_comparison']['reason'] == 'final_diagnostic_mismatch'
    assert len(calls) == 6  # alternative crop is rejected when the base inference changes


def test_robust_automatic_ties_grid_and_model_errors_never_force_prediction(monkeypatch, model):
    from test_photo import textured_scene
    for image in (textured_scene(2), textured_scene(0)):
        result = app.analyze_image(image)
        assert result[:2] == (None, None) and result[7]['visible'] is True
    assert len(model) == 10
    def broken():
        raise FileNotFoundError('modelo ausente')
    monkeypatch.setattr(app, 'get_model', broken)
    failure = app.analyze_image(textured_scene())
    assert failure[:2] == (None, None) and 'modelo ausente' in failure[3]
    assert failure[6]['candidate_comparison']['reason'] == 'model_error'


def recovery_scene(width=1):
    from test_photo import textured_scene
    image = textured_scene()
    draw = ImageDraw.Draw(image)
    for pos in range(15, 600, 25):
        draw.line((0, pos, 500, pos + 8), fill=(160,) * 3, width=width)
        draw.line((pos, 0, pos + 8, 600), fill=(160,) * 3, width=width)
    return image


@pytest.mark.parametrize('problem', ['dispersion', 'too_many_candidates', 'no_clear_winner'])
def test_unresolved_quality_retries_once_and_labels_even_strong_recovery(monkeypatch, model, problem):
    image = recovery_scene(2)
    real_diagnose, real_compare = app.diagnose_image, app.compare_candidates
    attempts, warnings = [], []
    def diagnose(*args, **kwargs):
        result = real_diagnose(*args, **kwargs)
        if problem == 'dispersion' and args[0] is image and not kwargs.get('robust'):
            # Isolate the routing rule: low-ratio dispersion is still unresolved
            # quality, not an instruction to skip all recovery.
            result[6]['quality'].update(ambiguity_ratio=.01,
                                       warning='El candidato es disperso; revisá el recorte.')
        return result
    def compare(*args, **kwargs):
        evidence = real_compare(*args, **kwargs)
        if problem == 'too_many_candidates' and not kwargs['robust']:
            # Control the routing signal independently of the opt-in geometry
            # gate, whose actual plausible-region budget is tested separately.
            assert evidence['selected_crop'] is None
            evidence['reason'] = 'too_many_candidates'
        attempts.append((kwargs['robust'], evidence['reason']))
        return evidence
    monkeypatch.setattr(app, 'diagnose_image', diagnose)
    monkeypatch.setattr(app, 'compare_candidates', compare)
    monkeypatch.setattr(app.gr, 'Warning', lambda message, **kw: warnings.append((message, kw)))
    result = app.analyze_upload(image)
    assert len(result) == 17 and result[0] == 'Posible 1'
    assert attempts == [(False, 'too_many_candidates' if problem == 'too_many_candidates'
                         else 'no_clear_winner'), (True, 'clear_winner')]
    assert warnings == [(app.TENTATIVE_ADVICE, {'duration': 12, 'title': 'Resultado incierto'})]
    assert app.TENTATIVE_ADVICE in result[3] and 'resultado incierto' in result[3]
    assert result[7]['visible'] is True and '¿' in result[8]
    report = result[6]
    assert report['prediction_status'] == 'tentative' and report['digit'] == 1
    comparison = report['candidate_comparison']
    assert comparison['fallback_used'] is True
    assert comparison['prediction_status'] == 'accepted'  # strong model evidence, uncertain recovery
    assert [a['reason'] for a in comparison['previous_attempts']] == [attempts[0][1]]
    page = preprocess_photo(image, robust=True)
    np.testing.assert_array_equal(result[2], page.preview)
    np.testing.assert_array_equal(report['tensor'], page.tensor)
    np.testing.assert_array_equal(model[-1], page.tensor)
    assert report['coordinates'] == page.stages['coordinates']
    assert report['source_bbox'] == page.stages['bbox']
    assert result[9:] == ('Fotografía', True, True, False, 0, 0, 0, 0)
    for source in (None, Image.new('RGB', image.size, 'white')):
        cleared = app.analyze_upload(source)
        assert cleared[:3] == (None, None, None) and cleared[4:7] == ([], None, {})
        assert cleared[9:] == result[9:]
    warnings.clear()
    clear = app.analyze_upload(scene())
    assert clear[0] == '1' and clear[7]['visible'] is False and clear[8] == ''
    assert 'prediction_status' not in clear[6] and warnings == []


@pytest.mark.parametrize('support', ['weak', 'four_classes', 'invalid', 'tie'])
def test_generalized_recovery_abstains_without_valid_support_and_stops(monkeypatch, support):
    image = recovery_scene()
    if support == 'tie':
        from test_photo import textured_scene
        image = textured_scene(2)
    calls, attempts, warnings = [], [], []
    def predict(tensor, training=False):
        calls.append(tensor.copy())
        if support == 'invalid':
            return np.ones((1, 10), dtype=np.float32)
        peak = .49 if support == 'weak' else .99
        output = np.full((1, 10), (1 - peak) / 9, dtype=np.float32)
        output[0, len(calls) % 4 if support == 'four_classes' else 1] = peak
        return output
    real_compare = app.compare_candidates
    def compare(*args, **kwargs):
        attempts.append(kwargs['robust'])
        return real_compare(*args, **kwargs)
    monkeypatch.setattr(app, 'get_model', lambda: predict)
    monkeypatch.setattr(app, 'compare_candidates', compare)
    monkeypatch.setattr(app.gr, 'Warning', lambda *a, **kw: warnings.append(a))
    result = app.analyze_image(image)
    assert result[:2] == (None, None) and result[7]['visible'] is True
    assert 'ADVERTENCIA' in result[3] and warnings == []
    assert attempts == ([True] if support == 'tie' else [False] if support == 'invalid'
                        else [False, True])
    comparison = result[6]['candidate_comparison']
    assert comparison['selected_crop'] is None
    assert comparison['reason'] == ('model_error' if support == 'invalid' else 'no_clear_winner')
    assert len(calls) == (1 if support == 'invalid' else 10)
    assert 'prediction_status' not in result[6]
    if support == 'invalid':
        assert 'Softmax inválida' in result[3]


@pytest.mark.parametrize('policy', ['strong', 'alternatives'])
def test_original_proposals_robust_local_result_revalidates_exact_context(monkeypatch, model, policy):
    import candidates
    from test_candidates import notebook_marks_scene
    image = notebook_marks_scene()
    real_diagnose, real_preprocess = app.diagnose_image, candidates.preprocess_photo
    def diagnose(*args, **kwargs):
        result = real_diagnose(*args, **kwargs)
        if not kwargs.get('robust') and not args[4]:
            # Controlled initial uncertainty; later local evidence is real.
            result[6]['quality'].update(status='uncertain', warning='Marcas exteriores.')
        return result
    def preprocess(source, **kwargs):
        if kwargs.get('robust') and kwargs.get('crop') is None:
            raise ValueError('No se pudo aislar un dígito; semillas externas dominantes.')
        result = real_preprocess(source, **kwargs)
        if kwargs.get('crop') is not None and not kwargs.get('robust'):
            result.quality.update(status='uncertain', warning='Marcas exteriores.')
        return result
    warnings = []
    if policy == 'alternatives':
        from test_candidates import AlternativeResponses
        model = AlternativeResponses()
        monkeypatch.setattr(app, 'get_model', lambda: model)
        real_compare = app.compare_candidates
        def compare(*args, **kwargs):
            model.calls = 0  # model's generated sequence starts at each comparison
            return real_compare(*args, **kwargs)
        monkeypatch.setattr(app, 'compare_candidates', compare)
    monkeypatch.setattr(app, 'diagnose_image', diagnose)
    monkeypatch.setattr(candidates, 'preprocess_photo', preprocess)
    monkeypatch.setattr(app.gr, 'Warning', lambda message, **kw: warnings.append((message, kw)))
    result = app.analyze_image(image)
    expected = 'Posible 1' if policy == 'strong' else 'Resultado incierto: podría ser 4, 1 o 9'
    assert result[0] == expected and len(result) == 9
    assert warnings == [(app.TENTATIVE_ADVICE, {'duration': 12, 'title': 'Resultado incierto'})]
    assert app.TENTATIVE_ADVICE in result[3] and result[7]['visible'] is True
    report = result[6]
    comparison = report['candidate_comparison']
    assert comparison['proposal_context'] == 'original_regions'
    assert len(comparison['previous_attempts']) == 1
    chosen = next(c for c in comparison['candidates'] if c['crop'] == comparison['selected_crop'])
    assert report['coordinates'] == chosen['coordinates']
    assert report['source_bbox'] == chosen['source_bbox']
    assert report['quality'] == chosen['quality']
    assert report['coordinates']['source_crop'] == chosen['crop']
    np.testing.assert_array_equal(report['tensor'], chosen['tensor'])
    np.testing.assert_array_equal(result[2], np.asarray(chosen['tensor']).reshape(28, 28))
    if policy == 'strong':
        np.testing.assert_array_equal(model[-1], chosen['tensor'])
        assert len(model) == 7  # initial, five robust local views, final revalidation
    else:
        assert result[1] is None and 'digit' not in report
        assert report['alternative_digits'] == [4, 1, 9] and model.calls == 6


@pytest.mark.parametrize('digits,text', [((4, 1, 4, 1, 9), '4, 1 o 9'), ((7, 2, 7, 2, 2), '2 o 7')])
def test_alternatives_show_exact_advice_clear_chart_and_preserve_context(monkeypatch, digits, text):
    from test_candidates import AlternativeResponses
    from test_photo import textured_scene
    model = AlternativeResponses(digits)
    warnings = []
    monkeypatch.setattr(app, 'get_model', lambda: model)
    monkeypatch.setattr(app.gr, 'Warning', lambda message, **kw: warnings.append((message, kw)))
    image = textured_scene()
    result = app.analyze_upload(image)
    assert len(result) == 17 and result[0] == 'Resultado incierto: podría ser ' + text
    assert result[1] is None
    assert app.TENTATIVE_ADVICE in result[3]
    assert warnings == [(app.TENTATIVE_ADVICE, {'duration': 12, 'title': 'Resultado incierto'})]
    report = result[6]
    assert report['prediction_status'] == 'alternatives' and 'digit' not in report
    assert report['base_digit'] == digits[0] and report['scores_calibrated'] is False
    assert len(report['alternative_digits']) == len(set(digits))
    assert [v['digit'] for v in report['views']] == list(digits)
    assert report['raw_probabilities'][str(digits[0])] == pytest.approx(.4)
    page = preprocess_photo(image, robust=True)
    np.testing.assert_array_equal(result[2], page.preview)
    np.testing.assert_array_equal(report['tensor'], page.tensor)
    assert report['coordinates'] == page.stages['coordinates']
    assert report['source_bbox'] == page.stages['bbox']
    assert result[7]['visible'] is True and 'más nítida' in result[8]
    assert model.calls == 6
    warnings.clear()
    cleared = app.analyze_upload(None)
    assert cleared[:3] == (None, None, None) and cleared[4:7] == ([], None, {})
    assert cleared[7]['visible'] is False and cleared[8] == ''
    assert cleared[9:] == result[9:]
    monkeypatch.setattr(app, 'get_model', lambda: lambda *a, **kw: np.eye(10, dtype=np.float32)[[1]])
    clear = app.analyze_upload(scene())
    assert clear[0] == '1' and clear[1] is not None
    assert clear[7]['visible'] is False and clear[8] == ''
    assert 'alternative_digits' not in clear[6] and warnings == []


@pytest.mark.parametrize('changed', ['tensor', 'preview', 'class', 'quality', 'bbox', 'context'])
def test_alternatives_final_revalidation_mismatch_abstains(monkeypatch, changed):
    from test_candidates import AlternativeResponses
    from test_photo import textured_scene
    monkeypatch.setattr(app, 'get_model', lambda: AlternativeResponses())
    warnings = []
    monkeypatch.setattr(app.gr, 'Warning', lambda *a, **kw: warnings.append(a))
    real_diagnose = app.diagnose_image
    def diagnose(*args, **kwargs):
        final = real_diagnose(*args, **kwargs)
        if kwargs.get('robust'):
            if changed == 'tensor': final[6]['tensor'][0][0] += 1
            elif changed == 'preview': final[2][0, 0] += 1
            elif changed == 'class': final = ('8', *final[1:])
            elif changed == 'quality': final[6]['quality']['component_count'] = 2
            elif changed == 'bbox': final[6]['source_bbox'][0] += 1
            else: final[6]['coordinates']['source_crop'][0] += 1
        return final
    monkeypatch.setattr(app, 'diagnose_image', diagnose)
    result = app.analyze_image(textured_scene())
    assert result[:3] == (None, None, None) and result[7]['visible'] is True
    assert result[6]['candidate_comparison']['reason'] == 'final_diagnostic_mismatch'
    assert warnings == [] and 'alternative_digits' not in result[6]


def test_build_config_and_import_are_local_and_explicit():
    demo = app.build_app()
    config = demo.config
    assert config['analytics_enabled'] is False
    assert any('Diagnóstico de imagen' in c['props'].get('value', '')
               for c in config['components'] if c['type'] == 'markdown')
    labels = [c['props'].get('label') for c in config['components']]
    assert {'x0', 'y0', 'x1', 'y1', 'Usar recorte manual'} <= set(labels)
    dependencies = {d['api_name']: d for d in config['dependencies']}
    assert len(dependencies['classify']['outputs']) == 4
    assert len(dependencies['diagnose_image']['outputs']) == 7
    assert 'retry_diagnosis' in dependencies
    automatic = dependencies['analyze_image']
    assert len(automatic['inputs']) == 1 and len(automatic['outputs']) == 17
    assert len(dependencies['resolve_image']['outputs']) == 9
    help_group = next(c for c in config['components'] if c['id'] == automatic['outputs'][7])
    assert help_group['props']['visible'] is False
    tree = ast.parse(Path(app.__file__).read_text())
    launches = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'launch']
    assert len(launches) == 1
    assert {k.arg: ast.literal_eval(k.value) for k in launches[0].keywords} == {
        'server_name': '127.0.0.1', 'share': False, 'inbrowser': False}
    assert not any(isinstance(n, ast.ImportFrom) and n.module == 'train' for n in ast.walk(tree))
    main_guard = tree.body[-1]
    assert isinstance(main_guard, ast.If)
    assert ast.unparse(main_guard.test) == "__name__ == '__main__'"
    assert launches[0] in list(ast.walk(main_guard))
    assert all(c['props'].get('label') != 'Ruta del archivo' for c in config['components'])
    demo.close()
