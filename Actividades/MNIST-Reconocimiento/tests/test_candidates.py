"""Candidate decisions are heuristics, tested without training or photo labels."""
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from candidates import compare_candidates


def ambiguous_scene(thin=False):
    image = Image.new('RGB', (400, 600), 'white')
    draw = ImageDraw.Draw(image)
    for x in (100, 260):
        draw.line((x, 250, x, 325), fill='black', width=7)
        if not thin or x == 100:
            draw.line((x - 25, 325, x + 25, 325), fill='black', width=7)
    return image


class Responses:
    def __init__(self, peaks):
        self.peaks = peaks
        self.calls = 0

    def __call__(self, tensor, training=False):
        assert tensor.shape == (1, 784) and tensor.dtype == np.float32
        assert 0 <= tensor.min() <= tensor.max() <= 255
        peak = self.peaks[min(self.calls, len(self.peaks) - 1)]
        self.calls += 1
        output = np.full((1, 10), (1 - peak) / 9, dtype=np.float32)
        output[0, 1] = peak
        return output


def test_robust_comparison_requires_stability_and_rejects_ties_and_grid():
    from test_photo import textured_scene
    winner = compare_candidates(textured_scene(), Responses([.99]), robust=True)
    assert winner['selected_crop'] is not None and winner['reason'] == 'clear_winner'
    assert winner['candidates'][0]['stable'] is True
    tied = compare_candidates(textured_scene(2), Responses([.99]), robust=True)
    assert tied['selected_crop'] is None and tied['reason'] == 'no_clear_winner'
    assert len(tied['candidates']) == 2
    unstable = compare_candidates(textured_scene(), Responses([.99, .4]), robust=True)
    assert unstable['selected_crop'] is None
    model = Responses([.99])
    grid = compare_candidates(textured_scene(0), model, robust=True)
    assert grid['selected_crop'] is None and model.calls == 0
    json.dumps(winner, allow_nan=False)


def test_single_robust_candidate_keeps_page_tensor_and_processing_context():
    from test_photo import textured_scene
    from photo import preprocess_photo
    image = textured_scene()
    page = preprocess_photo(image, robust=True)
    tensors = []
    class Capture(Responses):
        def __call__(self, tensor, training=False):
            tensors.append(tensor.copy())
            return super().__call__(tensor, training=training)
    result = compare_candidates(image, Capture([.99]), robust=True)
    assert result['evaluation_context'] == 'page'
    assert result['selected_crop'] == [0, 0, image.width, image.height]
    candidate, = result['candidates']
    assert candidate['source_bbox'] == page.stages['bbox']
    assert candidate['quality'] == page.quality
    np.testing.assert_array_equal(candidate['tensor'], page.tensor)
    np.testing.assert_array_equal(tensors[0], page.tensor)
    assert len(tensors) == 5 and candidate['eligible']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('failure', ['probability', 'class', 'margin'])
def test_single_page_context_still_requires_five_view_model_evidence(failure):
    from test_photo import textured_scene
    class Unstable(Responses):
        def __call__(self, tensor, training=False):
            output = super().__call__(tensor, training=training)
            if self.calls == 3:
                if failure == 'class':
                    output = np.roll(output, 1, axis=1)
                elif failure == 'margin':
                    output[:] = 0
                    output[0, :2] = [.45, .55]
            return output
    model = Unstable([.99, .99, .4 if failure == 'probability' else .99, .99, .99])
    result = compare_candidates(textured_scene(), model, robust=True)
    assert result['evaluation_context'] == 'page'
    assert result['selected_crop'] is None and result['reason'] == 'no_clear_winner'
    candidate, = result['candidates']
    assert not candidate['eligible'] and candidate['reason'] == 'model_evidence'
    assert model.calls == 5


@pytest.mark.parametrize('peak, expected', [(.49, None), (.5, 'tentative_winner'),
                                          (.741, 'tentative_winner'), (.799, 'tentative_winner'),
                                          (.8, 'clear_winner'), (.99, 'clear_winner')])
def test_tentative_policy_is_explicit_and_does_not_lower_strong_floor(peak, expected):
    from test_photo import textured_scene
    image = textured_scene()
    legacy = compare_candidates(image, Responses([peak]), robust=True)
    opted = compare_candidates(image, Responses([peak]), robust=True, allow_tentative=True)
    assert opted['reason'] == (expected or 'no_clear_winner')
    assert (opted['selected_crop'] is not None) == (expected is not None)
    assert opted['candidates'][0]['eligible'] == (peak >= .8)
    assert (legacy['selected_crop'] is not None) == (peak >= .8)
    if expected:
        assert opted['prediction_status'] == ('tentative' if peak < .8 else 'accepted')
        assert opted['min_probability'] == pytest.approx(peak)
    json.dumps(opted, allow_nan=False)


def test_tentative_local_winner_competes_with_all_ranked_rivals():
    clear = compare_candidates(ambiguous_scene(), Responses([.74] * 5 + [.3] * 5),
                               allow_tentative=True)
    assert clear['reason'] == 'tentative_winner' and clear['selected_crop'] is not None
    winner = next(c for c in clear['candidates'] if c['crop'] == clear['selected_crop'])
    assert not winner['eligible'] and winner['tentative_eligible']
    assert winner['tensor'] and winner['quality']['status'] == 'candidate'
    for rival in (.74, .7):
        tied = compare_candidates(ambiguous_scene(), Responses([.74] * 5 + [rival] * 5),
                                  allow_tentative=True)
        assert tied['selected_crop'] is None and tied['reason'] == 'no_clear_winner'
        assert len(tied['candidates']) == 2 and not any(c['eligible'] for c in tied['candidates'])


@pytest.mark.parametrize('failure', ['class', 'margin', 'geometry', 'model'])
def test_tentative_option_cannot_bypass_existing_guards(failure):
    from test_photo import textured_scene
    class Unsafe(Responses):
        def __call__(self, tensor, training=False):
            output = super().__call__(tensor, training=training)
            if failure == 'model':
                raise ValueError('modelo inválido')
            if failure == 'class' and self.calls == 3:
                output = np.roll(output, 1, axis=1)
            if failure == 'margin':
                output[:] = 0
                output[0, :2] = [.45, .55]
            return output
    if failure == 'geometry':
        image = Image.new('RGB', (400, 600), 'white')
        ImageDraw.Draw(image).line((205, 250, 205, 325), fill='black', width=7)
    else:
        image = textured_scene()
    if failure == 'model':
        with pytest.raises(ValueError, match='modelo inválido'):
            compare_candidates(image, Unsafe([.74]), robust=True, allow_tentative=True)
    else:
        result = compare_candidates(image, Unsafe([.74]), robust=True, allow_tentative=True)
        assert result['selected_crop'] is None
        assert all(not c.get('tentative_eligible') for c in result['candidates'])


def test_one_plausible_stable_candidate_is_selected_without_mutating_image():
    image = ambiguous_scene()
    before = np.asarray(image).copy()
    model = Responses([.99] * 5 + [.4] * 5)
    result = compare_candidates(image, model)
    assert result['selected_crop'] is not None and len(result['candidates']) == 2
    winner = next(c for c in result['candidates'] if c['eligible'])
    assert result['selected_crop'] == winner['crop']
    assert winner['digit'] == 1 and winner['stable'] is True
    assert winner['min_probability'] > .98 and winner['shape_score'] > 0
    assert result['scores_calibrated'] is False
    assert model.calls == 10
    np.testing.assert_array_equal(np.asarray(image), before)
    json.dumps(result, allow_nan=False)


def test_tied_digit_candidates_still_require_help():
    result = compare_candidates(ambiguous_scene(), Responses([.99]))
    assert result['selected_crop'] is None and result['reason'] == 'no_clear_winner'
    assert len(result['candidates']) == 2 and all(c['eligible'] for c in result['candidates'])


def test_high_softmax_cannot_overrule_implausible_shape():
    model = Responses([.99])
    result = compare_candidates(ambiguous_scene(thin=True), model)
    rejected = [c for c in result['candidates'] if c['reason'] == 'geometry']
    assert rejected and all(not c['eligible'] for c in rejected)
    assert result['selected_crop'] is not None and model.calls == 5


def test_unstable_predictions_and_blank_images_never_force_a_choice():
    result = compare_candidates(ambiguous_scene(), Responses([.99, .4, .99, .99, .99] * 2))
    assert result['selected_crop'] is None
    assert all(not c['eligible'] for c in result['candidates'])
    model = Responses([.99])
    blank = compare_candidates(Image.new('RGB', (400, 600), 'white'), model)
    assert blank['selected_crop'] is None and blank['reason'] == 'no_candidates'
    assert model.calls == 0


def test_changed_class_under_shift_abstains_even_with_high_softmax():
    class ChangingClass(Responses):
        def __call__(self, tensor, training=False):
            output = super().__call__(tensor, training=training)
            return np.roll(output, self.calls % 2, axis=1)
    result = compare_candidates(ambiguous_scene(), ChangingClass([.99]))
    assert result['selected_crop'] is None
    assert all(not c['stable'] and not c['eligible'] for c in result['candidates'])


def test_more_than_eight_regions_abstains_before_inference():
    image = Image.new('RGB', (600, 600), 'white')
    draw = ImageDraw.Draw(image)
    for y in (100, 280, 460):
        for x in (100, 280, 460):
            draw.line((x, y, x, y + 60, x + 30, y + 60), fill='black', width=7)
    model = Responses([.99])
    result = compare_candidates(image, model)
    assert result['selected_crop'] is None and result['reason'] == 'too_many_candidates'
    assert model.calls == 0
    opted = compare_candidates(image, model, allow_tentative=True)
    assert opted['selected_crop'] is None and opted['reason'] == 'too_many_candidates'
    assert model.calls == 0  # genuine plausible rivals cannot be hidden by the opt-in gate
    alternatives = compare_candidates(image, model, allow_alternatives=True)
    assert alternatives['selected_crop'] is None and alternatives['reason'] == 'too_many_candidates'
    assert model.calls == 0


def test_small_ranking_gap_and_low_class_margin_abstain():
    near_tie = compare_candidates(ambiguous_scene(), Responses([.99] * 5 + [.85] * 5))
    assert all(c['eligible'] for c in near_tie['candidates'])
    assert near_tie['selected_crop'] is None and near_tie['reason'] == 'no_clear_winner'
    class LowMargin(Responses):
        def __call__(self, tensor, training=False):
            super().__call__(tensor, training=training)
            output = np.zeros((1, 10), dtype=np.float32)
            output[0, :2] = [.45, .55]
            return output
    result = compare_candidates(ambiguous_scene(), LowMargin([.99]))
    assert result['selected_crop'] is None
    assert all(c['min_margin'] < .2 and not c['eligible'] for c in result['candidates'])


@pytest.mark.parametrize('alternatives', [False, True])
@pytest.mark.parametrize('problem', ['signal_loss', 'regions', 'local_uncertain', 'fragmented', 'border', 'shortened'])
def test_extraction_safety_guards_abstain_before_model(monkeypatch, problem, alternatives):
    # Controlled extractor evidence isolates each rejection rule; calls still go
    # through the public comparison API and never inspect its private state.
    import candidates
    real_preprocess = candidates.preprocess_photo
    def preprocess(image, **kwargs):
        result = real_preprocess(image, **kwargs)
        if kwargs.get('crop') is None:
            if problem == 'signal_loss':
                result.quality['removed_ink_fraction'] = .71
            if problem == 'regions':
                result.stages['hysteresis_mask'][:] = 0
            if problem == 'border':
                result.stages['hysteresis_mask'][:] = 0
                result.stages['hysteresis_mask'][:80, :7] = 255
                result.stages['hysteresis_mask'][73:80, :50] = 255
                result.quality['component_count'] = 1
        else:
            if problem == 'local_uncertain':
                result.quality.update(status='uncertain', warning='ruido')
            if problem == 'fragmented':
                result.quality['component_count'] = 2
            if problem == 'shortened':
                result.stages['bbox'][3] = result.stages['bbox'][1] + 3
        return result
    monkeypatch.setattr(candidates, 'preprocess_photo', preprocess)
    model = Responses([.99])
    result = compare_candidates(ambiguous_scene(), model, allow_alternatives=alternatives)
    assert result['selected_crop'] is None and model.calls == 0
    if problem == 'signal_loss':
        assert result['reason'] == 'excessive_signal_loss'
    elif problem == 'regions':
        assert result['reason'] == 'unrecoverable_regions'
    else:
        expected = {'local_uncertain': 'local_uncertain', 'fragmented': 'local_uncertain',
                    'border': 'geometry', 'shortened': 'incomplete_recovery'}[problem]
        assert all(c['reason'] == expected and not c['eligible'] for c in result['candidates'])


def notebook_marks_scene():
    image = Image.new('RGB', (600, 600), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 5, 599), fill=(20,) * 3)  # outside the paper
    for y in (100, 260, 420):
        for x in (80, 150, 220):
            draw.line((x, y, x, y + 50), fill='black', width=3)
    draw.line((420, 240, 420, 320, 470, 320), fill='black', width=9)
    return image


def test_opted_comparison_bounds_plausible_regions_not_invalid_notebook_marks():
    image = notebook_marks_scene()
    legacy_model, opted_model = Responses([.99]), Responses([.99])
    legacy = compare_candidates(image, legacy_model)
    assert legacy['reason'] == 'too_many_candidates' and legacy_model.calls == 0
    opted = compare_candidates(image, opted_model, allow_tentative=True)
    assert opted['reason'] == 'clear_winner' and opted['selected_crop'] is not None
    assert opted_model.calls == 5
    assert len([c for c in opted['candidates'] if c['reason'] == 'geometry']) == 9
    assert len([c for c in opted['candidates'] if 'ranking_score' in c]) == 1


def test_robust_local_context_can_reuse_original_regions_without_global_seeds(monkeypatch):
    import candidates
    from photo import preprocess_photo
    real = candidates.preprocess_photo
    def preprocess(image, **kwargs):
        if kwargs.get('robust') and kwargs.get('crop') is None:
            raise ValueError('No se pudo aislar un dígito; semillas externas dominantes.')
        return real(image, **kwargs)
    monkeypatch.setattr(candidates, 'preprocess_photo', preprocess)
    image, model = notebook_marks_scene(), Responses([.74])
    result = compare_candidates(image, model, robust=True, allow_tentative=True)
    assert result['reason'] == 'tentative_winner' and model.calls == 5
    assert result['proposal_context'] == 'original_regions'
    winner = next(c for c in result['candidates'] if c['crop'] == result['selected_crop'])
    local = preprocess_photo(image, crop=winner['crop'], robust=True)
    np.testing.assert_array_equal(winner['tensor'], local.tensor)
    assert winner['quality'] == local.quality
    assert winner['coordinates'] == local.stages['coordinates']
    assert winner['quality']['fallback_used'] is True


class AlternativeResponses:
    def __init__(self, digits=(4, 1, 4, 1, 9), peak=.4, problem=None):
        self.digits, self.peak, self.problem, self.calls = digits, peak, problem, 0

    def __call__(self, tensor, training=False):
        digit = self.digits[self.calls % len(self.digits)]
        self.calls += 1
        output = np.full((1, 10), (1 - self.peak) / 9, dtype=np.float32)
        output[0, digit] = self.peak
        if self.problem == 'margin':
            output[:] = .25 / 8
            output[0, digit], output[0, (digit + 1) % 10] = .4, .35
        elif self.problem == 'uniform':
            output[:] = .1
        elif self.problem == 'invalid':
            output[:] = 1
        return output


@pytest.mark.parametrize('digits,expected', [((4, 1, 4, 1, 9), [4, 1, 9]),
                                           ((9, 4, 4, 9, 1), [9, 4, 1]),
                                           ((7, 2, 7, 2, 2), [2, 7])])
def test_alternatives_are_opt_in_unique_and_frequency_ordered(digits, expected):
    from test_photo import textured_scene
    image = textured_scene()
    default = compare_candidates(image, AlternativeResponses(digits), robust=True)
    assert default['selected_crop'] is None and default['reason'] == 'no_clear_winner'
    result = compare_candidates(image, AlternativeResponses(digits), robust=True, allow_alternatives=True)
    assert result['prediction_status'] == 'alternatives'
    assert result['alternative_digits'] == expected
    assert result['selected_crop'] == [0, 0, image.width, image.height]
    candidate, = result['candidates']
    assert not candidate['eligible'] and not candidate['stable']
    assert 'digit' not in candidate and candidate['base_digit'] == digits[0]
    assert [v['digit'] for v in candidate['views']] == list(digits)
    assert candidate['raw_probabilities'][str(digits[0])] == pytest.approx(.4)
    assert [v['shift'] for v in candidate['views']] == [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]
    assert all(v['probability'] == pytest.approx(.4) and v['margin'] >= .1 for v in candidate['views'])
    assert candidate['tensor'] and candidate['quality']['status'] == 'candidate'
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('problem', ['weak', 'margin', 'uniform', 'invalid', 'four_classes', 'single_class', 'rivals'])
def test_alternatives_never_bypass_evidence_or_spatial_ambiguity(problem):
    from test_photo import textured_scene
    image = textured_scene(2 if problem == 'rivals' else 1)
    digits = (0, 1, 2, 3, 0) if problem == 'four_classes' else (4,) if problem == 'single_class' else (4, 1, 4, 1, 9)
    model = AlternativeResponses(digits, peak=.29 if problem == 'weak' else .4, problem=problem)
    if problem == 'invalid':
        with pytest.raises(ValueError, match='Softmax inválida'):
            compare_candidates(image, model, robust=True, allow_alternatives=True)
    else:
        result = compare_candidates(image, model, robust=True, allow_alternatives=True)
        assert result['selected_crop'] is None and result['reason'] == 'no_clear_winner'
        assert 'alternative_digits' not in result


@pytest.mark.parametrize('peak,accepted', [(.299, False), (.3, True)])
def test_alternatives_probability_floor_applies_to_every_view(peak, accepted):
    from test_photo import textured_scene
    class OneWeakView(AlternativeResponses):
        def __call__(self, *args, **kwargs):
            self.peak = peak if self.calls == 2 else .7
            return super().__call__(*args, **kwargs)
    result = compare_candidates(textured_scene(), OneWeakView(), robust=True, allow_alternatives=True)
    assert (result['selected_crop'] is not None) == accepted
    assert result['candidates'][0]['min_probability'] == pytest.approx(peak)


@pytest.mark.parametrize('margin,accepted', [(.099, False), (.101, True)])
def test_alternatives_margin_floor_applies_to_every_view(margin, accepted):
    from test_photo import textured_scene
    class OneCloseView(AlternativeResponses):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            if self.calls == 3:
                digit = self.digits[2]
                result[:] = (1 - .3 - (.3 + margin)) / 8
                result[0, digit], result[0, (digit + 1) % 10] = .3 + margin, .3
            return result
    result = compare_candidates(textured_scene(), OneCloseView(), robust=True, allow_alternatives=True)
    assert (result['selected_crop'] is not None) == accepted
    assert result['candidates'][0]['min_margin'] == pytest.approx(margin)


def test_exif_and_downscaled_boxes_preserve_strokes_and_margin():
    image = ambiguous_scene().resize((1600, 2400), Image.Resampling.NEAREST)
    image.getexif()[274] = 3  # 180 degrees, still upright enough for geometry
    result = compare_candidates(image, Responses([.99] * 5 + [.4] * 5))
    assert result['selected_crop'] is not None
    for candidate in result['candidates']:
        x0, y0, x1, y1 = candidate['crop']
        assert 0 <= x0 < x1 <= 1600 and 0 <= y0 < y1 <= 2400
        assert x1 - x0 > 200 and y1 - y0 > 300
    json.dumps(result, allow_nan=False)
