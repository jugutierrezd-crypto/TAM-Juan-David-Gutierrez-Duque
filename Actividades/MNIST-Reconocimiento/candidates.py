"""Conservative comparison of photo regions; scores are not calibrated accuracy."""
import math

import cv2
import numpy as np

from diagnose import infer
from photo import preprocess_photo


def _plausible_region(region, width, height):
    """Shared cheap gate, before bounded inference and before local extraction."""
    x, y, w, h, area = region
    return (.2 <= area / (w * h) <= .85 and .08 <= w / h <= 2.5
            and min(w, h) >= 3 and area >= 10
            and x > 0 and y > 0 and x + w < width and y + h < height)


def compare_candidates(image, model, *, remove_grid=True, use_clahe=True, robust=False,
                       allow_tentative=False, allow_alternatives=False):
    """Return JSON evidence and a source/EXIF crop only for a clear winner.

    Reuse the photographic extractor and its unnormalized tensor contract. Model
    errors propagate so the caller can report them rather than invent a choice.
    ``allow_tentative`` separately permits stable evidence below the strong
    probability floor; callers must visibly label it as uncertain. In this
    opt-in route the eight-region budget counts geometry-plausible regions;
    failed global robust seeds may use original proposals for robust local
    extraction, without reusing original tensors or lowering evidence floors.
    ``allow_alternatives`` separately permits 2–3 classes for exactly one locally
    valid region: all five top probabilities >= .3 and margins >= .1. It never
    certifies a class or resolves spatial rivalry, and preserves each view.
    """
    recovery = allow_tentative or allow_alternatives
    result = dict(selected_crop=None, candidates=[], reason='no_candidates',
                  scores_calibrated=False)
    try:
        page = preprocess_photo(image, remove_grid=remove_grid, use_clahe=use_clahe, robust=robust)
    except ValueError as error:
        result['detail'] = str(error)
        if not (robust and recovery and 'No se pudo aislar un dígito' in str(error)):
            return result
        # Outside-paper peaks may prevent global robust seeds. Reuse the
        # original extractor only as a proposer, then evaluate robust LOCAL
        # contexts with the same gates and budget; never reuse its tensor.
        try:
            page = preprocess_photo(image, remove_grid=remove_grid, use_clahe=use_clahe)
        except ValueError:
            return result
        result['proposal_context'] = 'original_regions'
    if page.quality['removed_ink_fraction'] > .7:
        result['reason'] = 'excessive_signal_loss'
        return result
    mask = page.stages['hysteresis_mask']
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    height, width = mask.shape
    regions = stats[1:].tolist()
    bounded_count = (sum(_plausible_region(region, width, height) for region in regions)
                     if recovery else count - 1)
    if bounded_count > 8:
        result['reason'] = 'too_many_candidates'
        return result
    if count <= 1 or (page.quality['component_count'] > 1 and count < 3):
        result['reason'] = 'unrecoverable_regions'
        return result
    coordinates = page.stages['coordinates']
    source_width, source_height = coordinates['source_size']
    ox, oy, _, _ = coordinates['source_crop']
    sx, sy = coordinates['source_per_working_pixel']
    page_context = (robust and 'proposal_context' not in result
                    and count == 2 and page.quality['component_count'] == 1
                    and page.quality['status'] == 'candidate')
    if page_context:
        result['evaluation_context'] = 'page'
    for x, y, w, h, area in regions:
        left, top = math.floor(ox + x * sx), math.floor(oy + y * sy)
        right, bottom = math.ceil(ox + (x + w) * sx), math.ceil(oy + (y + h) * sy)
        px, py = max(3, math.ceil((right - left) * .1)), max(3, math.ceil((bottom - top) * .1))
        crop = [max(0, left - px), max(0, top - py),
                min(source_width, right + px), min(source_height, bottom + py)]
        density = area / (w * h)
        shape_score = float(max(0, 1 - abs(density - .35) / .5))
        candidate = dict(crop=crop, eligible=False, reason='geometry', digit=None,
                         stable=False, min_probability=None, shape_score=shape_score)
        if page_context:
            # Keep the illumination/segmentation context that recovered this ink.
            # The crop describes processing context, not the selected ink box.
            candidate.update(crop=list(coordinates['source_crop']),
                             source_bbox=[left, top, right, bottom],
                             tensor=page.tensor.tolist(), quality=page.quality.copy())
        result['candidates'].append(candidate)
        if not _plausible_region((x, y, w, h, area), width, height):
            continue
        if page_context:
            local = page
        else:
            try:
                local = preprocess_photo(image, crop=crop, remove_grid=remove_grid, use_clahe=use_clahe, robust=robust)
            except ValueError as error:
                candidate.update(reason='local_rejected', detail=str(error))
                continue
        if local.quality['status'] != 'candidate' or local.quality['component_count'] != 1:
            candidate.update(reason='local_uncertain', detail=local.quality['warning'])
            continue
        # A new local illumination estimate must not silently shorten the stroke.
        lx0, ly0, lx1, ly1 = local.stages['bbox']
        lsx, lsy = local.stages['coordinates']['source_per_working_pixel']
        if ((lx1 - lx0) * lsx < .8 * (right - left)
                or (ly1 - ly0) * lsy < .8 * (bottom - top)):
            candidate['reason'] = 'incomplete_recovery'
            continue
        if recovery:
            context = local.stages['coordinates']
            cx, cy, _, _ = context['source_crop']
            csx, csy = context['source_per_working_pixel']
            candidate.update(tensor=local.tensor.tolist(), quality=local.quality.copy(),
                             coordinates=context.copy(),
                             source_bbox=[math.floor(cx + lx0 * csx), math.floor(cy + ly0 * csy),
                                          math.ceil(cx + lx1 * csx), math.ceil(cy + ly1 * csy)])
        predictions, probabilities, margins, views = [], [], [], []
        for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            # Translate on black without wrapping; no normalization or training.
            shifted = cv2.warpAffine(local.preview, np.float32([[1, 0, dx], [0, 1, dy]]),
                                     (28, 28), borderValue=0)
            digit, scores = infer(model, shifted.astype(np.float32).reshape(1, 784))
            ordered = sorted(scores.values(), reverse=True)
            predictions.append(digit)
            probabilities.append(ordered[0])
            margins.append(ordered[0] - ordered[1])
            if allow_alternatives:
                views.append(dict(shift=[dx, dy], digit=digit, probability=ordered[0],
                                  margin=margins[-1], probabilities=scores))
        if allow_alternatives:
            candidate.update(base_digit=predictions[0], raw_probabilities=views[0]['probabilities'], views=views)
        stable = len(set(predictions)) == 1
        minimum, margin = min(probabilities), min(margins)
        eligible = stable and minimum >= .8 and margin >= .2
        ranking = .3 * shape_score + .7 * (minimum + margin) / 2
        candidate.update(digit=predictions[0], stable=stable, min_probability=minimum,
                         min_margin=margin, ranking_score=float(ranking), eligible=eligible,
                         reason='eligible' if eligible else 'model_evidence')
        if allow_tentative:
            candidate['tentative_eligible'] = (not eligible and stable and minimum >= .5 and margin >= .2)
    ranked = sorted((c for c in result['candidates'] if 'ranking_score' in c),
                    key=lambda c: c['ranking_score'], reverse=True)
    result['reason'] = 'no_clear_winner'
    if (ranked and (len(ranked) == 1
                    or ranked[0]['ranking_score'] - ranked[1]['ranking_score'] >= .15)):
        winner = ranked[0]
        # Rank against every evaluated rival, including unstable or weak ones.
        if winner['eligible']:
            result.update(selected_crop=winner['crop'], reason='clear_winner')
        elif allow_tentative and winner['tentative_eligible']:
            result.update(selected_crop=winner['crop'], reason='tentative_winner')
        if allow_tentative and result['selected_crop'] is not None:
            result.update(prediction_status='accepted' if winner['eligible'] else 'tentative',
                          min_probability=winner['min_probability'])
    # Alternatives are class uncertainty in ONE locally consistent region, not
    # a weaker way to choose between spatial rivals (regardless of rank gap).
    if allow_alternatives and result['selected_crop'] is None and len(ranked) == 1:
        candidate = ranked[0]
        classes = list(dict.fromkeys(v['digit'] for v in candidate['views']))
        if (2 <= len(classes) <= 3 and candidate['min_probability'] >= .3
                and candidate['min_margin'] >= .1):
            frequencies = {digit: sum(v['digit'] == digit for v in candidate['views']) for digit in classes}
            alternatives = sorted(classes, key=lambda digit: (-frequencies[digit],
                                  digit != candidate['base_digit'], classes.index(digit)))
            candidate.pop('digit')  # the unshifted class is evidence, never a winner
            candidate.update(alternative_digits=alternatives, reason='alternatives')
            result.update(selected_crop=candidate['crop'], reason='uncertain_alternatives',
                          prediction_status='alternatives', alternative_digits=alternatives,
                          min_probability=candidate['min_probability'])
    return result
