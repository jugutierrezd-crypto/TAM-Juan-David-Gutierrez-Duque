"""Extracción optativa de fotografías; no usa etiquetas ni predicciones."""
from dataclasses import dataclass
from numbers import Integral

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class PhotoResult:
    """Tensor sin normalizar, vista y evidencia de segmentación inspeccionable."""

    tensor: np.ndarray
    preview: np.ndarray
    stages: dict
    quality: dict


def _fit_and_center(crop):
    height, width = crop.shape
    scale = 20 / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = np.asarray(Image.fromarray(crop).resize(size, Image.Resampling.LANCZOS))
    yy, xx = np.indices(resized.shape)
    mass = float(resized.sum())
    left = int(np.clip(round(13.5 - (xx * resized).sum() / mass), 0, 28 - size[0]))
    top = int(np.clip(round(13.5 - (yy * resized).sum() / mass), 0, 28 - size[1]))
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[top:top + size[1], left:left + size[0]] = resized
    return canvas


def _select_ink(raw, *, manual, robust=False):
    """Hysteresis on illumination-corrected ink, independent of CLAHE."""
    height, width = raw.shape
    minimum_area = max(10, height * width * .00004)
    strength = float(np.percentile(raw, 99.8)) * .7
    # Textured dark ruling can exceed the legacy cap and join every stroke.
    # The opt-in path keeps a genuinely strong core instead of lowering floors.
    if robust:
        # Sparse texture peaks must exceed the ruling baseline as well. Bound
        # this contrast requirement by the peak for tightly cropped ink.
        strength = max(strength, min(float(np.percentile(raw, 99)) * 2, float(raw.max()) * .85))
    high = max(6, strength if robust else min(80, strength))
    seeds = (raw >= high).astype(np.uint8)
    count, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(seeds, 8)
    candidates = []
    margin = max(1, round(min(height, width) * .005))

    def admissible(x, y, w, h, area):
        if area < minimum_area or min(w, h) < 2:
            return False
        if manual:
            return True
        return (x > margin and y > margin and x + w < width - margin
                and y + h < height - margin and w < width * .65
                and h < height * .65 and max(w / h, h / w) < 20
                and area / (w * h) >= .07)

    for index in range(1, count):
        x, y, w, h, area = map(int, seed_stats[index])
        if not admissible(x, y, w, h, area):
            continue
        seed = seed_labels == index
        # The local seed strength sets the weaker growth floor. Grid/noise below
        # that floor cannot bridge a compact pencil stroke to the entire page.
        low = max(3, high * (.4 if robust else .8),
                  float(np.median(raw[seed])) * (.3 if robust else .55))
        eligible = (raw >= low).astype(np.uint8)
        sy, sx = np.unravel_index(np.where(seed, raw, 0).argmax(), raw.shape)
        cv2.floodFill(eligible, None, (int(sx), int(sy)), 2, flags=8)
        region = eligible == 2
        ys, xs = np.nonzero(region)
        x, y = int(xs.min()), int(ys.min())
        w, h, area = int(xs.max()) + 1 - x, int(ys.max()) + 1 - y, len(xs)
        if not admissible(x, y, w, h, area):
            continue
        # Different high-threshold cores can reconstruct the same stroke with
        # slightly different floors. They are one candidate, not ambiguity.
        distinct = []
        for score, previous in candidates:
            overlap = np.count_nonzero(region & previous)
            if overlap > .5 * min(np.count_nonzero(region), np.count_nonzero(previous)):
                region |= previous
            else:
                distinct.append((score, previous))
        ys, xs = np.nonzero(region)
        area = len(xs)
        box_area = (int(xs.max()) + 1 - int(xs.min())) * (int(ys.max()) + 1 - int(ys.min()))
        # Strong compact ink beats broad, weak shadows; no label or model input.
        score = float(np.square(raw[region].astype(float)).sum() * np.sqrt(area / box_area))
        candidates = distinct + [(score, region)]
    if not candidates:
        raise ValueError('No se pudo aislar un dígito; revisá las etapas o indicá un recorte manual.')
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[0][1].copy()
    ratio = candidates[1][0] / candidates[0][0] if len(candidates) > 1 else 0.0
    reconstructed = np.logical_or.reduce([region for _, region in candidates])
    if manual:
        selected = reconstructed.copy()
    return selected, ratio, len(candidates), seeds * 255, reconstructed.astype(np.uint8) * 255


def preprocess_photo(image, *, crop=None, remove_grid=True, use_clahe=True, robust=False):
    """Aísla un candidato, no certifica su clase ni la calidad de la fotografía.

    ``crop`` usa píxeles enteros (izquierda, arriba, derecha, abajo), con fin
    exclusivo, sobre el original corregido por EXIF. Las cajas de ``stages``
    usan la imagen de trabajo (recorte reducido a un máximo de 1200 píxeles).
    Ante ambigüedad se devuelve una advertencia y una vista para inspección.
    ``robust=True`` suaviza textura antes de estimar iluminación y conserva
    semillas fuertes sin el techo histórico; no certifica una predicción.
    """
    if image is None:
        raise ValueError('Seleccioná una imagen con un solo dígito.')
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    oriented = ImageOps.exif_transpose(image).convert('RGBA')
    original = Image.alpha_composite(Image.new('RGBA', oriented.size, 'white'), oriented).convert('RGB')
    source_width, source_height = original.size
    manual = crop is not None
    if manual:
        if (not isinstance(crop, (tuple, list, np.ndarray)) or len(crop) != 4
                or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in crop)):
            raise ValueError('El recorte debe contener cuatro coordenadas enteras en píxeles.')
        x0, y0, x1, y1 = map(int, crop)
        if not (0 <= x0 < x1 <= source_width and 0 <= y0 < y1 <= source_height):
            raise ValueError('El recorte debe estar ordenado y dentro de la imagen corregida por EXIF.')
        source_box = [x0, y0, x1, y1]
    else:
        source_box = [0, 0, source_width, source_height]
    working = original.crop(source_box)
    before_width, before_height = working.size
    working.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    gray = np.asarray(working.convert('L'))
    height, width = gray.shape
    if min(height, width) < 3 or int(gray.max()) - int(gray.min()) < 3:
        raise ValueError('No se distingue un dígito del fondo; probá otra foto o un recorte.')

    sigma = max(.8, min(source_width * width / before_width,
                         source_height * height / before_height) * .0015)
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigma) if robust else gray
    # Closing estimates paper illumination without eroding the retained strokes.
    background_size = max(31, int(min(height, width) * .06) | 1)
    background = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                           (background_size, background_size)),
                                 borderType=cv2.BORDER_CONSTANT, borderValue=255)
    illumination = np.clip(smoothed.astype(np.float32) * 255 / np.maximum(background, 1), 0, 255).astype(np.uint8)
    contrast = (cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(illumination)
                if use_clahe else illumination.copy())
    ink = 255 - contrast
    # Grayscale line evidence: subtract only the baseline strength of very long
    # lines. At crossings the darker pencil excess survives, unlike binary opening.
    line_width = max(3, round(source_width * width / before_width * .45))
    line_height = max(3, round(source_height * height / before_height * .45))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, line_width), np.uint8),
                                 borderType=cv2.BORDER_CONSTANT, borderValue=0)
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((line_height, 1), np.uint8),
                               borderType=cv2.BORDER_CONSTANT, borderValue=0)
    lines = np.maximum(horizontal, vertical)
    cleaned = cv2.subtract(ink, lines) if remove_grid else ink.copy()
    grid_removed = 255 - cleaned
    block = min(51, min(height, width) if min(height, width) % 2 else min(height, width) - 1)
    adaptive = cv2.adaptiveThreshold(grid_removed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block, 4)
    raw = 255 - illumination
    raw_horizontal = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((1, line_width), np.uint8),
                                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    raw_vertical = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((line_height, 1), np.uint8),
                                   borderType=cv2.BORDER_CONSTANT, borderValue=0)
    raw_cleaned = cv2.subtract(raw, np.maximum(raw_horizontal, raw_vertical)) if remove_grid else raw
    # In the fallback only, reject narrow ruling before growing softer stroke
    # edges. Shared geometry, ambiguity and model-stability guards stay intact.
    robust_ink = (cv2.morphologyEx(raw_cleaned, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                  if robust else raw_cleaned)
    selected, ratio, component_count, seeds, hysteresis = _select_ink(robust_ink, manual=manual, robust=robust)
    mask = selected.astype(np.uint8) * 255
    ys, xs = np.nonzero(selected)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    x0, y0, x1, y1 = bbox
    foreground = np.where(selected, cleaned, 0).astype(np.uint8)
    digit_crop = foreground[y0:y1, x0:x1]
    preview = _fit_and_center(digit_crop)
    touches_border = x0 == 0 or y0 == 0 or x1 == width or y1 == height
    removed_fraction = float(np.maximum(ink.astype(float) - cleaned, 0).sum() / max(1, ink.sum()))
    warnings = []
    if float(selected.sum()) / ((x1 - x0) * (y1 - y0)) < .2:
        warnings.append('El candidato es disperso; puede incluir cuadrícula o marcas aisladas. Revisá el recorte.')
    if ratio >= .35:
        warnings.append('Hay varios componentes plausibles; inspeccioná el candidato o usá un recorte manual.')
    if touches_border:
        warnings.append('El trazo toca el borde; ampliá el recorte para comprobar que esté completo.')
    if removed_fraction > .7:
        warnings.append('Se eliminó mucha señal de líneas; compará las etapas y probá sin eliminar cuadrícula.')
    if not remove_grid and np.count_nonzero(lines > 15) > gray.size * .02:
        warnings.append('Quedan líneas largas; la cuadrícula puede dominar el candidato.')
    quality = dict(status='uncertain' if warnings else 'candidate', warning=' '.join(warnings),
                   manual_crop=manual, component_count=component_count, ambiguity_ratio=float(ratio),
                   removed_ink_fraction=removed_fraction, active_fraction=float(selected.mean()),
                   scores_calibrated=False)
    stages = dict(original=np.asarray(original), gray=gray, illumination=illumination,
                  contrast=contrast, lines=lines, grid_removed=grid_removed,
                  adaptive_mask=adaptive, seeds=seeds, hysteresis_mask=hysteresis,
                  mask=mask, crop=digit_crop, final=preview.copy(), bbox=bbox,
                  coordinates=dict(source_size=[source_width, source_height], source_crop=source_box,
                                   working_size=[width, height],
                                   source_per_working_pixel=[before_width / width, before_height / height],
                                   exif_corrected=True))
    if robust:
        stages['smoothed'] = smoothed
        stages['robust_ink'] = robust_ink
        quality.update(fallback_used=True, method='smoothed_uncapped_hysteresis')
    return PhotoResult(preview.astype(np.float32).reshape(1, 784), preview, stages, quality)
