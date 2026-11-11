"""Adaptive extraction rules use generated scenes, never the three evaluation labels."""
import numpy as np
import pytest
from PIL import Image, ImageDraw
from photo import preprocess_photo


def scene(digit, grid=True, shadow=True):
    y, x = np.indices((600, 400))
    paper = np.clip(235 - (65 * y / 600 if shadow else np.zeros_like(y)), 0, 255).astype('uint8')
    image = Image.fromarray(paper).convert('RGB')
    draw = ImageDraw.Draw(image)
    if grid:
        for pos in range(15, 400, 40):
            draw.line((pos, 0, pos, 600), fill=(160,) * 3, width=2)
        for pos in range(15, 600, 40):
            draw.line((0, pos, 400, pos), fill=(160,) * 3, width=2)
    if digit == '1':
        draw.line((205, 250, 205, 325), fill=(65,) * 3, width=7)
        draw.line((180, 325, 230, 325), fill=(65,) * 3, width=7)
    else:
        draw.line((180, 250, 180, 290, 230, 290), fill=(65,) * 3, width=7)
        draw.line((230, 245, 230, 325), fill=(65,) * 3, width=7)
    draw.ellipse((70, 100, 72, 102), fill='black')
    return image


@pytest.mark.parametrize('digit', ['1', '4'])
def test_grid_preserves_digit_strokes(digit):
    image = scene(digit)
    result = preprocess_photo(image)
    assert result.tensor.shape == (1, 784) and result.tensor.dtype == np.float32
    np.testing.assert_array_equal(result.tensor.reshape(28, 28), result.preview)
    assert 100 <= result.preview.max() <= 255
    x0, y0, x1, y1 = result.stages['bbox']
    assert 170 <= x0 <= 185 and 235 <= y0 <= 255
    assert x1 >= 230 and 325 <= y1 <= 340
    # Both the horizontal foot and the long vertical of 1/4 survive grid removal.
    mask = result.stages['mask']
    assert mask[325 if digit == '1' else 290, 190:225].mean() > 200
    assert mask[255:315, 205 if digit == '1' else 230].mean() > 200
    assert not mask[100:103, 70:73].any()
    yy, xx = np.indices((28, 28))
    assert abs((xx * result.preview).sum() / result.preview.sum() - 13.5) <= 0.7
    assert abs((yy * result.preview).sum() / result.preview.sum() - 13.5) <= 0.7


def test_ambiguous_scene_warns_and_manual_crop_is_validated():
    image = scene('1', grid=False, shadow=False)
    image.paste(image.crop((170, 240, 245, 340)), (55, 240))
    assert preprocess_photo(image).quality['status'] == 'uncertain'
    cropped = preprocess_photo(image, crop=(170, 240, 245, 340))
    assert cropped.quality['manual_crop'] is True
    for box in [(-1, 0, 20, 20), (20, 20, 10, 40), (0, 0, 999, 999),
                (0, 0, 20.5, 40), (False, 0, 20, 40)]:
        with pytest.raises(ValueError, match='recorte'):
            preprocess_photo(image, crop=box)


def test_blank_and_grid_only_rejected():
    grid = Image.new('RGB', (400, 600), 'white')
    draw = ImageDraw.Draw(grid)
    for x in range(15, 400, 40):
        draw.line((x, 0, x, 600), fill='gray', width=2)
    for y in range(15, 600, 40):
        draw.line((0, y, 400, y), fill='gray', width=2)
    for image in [Image.new('RGB', (400, 600), 'white'), grid]:
        with pytest.raises(ValueError, match='dígito'):
            preprocess_photo(image)


def test_low_contrast_and_ablation_contracts():
    image = scene('1', grid=False)
    values = np.asarray(image).astype(float)
    faint = Image.fromarray((225 + values * .09).astype(np.uint8))
    for options in [{}, {'use_clahe': False}, {'remove_grid': False}]:
        result = preprocess_photo(faint, **options)
        assert result.preview.sum() > 0
        assert np.isfinite(result.tensor).all()
        assert 0 <= result.tensor.min() <= result.tensor.max() <= 255
        assert {'gray', 'illumination', 'contrast', 'lines', 'grid_removed',
                'adaptive_mask', 'mask', 'crop', 'final', 'bbox'} <= result.stages.keys()
        assert result.quality['scores_calibrated'] is False


def test_manual_crop_keeps_border_strokes_and_exif_coordinates():
    image = scene('1', grid=False, shadow=False)
    result = preprocess_photo(image, crop=(202, 250, 231, 329))
    assert result.stages['mask'][:, 0].any()
    assert result.quality['status'] == 'uncertain'
    assert 'borde' in result.quality['warning']
    image.getexif()[274] = 6
    oriented = preprocess_photo(image, crop=(240, 170, 340, 245))
    assert oriented.stages['coordinates']['source_size'] == [600, 400]
    with pytest.raises(ValueError, match='recorte'):
        preprocess_photo(image, crop=(0, 0, 400, 600))


@pytest.mark.parametrize('digit', ['1', '4'])
def test_compact_ink_wins_over_dark_outside_paper_and_faint_grid(digit):
    image = scene(digit, grid=False)
    draw = ImageDraw.Draw(image)
    for x in range(15, 400, 40):
        draw.line((x, 0, x, 600), fill=(185,) * 3, width=2)
    for y in range(15, 600, 40):
        draw.line((0, y, 400, y), fill=(185,) * 3, width=2)
    # Redraw ink after the faint grid, matching pencil darker than ruling.
    original = scene(digit, grid=False)
    image.paste(original.crop((170, 240, 245, 340)), (170, 240))
    draw = ImageDraw.Draw(image)
    draw.polygon([(0, 440), (140, 455), (95, 600), (0, 600)], fill=(12,) * 3)
    draw.line((0, 470, 120, 480), fill=(95,) * 3, width=8)
    before = np.asarray(image).copy()
    result = preprocess_photo(image)
    np.testing.assert_array_equal(np.asarray(image), before)
    x0, y0, x1, y1 = result.stages['bbox']
    assert 170 <= x0 <= 185 and 235 <= y0 <= 255
    assert 230 <= x1 <= 245 and 325 <= y1 <= 340
    assert not result.stages['mask'][440:].any()
    assert result.stages['mask'][325 if digit == '1' else 290, 190:225].mean() > 200
    np.testing.assert_array_equal(result.tensor.reshape(28, 28), result.preview)


def textured_scene(digits=1):
    """Dark connected ruling plus reproducible paper grain, not a photo fixture."""
    rng = np.random.default_rng(23)
    y, x = np.indices((600, 500))
    paper = 230 - 35 * y / 600 + rng.normal(0, 7, y.shape)
    image = Image.fromarray(np.clip(paper, 0, 255).astype('uint8')).convert('RGB')
    draw = ImageDraw.Draw(image)
    for pos in range(15, 600, 25):
        draw.line((0, pos, 500, pos + 8), fill=(105,) * 3, width=2)
        draw.line((pos, 0, pos + 8, 600), fill=(105,) * 3, width=2)
    for offset in (range(digits)):
        x = 140 + offset * 180
        draw.line((x, 250, x, 330, x + 50, 330), fill=(15,) * 3, width=9)
    return image


def test_robust_texture_fallback_retains_complete_stroke_and_contract():
    image = textured_scene()
    with pytest.raises(ValueError, match='aislar'):
        preprocess_photo(image)
    before = np.asarray(image).copy()
    result = preprocess_photo(image, robust=True)
    assert result.quality['status'] == 'candidate'
    assert result.quality['fallback_used'] is True
    assert result.quality['method'] == 'smoothed_uncapped_hysteresis'
    x0, y0, x1, y1 = result.stages['bbox']
    assert 132 <= x0 <= 140 and 245 <= y0 <= 252
    assert 190 <= x1 <= 198 and 330 <= y1 <= 338
    assert result.stages['mask'][260:325, 140].mean() > 200
    assert result.stages['mask'][330, 145:190].mean() > 200
    assert result.preview.dtype == np.uint8 and result.tensor.dtype == np.float32
    assert result.tensor.shape == (1, 784) and 0 <= result.tensor.min() <= result.tensor.max() <= 255
    np.testing.assert_array_equal(result.preview, result.stages['final'])
    np.testing.assert_array_equal(result.preview, result.tensor.reshape(28, 28))
    np.testing.assert_array_equal(np.asarray(image), before)
    assert 'smoothed' in result.stages


@pytest.mark.parametrize('image', [Image.new('RGB', (500, 600), 'white'), textured_scene(0)])
def test_robust_blank_and_textured_grid_only_rejected(image):
    with pytest.raises(ValueError, match='dígito'):
        preprocess_photo(image, robust=True)


def test_robust_is_opt_in_and_preserves_source_coordinates():
    image = scene('4')
    default = preprocess_photo(image)
    explicit = preprocess_photo(image, robust=False)
    assert default.quality == explicit.quality
    for key in default.stages:
        if isinstance(default.stages[key], np.ndarray):
            np.testing.assert_array_equal(default.stages[key], explicit.stages[key])
        else:
            assert default.stages[key] == explicit.stages[key]
    image = textured_scene()
    image.getexif()[274] = 3
    result = preprocess_photo(image, robust=True, crop=(280, 250, 380, 370))
    assert result.stages['coordinates']['source_crop'] == [280, 250, 380, 370]
    assert result.stages['coordinates']['source_size'] == [500, 600]
    assert result.quality['manual_crop'] is True


def test_working_resolution_is_bounded_and_array_input_supported():
    image = scene('4').resize((1600, 2400))
    result = preprocess_photo(np.asarray(image))
    assert max(result.stages['gray'].shape) <= 1200
    assert result.stages['coordinates']['source_per_working_pixel'] == [2, 2]
