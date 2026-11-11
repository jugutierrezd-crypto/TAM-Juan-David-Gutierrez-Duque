"""Public-interface probes confirmed by the independent feature verifier."""
import numpy as np
from PIL import Image, ImageDraw
import app
from photo import preprocess_photo


def sample():
    image = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(image)
    for p in range(16, 512, 32):
        draw.line((p, 0, p, 511), fill=(180,) * 3, width=2)
        draw.line((0, p, 511, p), fill=(180,) * 3, width=2)
    draw.line((224, 208, 224, 272, 288, 272), fill=(40,) * 3, width=7)
    draw.line((288, 208, 288, 320), fill=(40,) * 3, width=7)
    draw.ellipse((70, 70, 72, 72), fill='black')
    return image


def test_crossing_grid_noise_transparency_and_tensor():
    image = sample()
    result = preprocess_photo(image)
    assert result.stages['mask'][272, 235:280].min() == 255
    assert result.stages['mask'][220:310, 288].min() == 255
    assert not result.stages['mask'][70:73, 70:73].any()
    assert result.tensor.dtype == np.float32 and result.tensor.shape == (1, 784)
    np.testing.assert_array_equal(result.tensor.reshape(28, 28), result.preview)
    rgba = image.convert('RGBA')
    ImageDraw.Draw(rgba).rectangle((0, 0, 10, 10), fill=(0, 0, 0, 0))
    white = image.copy()
    ImageDraw.Draw(white).rectangle((0, 0, 10, 10), fill='white')
    np.testing.assert_array_equal(preprocess_photo(rgba).tensor, preprocess_photo(white).tensor)


def test_exif_crop_source_coordinates_and_invalid_crop_clear(monkeypatch):
    monkeypatch.setattr(app, 'get_model', lambda: lambda t, training=False: np.eye(10)[[4]])
    image = sample()
    image.getexif()[274] = 6
    result = app.diagnose_image(image, manual=True, x0=180, y0=200, x1=320, y1=310)
    assert result[6]['coordinates']['source_crop'] == [180, 200, 320, 310]
    assert result[5].size == (512, 512)
    for crop in [(-1, 0, 100, 100), (0, 0, 512, 513), (0, 0, 0, 10),
                 (False, 0, 100, 100), (0.5, 0, 100, 100)]:
        failed = app.diagnose_image(image, manual=True,
                                   **dict(zip(('x0', 'y0', 'x1', 'y1'), crop)))
        assert failed[:3] == (None, None, None)
        assert failed[4:] == ([], None, {})
        assert 'ADVERTENCIA' in failed[3]
