"""Pruebas sin descargar MNIST ni entrenar; las imágenes son sintéticas locales."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from predict import preprocess_image, predict_digit, load_model


def digit_image(inverted=False):
    image = Image.new('RGB', (100, 120), 'white' if inverted else 'black')
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 20, 55, 95), fill='black' if inverted else 'white')
    return image


def test_tensor_contract():
    tensor, preview = preprocess_image(digit_image())
    assert tensor.shape == (1, 784)
    assert tensor.dtype == np.float32
    assert preview.shape == (28, 28)
    assert tensor.min() == 0
    assert tensor.max() == 255  # Detecta una división accidental por 255.
    np.testing.assert_array_equal(tensor.reshape(28, 28), preview)


def test_polarity_and_center():
    tensor, preview = preprocess_image(digit_image())
    other, _ = preprocess_image(digit_image(True))
    np.testing.assert_array_equal(tensor, other)
    y, x = np.indices(preview.shape)
    assert abs((x * preview).sum() / preview.sum() - 13.5) <= 0.6
    assert abs((y * preview).sum() / preview.sum() - 13.5) <= 0.6
    assert not preview[:3].any() and not preview[-3:].any()


@pytest.mark.parametrize('value', [0, 128, 255])
def test_blank_rejected(value):
    with pytest.raises(ValueError, match='contenido'):
        preprocess_image(Image.new('L', (60, 60), value))


def test_missing_image_rejected():
    with pytest.raises(ValueError, match='imagen'):
        preprocess_image(None)


def test_transparent_background():
    image = Image.new('RGBA', (100, 120), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((40, 20, 55, 95), fill=(0, 0, 0, 255))
    tensor, _ = preprocess_image(image)
    expected, _ = preprocess_image(digit_image(True))
    np.testing.assert_array_equal(tensor, expected)


def test_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match='train.py'):
        load_model(tmp_path / 'missing.keras')


class FakeModel:
    def __init__(self, output):
        self.output = output

    def __call__(self, tensor, training=False):
        assert tensor.shape == (1, 784) and tensor.dtype == np.float32
        assert training is False
        return np.asarray(self.output)


def test_probabilities():
    probabilities = [[0.01] * 9 + [0.91]]
    digit, scores, preview = predict_digit(digit_image(), model=FakeModel(probabilities))
    assert digit == 9
    assert set(scores) == {str(i) for i in range(10)}
    assert sum(scores.values()) == pytest.approx(1)
    assert scores['9'] == pytest.approx(.91)
    assert preview.shape == (28, 28)


@pytest.mark.parametrize('output', [[[1.0] * 10], [[float('nan')] * 10], [[-.1] + [.1] * 9], [[.5, .5]]])
def test_invalid_distribution(output):
    with pytest.raises(ValueError, match='Softmax'):
        predict_digit(digit_image(), model=FakeModel(output))


def test_real_keras_roundtrip(tmp_path):
    # Pesos aleatorios: verifica formato y persistencia, no exactitud predictiva.
    from train import build_model
    model = build_model()
    assert model.input_shape == (None, 784)
    assert model.output_shape == (None, 10)
    assert model.count_params() == 55050
    assert [layer.units for layer in model.layers if hasattr(layer, 'units')] == [64, 64, 10]
    assert [layer.activation.__name__ for layer in model.layers if hasattr(layer, 'activation')] == ['relu', 'relu', 'softmax']
    path = tmp_path / 'random.keras'
    model.save(path)
    loaded = load_model(path)
    digit, scores, _ = predict_digit(digit_image(), model=loaded)
    assert 0 <= digit <= 9
    assert all(0 <= score <= 1 for score in scores.values())
    assert sum(scores.values()) == pytest.approx(1, abs=1e-5)
    tensor, _ = preprocess_image(digit_image())
    np.testing.assert_allclose(model(tensor, training=False), loaded(tensor, training=False))
