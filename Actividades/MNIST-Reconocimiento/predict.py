"""Adapta una fotografía de un solo dígito al contrato MNIST original."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'mnist.keras'


def preprocess_image(image, *, trace=None):
    """Devuelve (lote float32 de 1×784, vista uint8 de 28×28), sin normalizar."""
    if image is None:
        raise ValueError('Seleccioná una imagen con un solo dígito.')
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = ImageOps.exif_transpose(image)
    # Transparencia equivale a papel blanco, no a un fondo negro artificial.
    rgba = image.convert('RGBA')
    paper = Image.new('RGBA', rgba.size, 'white')
    gray = np.asarray(Image.alpha_composite(paper, rgba).convert('L'))
    if gray.size == 0 or min(gray.shape) < 2:
        raise ValueError('Imagen sin contenido suficiente.')
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    foreground = 255 - gray if np.median(border) > 127 else gray.copy()
    # Restar el fondo evita que un papel gris ocupe todo el recorte.
    background = float(np.median(np.concatenate((foreground[0], foreground[-1], foreground[:, 0], foreground[:, -1]))))
    foreground = np.clip(foreground.astype(np.float32) - background, 0, 255)
    peak = float(foreground.max())
    if peak < 15:
        raise ValueError('Imagen sin contenido: no se distingue un dígito del fondo.')
    mask = foreground > max(15, peak * 0.2)
    if not mask.any():
        raise ValueError('Imagen sin contenido: no se distingue un dígito del fondo.')
    ys, xs = np.nonzero(mask)
    crop = foreground[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if trace is not None:
        trace.update(gray=gray.copy(), contrast=gray.copy(), grid_removed=gray.copy(),
                     mask=mask.astype(np.uint8) * 255, crop=crop.astype(np.uint8),
                     bbox=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
    height, width = crop.shape
    scale = 20 / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize(size, Image.Resampling.LANCZOS))
    canvas = np.zeros((28, 28), dtype=np.uint8)
    left, top = (28 - size[0]) // 2, (28 - size[1]) // 2
    canvas[top:top + size[1], left:left + size[0]] = resized
    # Centro de masa: el trazo, no solo su caja, queda cerca de (13.5,13.5).
    y, x = np.indices(canvas.shape)
    mass = float(canvas.sum())
    dx = int(round(13.5 - float((x * canvas).sum()) / mass))
    dy = int(round(13.5 - float((y * canvas).sum()) / mass))
    # Acotar el desplazamiento impide cortar trazos asimétricos al centrar.
    dx = int(np.clip(dx, -left, 28 - left - size[0]))
    dy = int(np.clip(dy, -top, 28 - top - size[1]))
    centered = np.zeros_like(canvas)
    centered[top + dy:top + dy + size[1], left + dx:left + dx + size[0]] = resized
    if trace is not None:
        trace['final'] = centered.copy()
    return centered.astype(np.float32).reshape(1, 784), centered


def load_model(model_path=MODEL_PATH):
    """Carga explícita; nunca entrena ni descarga MNIST si falta el archivo."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f'No existe el modelo: {path}. Ejecutá train.py cuando autorices el entrenamiento.')
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def predict_digit(image, model=None):
    """Retorna clase, probabilidades de las diez clases y tensor visible."""
    tensor, preview = preprocess_image(image)
    if model is None:
        model = load_model()
    output = np.asarray(model(tensor, training=False))
    if (output.shape != (1, 10) or not np.isfinite(output).all()
            or (output < 0).any() or (output > 1).any()
            or not np.isclose(output.sum(), 1.0, atol=1e-5)):
        raise ValueError('Salida Softmax inválida: se esperan diez probabilidades que sumen uno.')
    scores = {str(index): float(value) for index, value in enumerate(output[0])}
    return int(output[0].argmax()), scores, preview
