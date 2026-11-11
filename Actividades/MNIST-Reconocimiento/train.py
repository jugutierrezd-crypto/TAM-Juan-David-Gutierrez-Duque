"""Entrenamiento explícito de la arquitectura del cuaderno, sin normalización."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def build_model():
    """784 → 64 ReLU → 64 ReLU → 10 Softmax: 55.050 parámetros."""
    import tensorflow as tf
    inputs = tf.keras.layers.Input(shape=(784,))
    hidden = tf.keras.layers.Dense(64, activation='relu')(inputs)
    hidden = tf.keras.layers.Dense(64, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser(description='Entrena MNIST durante 30 épocas, evalúa test y guarda models/mnist.keras. No se ejecuta desde app.py.')
    parser.parse_args()
    target = ROOT / 'models' / 'mnist.keras'
    if target.exists():
        parser.error(f'Ya existe {target}; no se sobrescribe. Conservá tu modelo antes de volver a entrenar.')
    # Keras 3 requiere un nombre relativo para MNIST; KERAS_HOME fija su caché.
    os.environ['KERAS_HOME'] = str(ROOT / 'resultados' / '.keras')
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Keras convierte implícitamente a float32 en el cuaderno. Aquí es explícito.
    x_train = x_train.reshape(-1, 784).astype(np.float32)
    x_test = x_test.reshape(-1, 784).astype(np.float32)
    model = build_model()
    model.summary()
    history = model.fit(x_train, y_train, epochs=30, batch_size=32)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(target)
    results = {'test_loss': float(loss), 'test_accuracy': float(accuracy), 'history': history.history}
    # Solo se escribe este informe nuevo; nunca se sustituye uno anterior.
    report = ROOT / 'resultados' / 'training.json'
    if not report.exists():
        report.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Modelo guardado: {target}')
    print(f'Test: loss={loss:.6f}, accuracy={accuracy:.6f}')


if __name__ == '__main__':
    main()
