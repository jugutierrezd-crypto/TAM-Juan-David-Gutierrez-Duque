"""Public diagnostic artifacts, independent of trained weights or downloads."""
import json
import numpy as np
from PIL import Image, ImageDraw
import csv
import pytest
import diagnose as diagnostic
from diagnose import diagnose, statistics, similarities
from predict import preprocess_image


class RecordingModel:
    def __call__(self, tensor, training=False):
        self.tensor = tensor.copy()
        return np.array([[0.1] * 10], dtype=np.float32)


def test_exact_artifacts(tmp_path):
    image = Image.new('RGB', (100, 120), 'white')
    ImageDraw.Draw(image).line((45, 20, 45, 95), fill='black', width=8)
    source = tmp_path / 'source.png'
    image.save(source)
    model = RecordingModel()
    output = tmp_path / 'diagnostic'
    report = diagnose(source, output, model)
    assert (output / 'original.png').read_bytes() == source.read_bytes()
    tensor = np.load(output / 'tensor.npy')
    np.testing.assert_array_equal(tensor, model.tensor)
    np.testing.assert_array_equal(tensor, preprocess_image(image)[0])
    for name in ['gray', 'contrast', 'grid_removed', 'mask', 'crop', 'final', 'comparison']:
        assert (output / f'{name}.png').is_file()
    saved = json.loads((output / 'report.json').read_text())
    assert len(saved['probabilities']) == 10
    assert saved['stages']['grid_removed'] == 'not applied (legacy)'
    assert report['statistics']['active_threshold'] == 15
    # Existing evidence must not be overwritten.
    import pytest
    with pytest.raises(FileExistsError):
        diagnose(source, output, model)


def test_adaptive_stages_and_coordinates(tmp_path):
    from photo import preprocess_photo
    image = Image.new('RGB', (160, 180), 'white')
    ImageDraw.Draw(image).line((65, 65, 80, 55, 80, 125, 100, 125), fill='black', width=7)
    source = tmp_path / 'one.png'
    image.save(source)
    expected = preprocess_photo(image)
    model = RecordingModel()
    output = tmp_path / 'adaptive'
    references = {i: np.full((28, 28), i, dtype=np.uint8) for i in range(10)}
    report = diagnose(source, output, model, mode='adaptive', references=references)
    saved = json.loads((output / 'report.json').read_text())
    assert saved['coordinates'] == expected.stages['coordinates']
    assert saved['bbox'] == expected.stages['bbox']
    assert len(saved['mnist_similarities']) == 10
    assert saved['quality_ground_truth'] is False
    for name, stage in expected.stages.items():
        if isinstance(stage, np.ndarray):
            np.testing.assert_array_equal(np.asarray(Image.open(output / saved['artifacts'][name])), stage)
    np.testing.assert_array_equal(np.load(output / 'tensor.npy'), model.tensor)
    assert report['statistics'] == statistics(expected.preview)


def test_rejection_keeps_available_evidence(tmp_path):
    source = tmp_path / 'blank.png'
    Image.new('RGB', (80, 90), 'white').save(source)
    output = tmp_path / 'rejected'
    model = RecordingModel()
    report = diagnose(source, output, model, mode='adaptive')
    assert report['quality']['status'] == 'rejected'
    assert report['quality']['warning'] == 'No se distingue un dígito del fondo; probá otra foto o un recorte.'
    assert report['prediction'] is None and report['probabilities'] is None
    assert report['inference_ms'] is None and report['statistics'] is None
    assert not hasattr(model, 'tensor')
    assert not (output / 'tensor.npy').exists()
    assert (output / 'original.png').read_bytes() == source.read_bytes()
    assert (output / 'gray.png').is_file() and (output / 'comparison.png').is_file()
    assert json.loads((output / 'report.json').read_text()) == report


def test_compare_table_references_and_labels(tmp_path, monkeypatch, capsys):
    source = tmp_path / 'one.png'
    image = Image.new('RGB', (160, 180), 'white')
    ImageDraw.Draw(image).line((65, 65, 80, 55, 80, 125, 100, 125), fill='black', width=7)
    image.save(source)
    manifest = tmp_path / 'labels.csv'
    manifest.write_text('path,label\none.png,7\n', encoding='utf-8')
    cache = tmp_path / 'mnist.npz'
    np.savez(cache, x_train=np.arange(10 * 784, dtype=np.uint8).reshape(10, 28, 28), y_train=np.arange(10))
    model = RecordingModel()
    model.input_shape, model.output_shape = (None, 784), (None, 10)
    monkeypatch.setattr(diagnostic, 'load_model', lambda: model)
    monkeypatch.setattr(diagnostic, 'MODEL_PATH', source)
    output = tmp_path / 'comparison'
    args = ['--manifest', str(manifest), '--mode', 'compare', '--output', str(output), '--mnist', str(cache)]
    assert diagnostic.main(args) == 0
    captured = capsys.readouterr()
    assert captured.out == f'Diagnóstico guardado en {output.resolve()}\n'
    assert captured.err == 'Advertencia [one.png/legacy]: Recorte global sin aislamiento validado.\n'
    with (output / 'comparison.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    row = rows[0]
    assert row['label'] == '7'
    assert row['old_prediction'] == row['new_prediction'] == '0'
    assert float(row['old_confidence']) == float(row['new_confidence']) == float(np.float32(.1))
    assert row['quality_ground_truth'] == 'False'
    for prefix in ('old', 'new'):
        assert float(row[f'{prefix}_preprocessing_ms']) >= 0
        assert float(row[f'{prefix}_inference_ms']) >= 0
        assert row[f'{prefix}_quality']
    with (output / 'evaluation.csv').open() as stream:
        assert [r['mode'] for r in csv.DictReader(stream)] == ['legacy', 'adaptive', 'no-grid', 'no-clahe']
    for directory in output.iterdir():
        if directory.is_dir():
            report = json.loads((directory / 'report.json').read_text())
            assert 'label' not in report
            assert len(report['mnist_similarities']) == 10
            if report['mode'] == 'no-grid':
                assert report['stages']['grid_removed'] == 'not applied (no-grid)'
            if report['mode'] == 'no-clahe':
                assert report['stages']['contrast'] == 'not applied (no-clahe)'
    before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    with pytest.raises(SystemExit) as error:
        diagnostic.main(args)
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'El directorio de salida ya existe; elegí uno nuevo.' in captured.err
    assert before == {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}


def test_compare_rejected_rows_are_not_omitted(tmp_path, monkeypatch, capsys):
    source = tmp_path / 'blank.png'
    Image.new('RGB', (80, 90), 'white').save(source)
    model = RecordingModel()
    model.input_shape, model.output_shape = (None, 784), (None, 10)
    monkeypatch.setattr(diagnostic, 'load_model', lambda: model)
    monkeypatch.setattr(diagnostic, 'MODEL_PATH', source)
    output = tmp_path / 'rejections'
    assert diagnostic.main([str(source), '--mode', 'compare', '--output', str(output),
                            '--mnist', str(tmp_path / 'missing.npz')]) == 0
    captured = capsys.readouterr()
    assert captured.out == f'Diagnóstico guardado en {output.resolve()}\n'
    legacy_error = 'Imagen sin contenido: no se distingue un dígito del fondo.'
    adaptive_error = 'No se distingue un dígito del fondo; probá otra foto o un recorte.'
    assert captured.err == ''.join(f'Advertencia [blank.png/{mode}]: {message}\n' for mode, message in [
        ('legacy', legacy_error), ('adaptive', adaptive_error), ('no-grid', adaptive_error), ('no-clahe', adaptive_error)])
    with (output / 'evaluation.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    for row in rows:
        assert row['quality'] == 'rejected'
        assert row['prediction'] == row['confidence'] == row['inference_ms'] == ''
        assert row['error'] and float(row['preprocessing_ms']) >= 0
    with (output / 'comparison.csv').open() as stream:
        comparison = list(csv.DictReader(stream))
    assert len(comparison) == 1
    for prefix in ('old', 'new'):
        assert comparison[0][f'{prefix}_quality'] == 'rejected'
        assert comparison[0][f'{prefix}_prediction'] == ''
        assert comparison[0][f'{prefix}_error']
    # The only model call is initialization, never rejected-image inference.
    np.testing.assert_array_equal(model.tensor, np.zeros((1, 784), dtype=np.float32))
    for directory in output.iterdir():
        if directory.is_dir():
            assert (directory / 'original.png').read_bytes() == source.read_bytes()
            assert (directory / 'comparison.png').is_file()
            assert not (directory / 'tensor.npy').exists()
            assert json.loads((directory / 'report.json').read_text())['mnist_similarities'] is None


def test_statistics_and_similarity():
    x = np.arange(784, dtype=np.float32).reshape(28, 28) % 256
    stats = statistics(x)
    assert stats['mean'] == float(x.astype(float).mean())
    assert stats['variance'] == float(x.astype(float).var())
    assert stats['std'] == float(x.astype(float).std())
    assert stats['active_fraction'] == float(np.mean(x > 15))
    values = similarities(x, x)
    assert abs(values['pearson'] - 1) < 1e-12
    assert abs(values['cosine'] - 1) < 1e-12
    np.testing.assert_allclose(values['covariance'], np.var(x.astype(float), ddof=1), rtol=1e-12)
    assert similarities(np.zeros((28, 28)), x)['pearson'] is None
    assert similarities(np.zeros((28, 28)), x)['cosine'] is None
