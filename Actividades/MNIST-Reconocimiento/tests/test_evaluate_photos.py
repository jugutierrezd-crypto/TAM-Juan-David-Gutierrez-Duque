"""Public CLI checks with isolated generated images and a deterministic model."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

import evaluate_photos


def image(path, blank=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    source = Image.new('RGB', (200, 300), 'white')
    if not blank:
        ImageDraw.Draw(source).line((100, 100, 100, 190), fill='black', width=7)
    source.save(path)


@pytest.fixture
def model(monkeypatch, tmp_path):
    artifact = tmp_path / 'test-model.keras'
    artifact.write_bytes(b'fake model artifact; no weights trained')
    monkeypatch.setattr(evaluate_photos, 'MODEL_PATH', artifact)
    calls = []
    def predict(tensor, training=False):
        assert training is False and tensor.shape == (1, 784)
        calls.append(tensor.copy())
        return np.eye(10, dtype=np.float32)[[1]]
    monkeypatch.setattr(evaluate_photos, 'load_model', lambda: predict)
    return calls


def test_comparison_metrics_and_rejections(tmp_path, model, capsys):
    root, output = tmp_path / 'photos', tmp_path / 'report'
    image(root / '1' / 'one.png')
    image(root / '4' / 'blank.png', blank=True)
    assert evaluate_photos.main([str(root), '--output', str(output), '--partition', 'development']) == 0
    captured = capsys.readouterr()
    assert captured.out == f'Evaluación guardada en {output.resolve()}\n'
    assert captured.err == ''
    rows = list(csv.DictReader((output / 'comparison.csv').open()))
    assert len(rows) == 2 and [r['label'] for r in rows] == ['1', '4']
    for mode in ('legacy', 'adaptive'):
        assert rows[0][f'{mode}_prediction'] == '1'
        assert rows[0][f'{mode}_correct'] == 'True'
        assert float(rows[0][f'{mode}_confidence']) == 1
        assert float(rows[0][f'{mode}_inference_ms']) >= 0
        assert float(rows[0][f'{mode}_preprocessing_ms']) >= 0
        assert rows[1][f'{mode}_prediction'] == ''
        assert rows[1][f'{mode}_correct'] == 'False'
        assert rows[1][f'{mode}_status'] == 'rejected'
        assert rows[1][f'{mode}_inference_ms'] == ''
        assert rows[1][f'{mode}_error']
    report = json.loads((output / 'summary.json').read_text())
    assert report['partition'] == 'development' and not report['independent_claim']
    assert len(report['inputs']) == 2 and all(len(x['sha256']) == 64 for x in report['inputs'])
    assert report['model_sha256'] == hashlib.sha256(b'fake model artifact; no weights trained').hexdigest()
    for mode in ('legacy', 'adaptive'):
        metrics = report['metrics'][mode]
        assert metrics['accuracy'] == .5 and metrics['coverage'] == .5
        assert metrics['confusion_matrix'][1][1] == 1
        assert sum(map(sum, metrics['confusion_matrix'])) == 1
        assert metrics['rejected_by_label'][4] == 1
        assert metrics['per_class'][1] == dict(label=1, support=1, precision=1., recall=1., f1=1.)
        assert metrics['per_class'][4] == dict(label=4, support=1, precision=0., recall=0., f1=0.)
        assert metrics['macro_f1'] == metrics['weighted_f1'] == .5
        assert metrics['per_class'][0]['recall'] is None
    assert report['evidence_status'] == 'descriptive_only'
    assert len(model) == 3  # warmup plus two accepted systems, never blank input
    snapshot = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(SystemExit) as failure:
        evaluate_photos.main([str(root), '--output', str(output)])
    assert failure.value.code == 2
    assert {p.name: p.read_bytes() for p in output.iterdir()} == snapshot
    assert len(model) == 3


@pytest.mark.parametrize('case', ['empty', 'bad-label', 'duplicate', 'development-leak', 'corrupt'])
def test_invalid_corpus_leaves_no_output(tmp_path, model, capsys, case):
    root, output = tmp_path / 'photos', tmp_path / 'report'
    root.mkdir()
    if case == 'bad-label':
        image(root / '10' / 'bad.png')
    elif case == 'duplicate':
        image(root / '1' / 'a.png')
        (root / '1' / 'b.png').write_bytes((root / '1' / 'a.png').read_bytes())
    elif case == 'development-leak':
        image(root / '1' / 'a.png')
        image(tmp_path / 'development' / 'original.png')
    elif case == 'corrupt':
        (root / '1').mkdir()
        (root / '1' / 'bad.png').write_bytes(b'not an image')
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(SystemExit) as failure:
        evaluate_photos.main([str(root), '--output', str(output),
                              '--development-root', str(tmp_path / 'development')])
    assert failure.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'error:' in captured.err
    assert not output.exists() and not model
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
