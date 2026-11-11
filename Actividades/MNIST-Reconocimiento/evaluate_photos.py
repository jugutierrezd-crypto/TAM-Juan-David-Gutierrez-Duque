"""Paired evaluation of two preprocessors, using one unchanged local model."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps

from diagnose import infer
from photo import preprocess_photo
from predict import MODEL_PATH, ROOT, load_model, preprocess_image

SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
REGISTRY = ROOT / 'datos/fotos_reales/development.json'


def fingerprint(path):
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert('RGBA')
        rgb = Image.alpha_composite(Image.new('RGBA', rgb.size, 'white'), rgb).convert('RGB')
        pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), pixels_sha256=pixels)


def collect(root):
    entries = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in SUFFIXES:
            relative = path.relative_to(root)
            if len(relative.parts) != 2 or relative.parts[0] not in list('0123456789'):
                raise ValueError(f'La fotografía debe estar directamente en una carpeta 0–9: {relative}')
            entries.append(dict(path=relative.as_posix(), label=int(relative.parts[0]), **fingerprint(path)))
    if not entries:
        raise ValueError('No hay fotografías etiquetadas; cargá archivos en las carpetas 0–9.')
    return entries


def metrics(rows, mode):
    matrix = np.zeros((10, 10), dtype=int)
    support, rejected = np.zeros(10, dtype=int), np.zeros(10, dtype=int)
    for row in rows:
        label, prediction = row['label'], row[f'{mode}_prediction']
        support[label] += 1
        if prediction is None:
            rejected[label] += 1
        else:
            matrix[label, prediction] += 1
    per_class = []
    for label in range(10):
        tp, predicted, actual = int(matrix[label, label]), int(matrix[:, label].sum()), int(support[label])
        precision = tp / predicted if predicted else (0. if actual else None)
        recall = tp / actual if actual else None
        f1 = 2 * tp / (predicted + actual) if predicted + actual else None
        per_class.append(dict(label=label, support=actual, precision=precision, recall=recall, f1=f1))
    present = [row for row in per_class if row['support']]
    result = dict(accuracy=float(matrix.trace() / len(rows)), coverage=float(matrix.sum() / len(rows)),
                  labels=list(range(10)), confusion_matrix=matrix.tolist(),
                  rejected_by_label=rejected.tolist(), per_class=per_class)
    for field in ('precision', 'recall', 'f1'):
        result[f'macro_{field}'] = sum(row[field] for row in present) / len(present)
        result[f'weighted_{field}'] = sum(row[field] * row['support'] for row in present) / len(rows)
    return result


def evaluate(image, model, mode):
    start = perf_counter()
    result = dict(prediction=None, confidence=None, status='rejected', error='', inference_ms=None)
    tensor = None
    try:
        if mode == 'legacy':
            tensor, _ = preprocess_image(image)
            result['status'] = 'unassessed'
        else:
            photo = preprocess_photo(image)
            result['status'] = photo.quality['status']
            if result['status'] == 'uncertain':
                result['error'] = photo.quality['warning']
            else:
                tensor = photo.tensor
    except ValueError as error:
        result['error'] = str(error)
    result['preprocessing_ms'] = (perf_counter() - start) * 1000
    if tensor is not None:
        start = perf_counter()
        try:
            prediction, probabilities = infer(model, tensor)
            result.update(prediction=prediction, confidence=probabilities[str(prediction)])
        except ValueError as error:
            result.update(status='rejected', error=str(error))
        result['inference_ms'] = (perf_counter() - start) * 1000
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compara preprocesamiento original/fotográfico con los mismos pesos; no entrena ni descarga.')
    parser.add_argument('root', type=Path, help='Directorio con carpetas 0–9 y fotografías directamente dentro.')
    parser.add_argument('--output', required=True, type=Path, help='Directorio nuevo para comparison.csv y summary.json; nunca sobrescribe.')
    parser.add_argument('--partition', choices=['development', 'evaluation'], default='evaluation',
                        help='evaluation exige fotos nuevas; development no constituye evaluación independiente.')
    parser.add_argument('--development-root', type=Path, default=ROOT / 'datos/fotos_reales/desarrollo',
                        help='Fotos de desarrollo excluidas de evaluation, además del registro histórico fijo.')
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError('El directorio de salida ya existe; elegí uno nuevo.')
        if args.root.resolve() == args.development_root.resolve() and args.partition == 'evaluation':
            raise ValueError('La carpeta de desarrollo requiere --partition development.')
        entries = collect(args.root)
        known = json.loads(REGISTRY.read_text(encoding='utf-8')) if REGISTRY.exists() else []
        if args.development_root.exists():
            for path in args.development_root.rglob('*'):
                if path.is_file() and path.suffix.lower() in SUFFIXES:
                    known.append(fingerprint(path))
        seen = set()
        for entry in entries:
            identity = entry['pixels_sha256']
            if identity in seen:
                raise ValueError(f'Fotografía duplicada: {entry["path"]}')
            seen.add(identity)
            if args.partition == 'evaluation' and any(
                    entry['sha256'] == old['sha256'] or identity == old['pixels_sha256'] for old in known):
                raise ValueError(f'Foto utilizada en desarrollo: {entry["path"]}; no es evaluación independiente.')
        model_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    model = load_model()
    infer(model, np.zeros((1, 784), dtype=np.float32))  # initialization excluded from timing
    rows = []
    for index, entry in enumerate(entries):
        row = dict(file=entry['path'], label=entry['label'])
        with Image.open(args.root / entry['path']) as image:
            image.load()  # exclude file decoding from preprocessing timing
            # Alternate ordering to limit systematic first-system timing bias.
            modes = ('legacy', 'adaptive') if index % 2 == 0 else ('adaptive', 'legacy')
            results = {mode: evaluate(image, model, mode) for mode in modes}
        for mode in ('legacy', 'adaptive'):
            row.update({f'{mode}_{key}': value for key, value in results[mode].items()})
            row[f'{mode}_correct'] = results[mode]['prediction'] == entry['label']
        rows.append(row)
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != model_hash:
        raise RuntimeError('El modelo cambió durante la evaluación; no se exportan resultados.')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    counts = [sum(entry['label'] == label for entry in entries) for label in range(10)]
    summary = dict(partition=args.partition, independent_claim=False,
                   independence='User must confirm unseen authors/sessions/originals; hashes only detect exact byte/pixel reuse.',
                   evidence_status='descriptive_only' if args.partition == 'development' or min(counts) < 20 else 'holdout_candidate_not_statistical_proof',
                   support=counts, inputs=entries, root=str(args.root.resolve()),
                   model=str(MODEL_PATH.resolve()), model_sha256=model_hash,
                   git_head=head.stdout.strip() if head.returncode == 0 else None,
                   code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ('evaluate_photos.py', 'predict.py', 'photo.py', 'diagnose.py')},
                   environment=dict(python=platform.python_version(),
                                    packages={name: version(name) for name in ('numpy', 'pillow', 'tensorflow-cpu', 'keras', 'opencv-python-headless')}),
                   timing='One measurement per image/system after one model warmup; excludes decoding, model load and export; not a benchmark.',
                   confidence='Uncalibrated softmax, not probability of correctness.',
                   metric_policy='Rejected/uncertain count as incorrect in accuracy and recall; matrix excludes them, rejected_by_label retains them. Macro uses supported true classes; undefined values null, zero predicted with positive support uses zero.',
                   metrics={mode: metrics(rows, mode) for mode in ('legacy', 'adaptive')})
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Evaluación guardada en {args.output.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
