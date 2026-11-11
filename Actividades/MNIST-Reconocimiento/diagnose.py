"""Save reproducible photographic diagnostics; never train or overwrite evidence."""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from predict import MODEL_PATH, ROOT, load_model, preprocess_image


def statistics(image):
    x = np.asarray(image, dtype=np.float64)
    return dict(mean=float(x.mean()), variance=float(x.var()), std=float(x.std()),
                active_threshold=15, active_fraction=float(np.mean(x > 15)))


def similarities(image, reference):
    x, y = (np.asarray(a, dtype=np.float64).ravel() for a in (image, reference))
    if x.shape != y.shape:
        raise ValueError('Las imágenes comparadas deben tener las mismas dimensiones.')
    a, b = x - x.mean(), y - y.mean()
    norm, centered_norm = np.linalg.norm(x) * np.linalg.norm(y), np.linalg.norm(a) * np.linalg.norm(b)
    return dict(pearson=float(a @ b / centered_norm) if centered_norm else None,
                covariance=float(a @ b / (x.size - 1)),
                cosine=float(x @ y / norm) if norm else None)


def contact_sheet(images):
    """Uniform-size panels without changing saved full-resolution stage arrays."""
    sheet = Image.new('RGB', (4 * 240, ((len(images) + 3) // 4) * 280), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, (caption, array) in enumerate(images):
        x, y = (index % 4) * 240, (index // 4) * 280
        panel = array.copy() if isinstance(array, Image.Image) else Image.fromarray(array)
        panel = ImageOps.contain(panel.convert('RGB'), (232, 244))
        sheet.paste(panel, (x + (240 - panel.width) // 2, y + 26))
        draw.text((x + 4, y + 5), caption, fill='black')
    return sheet


def infer(model, tensor):
    output = np.asarray(model(tensor, training=False))
    if (output.shape != (1, 10) or not np.isfinite(output).all()
            or (output < 0).any() or (output > 1).any()
            or not np.isclose(output.sum(), 1, atol=1e-5)):
        raise ValueError('Salida Softmax inválida: se esperan diez probabilidades que sumen uno.')
    return int(output.argmax()), {str(i): float(p) for i, p in enumerate(output[0])}


def diagnose(source, output, model, *, mode='legacy', references=None):
    source, output = Path(source), Path(output)
    # Reserve a fresh directory, refusing to overwrite any previous diagnosis.
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output / ('original' + source.suffix.lower()))
    tensor = preview = digit = scores = inference_ms = None
    error = ''
    with Image.open(source) as opened:
        original = ImageOps.exif_transpose(opened).convert('RGBA')
        original = Image.alpha_composite(Image.new('RGBA', original.size, 'white'), original).convert('RGB')
        # Safe fallback only: preprocess_photo cannot expose internal stages on failure.
        trace = {'gray': np.asarray(original.convert('L'))}
        start = perf_counter()
        try:
            if mode == 'legacy':
                tensor, preview = preprocess_image(opened, trace=trace)
                quality = {'status': 'unassessed', 'warning': 'Recorte global sin aislamiento validado.'}
            else:
                from photo import preprocess_photo
                result = preprocess_photo(opened, remove_grid=mode != 'no-grid', use_clahe=mode != 'no-clahe')
                tensor, preview, trace, quality = result.tensor, result.preview, result.stages, result.quality
        except ValueError as failure:
            error = str(failure)
            quality = {'status': 'rejected', 'warning': error}
        preprocessing_ms = (perf_counter() - start) * 1000
        if tensor is not None:
            np.save(output / 'tensor.npy', tensor, allow_pickle=False)
            start = perf_counter()
            try:
                digit, scores = infer(model, tensor)
                inference_ms = (perf_counter() - start) * 1000
            except ValueError as failure:
                error = str(failure)
                quality = {'status': 'rejected', 'warning': error}

    statuses, artifacts = {}, {}
    panels = [('original (EXIF, papel blanco)', original)]
    for name, stage in trace.items():
        if not isinstance(stage, np.ndarray):
            continue
        filename = 'original_stage.png' if name == 'original' else f'{name}.png'
        Image.fromarray(stage).save(output / filename)
        artifacts[name] = filename
        status = 'applied'
        if mode == 'legacy' and name in ('contrast', 'grid_removed'):
            status = 'not applied (legacy)'
        elif mode == 'no-grid' and name == 'grid_removed':
            status = 'not applied (no-grid)'
        elif mode == 'no-clahe' and name == 'contrast':
            status = 'not applied (no-clahe)'
        elif tensor is None:
            status = 'diagnostic only; partial stages unavailable'
        statuses[name] = status
        if name != 'original':
            caption = name if status == 'applied' else f'{name}: sin aplicar / parcial'
            panels.append((caption, stage))
    contact_sheet(panels).save(output / 'comparison.png')
    coordinates = trace.get('coordinates')
    if coordinates is None and tensor is not None:
        width, height = original.size
        coordinates = dict(source_size=[width, height], source_crop=[0, 0, width, height],
                           working_size=[width, height], source_per_working_pixel=[1, 1], exif_corrected=True)
    report = dict(source=str(source.resolve()), mode=mode, prediction=digit, probabilities=scores,
                  confidence=max(scores.values()) if scores else None, quality=quality, quality_ground_truth=False,
                  error=error, bbox=trace.get('bbox'), coordinates=coordinates,
                  preprocessing_ms=preprocessing_ms, inference_ms=inference_ms,
                  statistics=statistics(preview) if preview is not None else None,
                  tensor_shape=list(tensor.shape) if tensor is not None else None,
                  tensor_dtype=str(tensor.dtype) if tensor is not None else None,
                  stages=statuses, artifacts=artifacts,
                  partial_evidence=tensor is None)
    report['mnist_similarities'] = ({str(label): similarities(preview, reference)
                                     for label, reference in references.items()}
                                    if references is not None and preview is not None else None)
    if references is not None:
        contact_sheet([(f'MNIST training class {label}', ref) for label, ref in references.items()]).save(output / 'mnist_references.png')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Diagnóstico local sin entrenamiento. Crea evidencia nueva; nunca sobrescribe resultados.')
    parser.add_argument('images', nargs='*', type=Path, help='Fotografías locales (sin etiquetas usadas en inferencia).')
    parser.add_argument('--manifest', type=Path, help='CSV independiente con columnas path,label; rutas relativas al CSV.')
    parser.add_argument('--output', type=Path, required=True, help='Directorio nuevo: etapas PNG, tensor NPY, JSON, evaluation.csv y comparison.csv (compare).' )
    parser.add_argument('--mode', choices=['legacy', 'adaptive', 'compare'], default='legacy', help='compare incluye ablaciones sin cuadrícula y sin CLAHE.')
    parser.add_argument('--mnist', type=Path, default=ROOT / 'resultados/.keras/datasets/mnist.npz', help='Caché local; usa primer ejemplo de entrenamiento por clase, sin descargar.')
    args = parser.parse_args(argv)
    entries = [(path, '') for path in args.images]
    if args.manifest:
        with args.manifest.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if row['label'] not in list('0123456789'):
                    parser.error('Cada etiqueta debe ser un entero de 0 a 9.')
                entries.append((args.manifest.parent / row['path'], row['label']))
    if not entries or any(not path.is_file() for path, _ in entries):
        parser.error('Indicá al menos una fotografía existente; revisá todas las rutas.')
    if args.output.exists():
        parser.error('El directorio de salida ya existe; elegí uno nuevo.')
    references = None
    if args.mnist.is_file():
        with np.load(args.mnist, allow_pickle=False) as data:
            data_labels = data['y_train']
            references = {i: data['x_train'][np.flatnonzero(data_labels == i)[0]] for i in range(10)}
    model = load_model()
    # Avoid mixing model initialization / warmup with measured image inference.
    infer(model, np.zeros((1, 784), dtype=np.float32))
    args.output.mkdir(parents=True)
    modes = ['legacy', 'adaptive', 'no-grid', 'no-clahe'] if args.mode == 'compare' else [args.mode]
    rows, comparisons = [], []
    for index, (source, label) in enumerate(entries):
        reports = {}
        for mode in modes:
            destination = args.output / f'{index:03d}_{source.stem}_{mode}'
            report = diagnose(source, destination, model, mode=mode, references=references)
            reports[mode] = report
            rows.append(dict(file=str(source), label=label, mode=mode, prediction=report['prediction'],
                             confidence=report['confidence'], quality=report['quality']['status'],
                             quality_ground_truth=False, warning=report['quality']['warning'],
                             preprocessing_ms=report['preprocessing_ms'], inference_ms=report['inference_ms'],
                             error=report['error']))
            if report['quality']['warning']:
                print(f'Advertencia [{source.name}/{mode}]: {report["quality"]["warning"]}', file=sys.stderr)
        if args.mode == 'compare':
            comparison = dict(file=str(source), label=label, quality_ground_truth=False)
            for prefix, mode in [('old', 'legacy'), ('new', 'adaptive')]:
                report = reports[mode]
                for field in ('prediction', 'confidence', 'preprocessing_ms', 'inference_ms', 'error'):
                    comparison[f'{prefix}_{field}'] = report[field]
                comparison[f'{prefix}_quality'] = report['quality']['status']
                comparison[f'{prefix}_warning'] = report['quality']['warning']
            comparisons.append(comparison)
    write_csv(args.output / 'evaluation.csv', rows)
    if comparisons:
        write_csv(args.output / 'comparison.csv', comparisons)
    metadata = dict(model=str(MODEL_PATH.resolve()), model_mtime=MODEL_PATH.stat().st_mtime,
                    model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                    model_input=str(model.input_shape), model_output=str(model.output_shape),
                    scale='float32 0..255', references='first training sample of each class' if references else 'unavailable',
                    reference_indices={str(i): int(np.flatnonzero(data_labels == i)[0]) for i in range(10)} if references else None,
                    limitation='Exploratory photos are not independent evaluation. Softmax is not calibrated confidence.')
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Diagnóstico guardado en {args.output.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
