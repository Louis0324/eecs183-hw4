"""Small, atomic per-layer prediction exports for HW4 A (no checkpoints)."""
import json
from pathlib import Path
import warnings

DATASET_ID = 'hw4-compact-probes-2026-v1'


def save_probe_predictions(task, result, output_dir):
    """Save one completed layer. Missing predictions warn without blocking plots.

    result['test_predictions'] must follow the loader's original test-row order:
    phoneme class IDs [N], or unscaled MFCC predictions [N,13].
    """
    if task not in {'phoneme', 'mfcc'}:
        raise ValueError('task must be phoneme or mfcc')
    layer = result.get('layer')
    if type(layer) is not int or not 0 <= layer <= 12:
        raise ValueError('layer must be an integer from 0 to 12')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{task}_layer_{layer}.json'
    predictions = result.get('test_predictions')
    if predictions is None:
        # Do not silently retain an older run's predictions for this layer.
        path.unlink(missing_ok=True)
        warnings.warn(f'{task} layer {layer}: return test_predictions for performance credit. Code and plots can still be graded.')
        return None
    if hasattr(predictions, 'detach'):
        predictions = predictions.detach().cpu()
    if hasattr(predictions, 'tolist'):
        predictions = predictions.tolist()
    payload = dict(format_version=1, dataset_id=DATASET_ID, task=task,
                   layer=layer, predictions=predictions)
    try:
        contents = json.dumps(payload, allow_nan=False, separators=(',', ':'))
    except (TypeError, ValueError) as error:
        path.unlink(missing_ok=True)
        warnings.warn(f'{task} layer {layer}: could not export predictions ({error}); other layers remain available.')
        return None
    temporary = path.with_suffix('.tmp')
    temporary.write_text(contents)
    temporary.replace(path)
    return path
