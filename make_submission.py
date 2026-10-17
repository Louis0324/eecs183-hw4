"""Build one Gradescope ZIP from any available HW4 work; never run student code."""
import argparse
import ast
import json
import os
import tempfile
import zipfile
from pathlib import Path

NOTEBOOKS = ['hw4-a.ipynb', 'hw4-b.ipynb', 'hw4-c.ipynb']


def build_submission(root='.', report='report.pdf', out='submission.zip', *, additional_warnings=()):
    root, out = Path(root), Path(out)
    entries, notebook_bytes, warnings = {}, {}, list(additional_warnings)
    for name in NOTEBOOKS:
        path = root / name
        if not path.is_file():
            warnings.append(f'{name} missing; its code questions receive no credit.')
            continue
        try:
            doc = json.loads(path.read_text())
            if not isinstance(doc, dict) or doc.get('nbformat') != 4 or not isinstance(doc.get('cells'), list):
                raise ValueError('Expected a version 4 notebook')
            for i, cell in enumerate(doc['cells']):
                if not isinstance(cell, dict):
                    continue
                if cell.get('cell_type') == 'code':
                    try:
                        ast.parse(''.join(cell['source']))
                    except SyntaxError:
                        warnings.append(f'{name} cell {i + 1}: syntax/magic; definitions in this cell may be skipped.')
                    cell['outputs'], cell['execution_count'] = [], None
            doc.get('metadata', {}).pop('widgets', None)
            notebook_bytes[name] = json.dumps(doc, ensure_ascii=False).encode()
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            warnings.append(f'{name}: {error}; other notebooks can still be graded.')
        entries[name] = path
    if report is not None:
        path = Path(report)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            if path.read_bytes()[:5] != b'%PDF-':
                warnings.append('report.pdf is not a PDF; replace it before final submission.')
            entries['report.pdf'] = path
        else:
            warnings.append('report.pdf missing; code is still graded.')
    else:
        warnings.append('report.pdf omitted.')
    # Explicit allowlist: never include datasets, model weights, staff tests, or keys.
    patterns = ['asr_submission.txt', 'tts_submission.json', 'submission_*.txt',
                'results/asr_submission.txt', 'results/tts_submission.json',
                'results/submission_*.txt', 'results/a/*.json', 'results/a/*.png']
    conflicts = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and not path.is_symlink():
                name = ('results/a/' if path.parent == root / 'results/a' else 'results/') + path.name
                if name in conflicts:
                    continue
                if name in entries and entries[name].read_bytes() != path.read_bytes():
                    warnings.append(f'Conflicting {name} files were omitted; keep only the intended version and rebuild. Other work is included.')
                    conflicts.add(name)
                    entries.pop(name)
                    continue
                entries[name] = path
    for task in ['phoneme', 'mfcc']:
        count = sum(f'results/a/{task}_layer_{layer}.json' in entries for layer in range(13))
        if count < 13:
            warnings.append(f'A {task}: {count}/13 layer prediction files found; available layers and code are graded independently.')
    if not any(k.endswith('asr_submission.txt') for k in entries):
        warnings.append('ASR predictions missing; ASR code can still earn credit.')
    if not any(k.endswith('tts_submission.json') or Path(k).name.startswith('submission_') for k in entries):
        warnings.append('TTS predictions missing; TTS code can still earn credit.')
    for name, path in list(entries.items()):
        if path.is_symlink() or path.resolve() == out.resolve():
            raise ValueError(f'Unsafe input path: {path}')
        size = len(notebook_bytes[name]) if name in notebook_bytes else path.stat().st_size
        if size > 30 * 1024 * 1024:
            warnings.append(f'{path} exceeds 30 MiB and was omitted. Reduce its size; other files are included.')
            entries.pop(name)
    if out.suffix.lower() != '.zip':
        raise ValueError('Output filename must end in .zip')
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=out.parent, suffix='.zip')
    os.close(fd)
    try:
        with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, path in entries.items():
                if name in notebook_bytes:
                    z.writestr(name, notebook_bytes[name])
                else:
                    z.write(path, name)
            z.writestr('submission_manifest.json', json.dumps({'format_version': 1, 'files': sorted(entries), 'warnings': warnings}, indent=2))
        os.replace(temp, out)
    finally:
        Path(temp).unlink(missing_ok=True)
    for warning in warnings:
        print('Warning:', warning)
    print(f'Created {out}: {len(entries)} available files, {out.stat().st_size / 1024**2:.2f} MiB compressed.')
    parts = ', '.join(name[4].upper() for name in NOTEBOOKS if name in entries) or 'none'
    print(f'Included notebooks: {parts}. Shared report.pdf: {"included" if "report.pdf" in entries else "missing"}.')
    for name in sorted(entries):
        print('  ', name)
    print('Check this list against the work you want graded, and open report.pdf to confirm it is the latest combined report.')
    print('Checkpoints, datasets, pretrained weights, and audio are excluded. These checks do not grade correctness.')
    return out


def merge_submission(archive, root):
    """Recover earlier parts from our own ZIP without importing or executing them."""
    root = Path(root)
    pending = {}
    with zipfile.ZipFile(archive) as z:
        if {'make_submission.py', 'hw4_runtime.py'}.issubset(z.namelist()):
            raise ValueError('This is the course support bundle, not your saved submission ZIP')
        if sum(i.file_size for i in z.infolist()) > 100 * 1024 * 1024:
            raise ValueError('Submission archive exceeds 100 MiB')
        for info in z.infolist():
            p = Path(info.filename)
            allowed = (p.as_posix() in NOTEBOOKS + ['report.pdf'] or
                       len(p.parts) == 2 and p.parts[0] == 'results' and
                       (p.name in {'asr_submission.txt', 'tts_submission.json'} or
                        p.name.startswith('submission_') and p.suffix == '.txt') or
                       len(p.parts) == 3 and p.parts[:2] == ('results', 'a') and p.suffix in {'.json', '.png'})
            if not allowed or info.is_dir():
                continue
            target = root / p
            if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
                raise ValueError(f'Unsafe destination {target}')
            # Read/CRC-check every member before replacing earlier saved work.
            content = z.read(info)
            if p in pending and pending[p] != content:
                raise ValueError(f'Conflicting copies of {p} inside the ZIP')
            pending[p] = content
    if not pending:
        raise ValueError('No HW4 submission files found; select a ZIP made by a notebook submission cell')
    for p, content in pending.items():
        target = root / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return sorted(p.as_posix() for p in pending)


def copy_results(results_dir, destination):
    """Collect only submission artifacts, including results saved on Drive."""
    source, destination = Path(results_dir), Path(destination)
    for pattern in ['asr_submission.txt', 'tts_submission.json', 'submission_*.txt', 'a/*.json', 'a/*.png']:
        for path in source.glob(pattern):
            if not path.is_file() or path.is_symlink():
                continue
            target = destination / 'results' / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


def from_colab(part, include_report=True, include_previous=False, results_dir='results'):
    """Capture browser edits, optionally merge previous parts, and download a ZIP."""
    from google.colab import _message, files
    if part not in 'abc' or len(part) != 1:
        raise ValueError('part must be a, b, or c')
    # Never package the pristine starter copies as completed work.
    saved = Path('submission_work')
    saved.mkdir(exist_ok=True)
    warnings = []
    if include_previous:
        print('Step 1: upload your latest combined submission ZIP from earlier parts.')
        print('If you have separate A/B/C ZIPs, select the ones with different parts together. Do not select old and new versions of the same part.')
        print('No earlier work to add? Cancel this upload. Do not upload the course support bundle or data archives here.')
        for name, content in files.upload().items():
            try:
                if not name.lower().endswith('.zip'):
                    raise ValueError('Expected a submission .zip file')
                with tempfile.NamedTemporaryFile(suffix='.zip') as f:
                    f.write(content)
                    f.flush()
                    imported = merge_submission(f.name, saved)
                print(f'Imported {name}: {len(imported)} files.')
            except (OSError, ValueError, RuntimeError, NotImplementedError, zipfile.BadZipFile) as error:
                warning = f'Could not import {name}: {error}. Other work is still packaged; correct this upload and rerun before submitting.'
                warnings.append(warning)
                print('Warning:', warning)
    try:
        doc = _message.blocking_request('get_ipynb', timeout_sec=30)['ipynb']
        if doc.get('nbformat') != 4 or not doc.get('cells'):
            raise ValueError('Invalid notebook response')
        (saved / f'hw4-{part}.ipynb').write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    except Exception as error:
        print('Could not capture browser edits:', error)
        print('Download your edited .ipynb using File > Download, then upload it here; cancel to package other work.')
        uploaded = files.upload()
        notebooks = [content for name, content in uploaded.items() if name.lower().endswith('.ipynb')]
        if len(notebooks) == 1:
            (saved / f'hw4-{part}.ipynb').write_bytes(notebooks[0])
        else:
            warnings.append(f'The current {part.upper()} notebook was not captured. Upload exactly one edited .ipynb when prompted and rerun; any earlier saved copy was kept.')
    # Copy only available predictions/plots, including a custom Drive RESULTS path.
    copy_results('.', saved)
    copy_results(results_dir, saved)
    if include_report:
        print('Report: upload ONE combined PDF with sections A1, A2, A3, B1, C1. Separate PDFs are not merged.')
        print('Cancel to keep the report already imported/saved, or continue without one if it is not ready.')
        uploaded = files.upload()
        if uploaded:
            if len(uploaded) != 1:
                warnings.append('Multiple report files uploaded; none was used. Combine your answers into ONE report.pdf, then rerun. Any earlier report was kept.')
            else:
                name, content = next(iter(uploaded.items()))
                if not name.lower().endswith('.pdf') or not content.startswith(b'%PDF-'):
                    warnings.append('The uploaded report is not a PDF; it was not used. Export ONE combined report.pdf and rerun. Any earlier report was kept.')
                elif len(content) > 30 * 1024 * 1024:
                    warnings.append('The uploaded PDF exceeds 30 MiB; it was not used. Compress its figures and rerun. Any earlier report was kept.')
                else:
                    (saved / 'report.pdf').write_bytes(content)
                    print(f'Using {name} as the shared report.pdf.')
    output = build_submission(saved, report='report.pdf' if include_report else None,
                              additional_warnings=warnings)
    files.download(str(output))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--report', default='report.pdf')
    parser.add_argument('--no-report', action='store_true')
    parser.add_argument('--out', default='submission.zip')
    args = vars(parser.parse_args())
    if args.pop('no_report'):
        args['report'] = None
    build_submission(**args)
