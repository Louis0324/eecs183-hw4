"""Download only the HW4 course assets needed by the current notebook."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILES = {'a': ('hw4-data-a.zip', 'probes'),
              'b': ('hw4-data-b.zip', 'asr'),
              'c': ('hw4-data-c.zip', 'tts'),
              'speakers': ('hw4-speakers.zip', 'speakers')}


def asset_spec(filename):
    if filename == 'hw4-tts-starter.pt':
        spec = json.loads((ROOT / 'tts_starter_manifest.json').read_text())
    else:
        releases = json.loads((ROOT / 'data_manifest.json').read_text())['releases']
        if filename not in releases:
            raise ValueError(f'Unknown course asset: {filename}')
        spec = releases[filename]
    return {key: spec[key] for key in ['bytes', 'sha256']}


def verify_asset(path, spec):
    path = Path(path)
    if path.stat().st_size != spec['bytes']:
        raise ValueError(f'{path.name}: unexpected size; use the matching course release.')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != spec['sha256']:
        raise ValueError(f'{path.name}: checksum mismatch; download a fresh copy of the matching release.')
    return path


def download_course_file(filename, local_path=None):
    """Use a verified local copy, otherwise download anonymously into the HF cache."""
    spec = asset_spec(filename)
    if local_path is not None:
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(f'Local course file not found: {path}')
        return verify_asset(path, spec)
    if Path(filename).is_file():
        return verify_asset(Path(filename), spec)

    settings = json.loads((ROOT / 'course_release.json').read_text())
    repo_id = os.environ.get('HW4_HF_REPO') or settings['repo_id']
    revision = os.environ.get('HW4_HF_REVISION') or settings['revision']
    if not repo_id:
        raise FileNotFoundError(f'The course Hugging Face repository is not configured yet. '
                                f'For local testing, supply {filename} using local_path or the Files sidebar.')
    from huggingface_hub import hf_hub_download
    print(f'Course file: {filename} ({spec["bytes"] / 1024**2:.1f} MiB); cached copies are reused.')
    for force_download in [False, True]:
        path = hf_hub_download(repo_id=repo_id, repo_type='dataset', revision=revision,
                               filename=filename, token=False, force_download=force_download)
        try:
            return verify_asset(path, spec)
        except ValueError:
            if force_download:
                raise
            print('Cached file failed verification; downloading a fresh copy.')


def ensure_course_data(part, data_root='data', archive=None):
    """Verify/extract one dataset; mark it complete only after successful extraction."""
    if part not in DATA_FILES:
        raise ValueError('part must be a, b, c, or speakers')
    filename, directory = DATA_FILES[part]
    root = Path(data_root)
    spec = asset_spec(filename)
    marker = root / f'.{filename}.complete'
    destination = root / directory
    if marker.is_file() and marker.read_text().strip() == spec['sha256'] and destination.is_dir():
        print('Using prepared data:', destination)
        return destination
    path = download_course_file(filename, local_path=archive)
    from prepare_data import install_release
    install_release(path, root)
    if not destination.is_dir():
        raise ValueError(f'{filename} did not contain the expected data/{directory} directory.')
    marker.write_text(spec['sha256'] + '\n')
    print('Prepared data:', destination)
    return destination
