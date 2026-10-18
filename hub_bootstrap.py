"""Small, dependency-free bootstrap embedded in the distributed notebooks."""


def ensure_hw4_support(repo_id=None, revision='hw4-2026-v4', packager_only=False):
    """Fetch the public support bundle, or use a local/uploaded offline copy."""
    import hashlib
    from io import BytesIO
    import json
    from pathlib import Path
    import re
    from urllib.parse import quote
    from urllib.request import urlopen
    import zipfile

    required = 'make_submission.py' if packager_only else 'course_download.py'
    if Path(required).is_file():
        return
    local_bundle = Path('hw4-student.zip')
    if local_bundle.is_file():
        content = local_bundle.read_bytes()
    elif repo_id:
        if not re.fullmatch(r'[\w.-]+/[\w.-]+', repo_id):
            raise ValueError('Expected a Hugging Face repository ID like COURSE_ORG/eecs183-hw4.')
        base = f'https://huggingface.co/datasets/{repo_id}/resolve/{quote(revision, safe="")}/'
        print('Downloading the small HW4 support bundle from Hugging Face; no login is required.')
        try:
            with urlopen(base + 'release_manifest.json', timeout=30) as response:
                manifest = json.loads(response.read(1024 * 1024))
            spec = manifest['files']['hw4-student.zip']
            if not 0 < spec['bytes'] < 30 * 1024 * 1024:
                raise ValueError('Invalid support bundle size in the release manifest.')
            with urlopen(base + 'hw4-student.zip', timeout=30) as response:
                content = response.read(spec['bytes'] + 1)
            if len(content) != spec['bytes'] or hashlib.sha256(content).hexdigest() != spec['sha256']:
                raise ValueError('Support bundle checksum mismatch; rerun to download a fresh copy.')
        except Exception as error:
            raise RuntimeError('Could not download the HW4 support release. Rerun this cell; '
                               'if needed, download hw4-student.zip from the course Hugging Face repo '
                               'and upload it through the Files sidebar before retrying.') from error
    else:
        # Allows local staff testing and offline/manual use before publication.
        try:
            from google.colab import files
        except ImportError as error:
            raise FileNotFoundError('Put hw4-student.zip in this folder or configure the course Hugging Face repository.') from error
        print('Upload hw4-student.zip to load the supplied helpers.')
        uploaded = files.upload()
        archives = [value for name, value in uploaded.items() if name.lower().endswith('.zip')]
        if len(archives) != 1:
            raise ValueError('Upload exactly one hw4-student.zip support bundle, then rerun.')
        content = archives[0]

    with zipfile.ZipFile(BytesIO(content)) as archive:
        if archive.testzip() is not None:
            raise ValueError('Corrupt support ZIP; download a fresh copy.')
        for item in archive.infolist():
            path = Path(item.filename)
            if (path.is_absolute() or '..' in path.parts or
                    ((item.external_attr >> 16) & 0o170000) == 0o120000 or
                    not path.resolve().is_relative_to(Path.cwd().resolve())):
                raise ValueError('Unsafe support archive path.')
            # Browser/local notebook edits must never be replaced by starter copies.
            if item.is_dir() or path.suffix == '.ipynb' or (packager_only and item.filename != required):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(item))
    if not Path(required).is_file():
        raise ValueError(f'This ZIP is missing {required}; use the current hw4-student.zip support bundle.')
