"""Download course data, or install a replacement archive supplied by staff."""
import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def install_release(archive, data_root='data'):
    """Install a course release, including the original speaker archive layout."""
    destination = Path(data_root)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            p = Path(info.filename)
            if p.is_absolute() or '..' in p.parts or ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Unsafe release archive path')
            if not p.parts or info.is_dir():
                continue
            if p.parts[0] == 'data':
                target = destination.joinpath(*p.parts[1:])
            elif p.parts[0] == 'lsmini-20speakers-50recordings':
                # Preserve the original 1,000 recordings; normalize only their
                # archive prefix to the directory expected by Part A.
                target = destination.joinpath('speakers', *p.parts[1:])
            else:
                continue
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Release path leaves the data directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)
    return destination


def prepare_data(part, archive=None, data_root=None):
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    entry = manifest[part]
    destination = Path(data_root or ROOT / "data") / entry["destination"]
    if (destination / ".complete").exists():
        return destination
    if archive is None and "replacement required" in entry["status"]:
        raise FileNotFoundError(
            f"The original {part} archive is unavailable. Ask course staff for its replacement; "
            f"then run prepare_data('{part}', archive='/path/to/replacement.zip'). "
            "Do not substitute smoke-test data for the graded experiments."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        temporary = Path(temporary)
        if archive is None:
            import gdown
            archive = temporary / "download.zip"
            if not gdown.download(id=entry["id"], output=str(archive), quiet=False):
                raise RuntimeError(f"Download failed for {part}; ask staff for the archive.")
        with zipfile.ZipFile(archive) as z:
            members = []
            for info in z.infolist():
                p = Path(info.filename)
                if p.is_absolute() or ".." in p.parts or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive contains an unsafe path")
                if any(x.startswith('.') or x in {'__MACOSX', '__pycache__'} for x in p.parts):
                    continue
                # Course code comes from this repository, not an old archive.
                if part == "tts" and "libritts_token_cache" not in p.parts:
                    continue
                members.append(info)
            z.extractall(temporary / "unpacked", members)
        source = temporary / "unpacked"
        if part == "tts":
            matches = list(source.rglob("libritts_train.pt"))
            if len(matches) != 1:
                raise ValueError("Expected one libritts_train.pt in the TTS archive")
            source = matches[0].parent
        elif part == "asr":
            matches = list(source.rglob("train_transcriptions.txt"))
            if len(matches) != 1:
                raise ValueError("Expected train_transcriptions.txt in the ASR archive")
            source = matches[0].parent
        elif part in {"phoneme", "mfcc"}:
            matches = list(source.rglob("layer_0"))
            if len(matches) != 1:
                raise ValueError("Expected one layer_0 directory in the probe archive")
            source = matches[0].parent
        shutil.copytree(source, destination, dirs_exist_ok=True)
        (destination / ".complete").write_text("Archive extracted successfully.\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=["phoneme", "mfcc", "asr", "speakers", "tts"])
    parser.add_argument("--archive")
    args = parser.parse_args()
    print(prepare_data(args.part, args.archive))
