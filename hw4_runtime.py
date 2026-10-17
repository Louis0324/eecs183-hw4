"""Provided I/O and training helpers, shared by the notebooks and smoke tests."""
import json
import os
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def load_hubert():
    from transformers import HubertModel
    spec = json.loads((Path(__file__).resolve().parent / 'model_manifest.json').read_text())['hubert']
    source = os.environ.get('HW4_HUBERT_PATH', spec['repository'])
    return HubertModel.from_pretrained(source, revision=None if Path(source).is_dir() else spec['revision'])


def load_audio(path, sample_rate=16000):
    import soundfile as sf
    import librosa
    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f'Empty or nonfinite audio: {path}')
    return torch.from_numpy(np.ascontiguousarray(audio)), sample_rate


def load_transcription(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            tag, text = line.split('|', 1)
            if tag in result:
                raise ValueError(f'Duplicate audio ID {tag}')
            result[tag] = text.strip()
    return result


class CharacterTokenizer:
    def __init__(self):
        self.vocab = ['blank', 'pad'] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")
        self.ch2idx = {x: i for i, x in enumerate(self.vocab)}
        self.pad_id = 1
        self.blank = 0

    def encode(self, text):
        return np.array([self.ch2idx[c] for c in text.upper() if c in self.ch2idx], dtype=np.int64)

    def encode_torch_batch(self, texts):
        tokens = [torch.from_numpy(self.encode(t)) for t in texts]
        return nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.pad_id), torch.tensor([len(t) for t in tokens])

    enocode_torch_batch = encode_torch_batch  # Compatibility with last year's typo.

    def decode(self, ids):
        return ''.join(self.vocab[int(i)] for i in ids if int(i) not in (0, 1))

    def blank_id(self):
        return self.blank


class SpeechTextDataset(Dataset):
    def __init__(self, DATA_DIR, split='train'):
        if split not in {'train', 'dev', 'test'}:
            raise ValueError('split must be train, dev, or test')
        root = Path(DATA_DIR)
        self.tag2text = load_transcription(root / f'{split}_transcriptions.txt') if split != 'test' else {}
        self.wav_files = sorted(p for p in (root / split).rglob('*') if p.suffix.lower() in {'.wav', '.flac'})
        if not self.wav_files:
            raise FileNotFoundError(f'No audio in {root / split}; run the data setup cell.')
        self.split = split

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, i):
        p = self.wav_files[i]
        wav, _ = load_audio(p)
        wav = (wav - wav.mean()) / wav.std(unbiased=False).clamp_min(1e-7)
        return {'wav': wav, 'text': self.tag2text[p.stem] if self.split != 'test' else '', 'id': p.stem}

    @staticmethod
    def collate(batch):
        return dict(input_values=nn.utils.rnn.pad_sequence([x['wav'] for x in batch], batch_first=True),
                    attention_mask=nn.utils.rnn.pad_sequence([torch.ones(len(x['wav']), dtype=torch.long) for x in batch], batch_first=True),
                    text=[x['text'] for x in batch], ids=[x['id'] for x in batch])


def load_probe_data(base_folder, layer_idx, regression=False):
    root = Path(base_folder) / f'layer_{layer_idx}'
    features, labels = [], []
    key = 'mfcc' if regression else 'labels'
    suffix = 'mfcc_labels' if regression else 'labels'
    for split in ['train', 'dev', 'test']:
        x = torch.load(root / f'{split}_features.pt', map_location='cpu', weights_only=True).float()
        y = json.loads((root / f'{split}_{suffix}.json').read_text())[key]
        if x.ndim != 2 or len(x) != len(y) or not torch.isfinite(x).all():
            raise ValueError(f'Invalid features/labels for {split}, layer {layer_idx}')
        features.append(x)
        labels.append(y)
    if regression:
        labels = [torch.tensor(y, dtype=torch.float32) for y in labels]
    else:
        # A fixed label inventory shared across splits, never a per-split remapping.
        vocab = {v: i for i, v in enumerate(sorted({v for split in labels for v in split}))}
        labels = [torch.tensor([vocab[v] for v in y], dtype=torch.long) for y in labels]
    return tuple(item for pair in zip(features, labels) for item in pair)


def load_phoneme_data(base_folder, layer_idx):
    return load_probe_data(base_folder, layer_idx)


def load_mfcc_data(base_folder, layer_idx):
    return load_probe_data(base_folder, layer_idx, regression=True)


def save_checkpoint(path, model, optimizer=None, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(model_state_dict=model.state_dict(), **metadata)
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def make_asr_optimizer(model, upstream_lr=1e-5, head_lr=1e-3):
    """Use separate learning rates for pretrained HuBERT and the new CTC layers."""
    upstream, head = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (upstream if name.startswith('upstream.') else head).append(parameter)
    return torch.optim.AdamW([{'params': upstream, 'lr': upstream_lr},
                             {'params': head, 'lr': head_lr}], weight_decay=0.01)


def model_fingerprint(model):
    """Identify weights so prediction resumption cannot mix different models."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        value = value.detach().cpu().contiguous()
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def train_asr_epoch(model, loader, optimizer, device, max_steps=None):
    model.train()
    losses = []
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch['input_values'].to(device), batch['text'], batch['attention_mask'].to(device))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite CTC loss; check lengths, log_softmax, and targets.')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_asr(model, loader, device, max_steps=None):
    from jiwer import wer
    model.eval()
    refs, hyps = [], []
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        predictions, _ = model.predict(batch['input_values'].to(device), batch['attention_mask'].to(device))
        refs.extend(batch['text'])
        hyps.extend(predictions)
    return float(wer(refs, hyps)) if refs else 1.0
