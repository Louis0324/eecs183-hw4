"""Provided loading checks for the course's partially trained TTS model."""
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_signature(text_tokenizer):
    """Include token-to-ID assignments, including the added speech/text tags."""
    tokenizer = text_tokenizer.tokenizer
    vocabulary = json.dumps(tokenizer.get_vocab(), sort_keys=True,
                            ensure_ascii=False, separators=(',', ':'))
    return dict(vocab_size=text_tokenizer.vocab_size,
                vocabulary_sha256=hashlib.sha256(vocabulary.encode()).hexdigest(),
                pad_id=text_tokenizer.pad_id, eos_id=text_tokenizer.eos_id)


def describe_architecture(model):
    """Check structural choices that state-dict shapes alone cannot distinguish."""
    layers = list(model.transformer.layers)
    if not layers:
        raise ValueError('The TTS model needs Transformer layers.')
    first = layers[0]
    if any(layer.activation is not F.relu for layer in layers):
        raise ValueError('Use the supplied TTS constructor: activation must be ReLU.')
    settings = [(layer.self_attn.num_heads, layer.linear1.out_features,
                 layer.dropout.p, layer.dropout1.p, layer.dropout2.p,
                 layer.self_attn.dropout, layer.norm_first, layer.norm1.eps,
                 layer.norm2.eps, layer.self_attn.batch_first,
                 layer.linear1.bias is not None, layer.linear2.bias is not None,
                 layer.self_attn.in_proj_bias is not None) for layer in layers]
    if any(setting != settings[0] for setting in settings):
        raise ValueError('Use identical settings for every Transformer layer.')
    return dict(family='hw4-tts-transformer-encoder-v1',
                text_vocab_size=model.text_embedding.num_embeddings,
                speech_vocab_size=model.speech_vocab_size,
                speech_embedding_size=model.speech_embedding.num_embeddings,
                special_vocab_size=model.special_embedding.num_embeddings,
                output_size=model.output_projection.out_features,
                d_model=model.text_embedding.embedding_dim,
                n_heads=first.self_attn.num_heads, n_layers=len(layers),
                max_seq_len=model.position_embedding.num_embeddings,
                dim_feedforward=first.linear1.out_features,
                dropout_settings=[first.dropout.p, first.dropout1.p,
                                  first.dropout2.p, first.self_attn.dropout],
                activation='relu', norm_first=first.norm_first,
                layer_norm_eps=[first.norm1.eps, first.norm2.eps],
                bias_settings=[first.linear1.bias is not None, first.linear2.bias is not None,
                               first.self_attn.in_proj_bias is not None],
                batch_first=first.self_attn.batch_first,
                final_norm=model.transformer.norm is not None)


def load_tts_starter(model, text_tokenizer, checkpoint_path, manifest_path=None):
    """Load verified weights into an already constructed model, keeping its dtype.

    This initializes weights only. Create a fresh optimizer/scheduler afterward;
    use the separate RESUME path for your own interrupted training checkpoints.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f'Upload the course-provided {path.name}, or set USE_STARTER=False.')
    manifest_path = Path(manifest_path) if manifest_path is not None else ROOT / 'tts_starter_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if path.stat().st_size != manifest['bytes'] or file_sha256(path) != manifest['sha256']:
        raise ValueError('Starter checkpoint checksum mismatch. Use the matching course checkpoint and support ZIP.')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format_version') != 1 or state.get('kind') != 'hw4_tts_starter':
        raise ValueError('Expected the course weights-only TTS starter checkpoint.')
    architecture = describe_architecture(model)
    if architecture != state.get('architecture') or architecture != manifest['architecture']:
        raise ValueError('Starter architecture mismatch. Keep the supplied model constructor and original MODEL_KWARGS.')
    signature = tokenizer_signature(text_tokenizer)
    if signature != state.get('text_tokenizer') or signature != manifest['text_tokenizer']:
        raise ValueError('Text tokenizer IDs do not match this checkpoint. Use the pinned course tokenizer.')
    current, supplied = model.state_dict(), state['model_state_dict']
    missing = sorted(set(current) - set(supplied))
    unexpected = sorted(set(supplied) - set(current))
    wrong_shapes = [key for key in set(current) & set(supplied)
                    if current[key].shape != supplied[key].shape]
    if missing or unexpected or wrong_shapes:
        raise ValueError(f'Keep the supplied constructor and module names. Missing={missing}; '
                         f'unexpected={unexpected}; wrong shapes={sorted(wrong_shapes)}')
    # Do not use strict=False: silently skipped weights undermine the warm start.
    model.load_state_dict(supplied, strict=True)
    metadata = {key: value for key, value in state.items() if key != 'model_state_dict'}
    # Evaluation happens after export; update its metadata without rewriting the
    # large checkpoint or changing the verified weights' checksum.
    metadata['evaluation_status'] = manifest.get('evaluation_status',
                                                metadata.get('evaluation_status', 'pending'))
    if 'evaluation' in manifest:
        metadata['evaluation'] = manifest['evaluation']
    return metadata
