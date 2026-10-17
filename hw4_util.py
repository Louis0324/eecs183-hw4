"""Provided frozen TTS components. Adapted from the 2025 HW4 utility package.

No random-audio fallbacks: dependency/checkpoint errors must be fixed before evaluation.
"""
from pathlib import Path
import sys
import os
import math
import torch
from torch import nn
import numpy as np
import json

_ROOT = Path(__file__).resolve().parent
for _path in [_ROOT / "vendor/CosyVoice", _ROOT / "vendor/CosyVoice/third_party/Matcha-TTS"]:
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def download_pretrained_models(cache_dir='./pretrained_models', tokenizer_only=False):
    from huggingface_hub import snapshot_download
    metadata = json.loads((_ROOT / 'model_manifest.json').read_text())['cosy']
    patterns = ['CosyVoice-BlankEN/*.json', 'CosyVoice-BlankEN/*.txt']
    if not tokenizer_only:
        patterns += ['flow.pt', 'hift.pt', 'speech_tokenizer_v2.onnx', 'campplus.onnx']
    return snapshot_download(metadata['repository'], revision=metadata['revision'],
                             local_dir=str(Path(cache_dir) / 'CosyVoice2-0.5B'), allow_patterns=patterns)


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    def factor(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1., max(0., (step - warmup_steps) / max(1, total_steps - warmup_steps)))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, filepath,
                    model_config=None, starter_metadata=None):
    from hw4_runtime import save_checkpoint as save
    save(filepath, model, optimizer, epoch=epoch, train_loss=float(train_loss), val_loss=float(val_loss),
         scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
         model_config=model_config or getattr(model, 'model_config', None),
         starter_metadata=starter_metadata)


def load_trained_model(checkpoint_path, text_vocab_size, speech_vocab_size, TextToSpeechLM,
                       device='cuda', **model_kwargs):
    model = TextToSpeechLM(text_vocab_size, speech_vocab_size, **model_kwargs).to(device)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    return model.eval()


def record_or_load_voice(prompt_text, results_dir='./results', audio_path=None):
    """Load a short recording; in Colab, upload a WAV/FLAC instead of hosting a UI."""
    from hw4_runtime import load_audio
    if audio_path is None:
        from google.colab import files
        print('Upload a short recording of:', prompt_text)
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError('Upload exactly one recording, or pass audio_path explicitly.')
        name, content = next(iter(uploaded.items()))
        path = Path(results_dir) / Path(name).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        audio_path = path
    audio, _ = load_audio(audio_path, 16000)
    if not 1 <= len(audio) / 16000 <= 15:
        raise ValueError('Use a recording between 1 and 15 seconds.')
    return audio[None], prompt_text, str(audio_path)


def prompt_features(prompt_audio, speech_tokenizer, model_dir, device):
    import torchaudio
    import onnxruntime as ort
    from matcha.utils.audio import mel_spectrogram
    audio = prompt_audio.reshape(1, -1).cpu()
    tokens = speech_tokenizer.encode(audio.flatten(), sr=16000).to(device).reshape(1, -1)
    audio_24k = torchaudio.functional.resample(audio, 16000, 24000).to(device)
    mel = mel_spectrogram(audio_24k, 1920, 80, 24000, 480, 1920, 0, 8000, center=False)
    n = min(tokens.shape[1], mel.shape[-1] // 2)
    if n == 0:
        raise ValueError('Voice prompt is too short after tokenization.')
    tokens, mel = tokens[:, :n], mel[:, :, :n * 2].transpose(1, 2)
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(Path(model_dir) / 'campplus.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    features = torchaudio.compliance.kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
    features -= features.mean(0, keepdim=True)
    embedding = session.run(None, {session.get_inputs()[0].name: features[None].numpy()})[0]
    return tokens, mel, torch.as_tensor(embedding, device=device).reshape(1, 192)


@torch.inference_mode()
def synthesize(text, model, text_tokenizer, speech_tokenizer, flow_model, vocoder,
               device='cuda', prompt_audio=None, prompt_text='', output_path=None,
               model_dir=None, max_length=500):
    import soundfile as sf
    model.eval()
    text_tokens = torch.tensor([text_tokenizer.encode(text)], device=device, dtype=torch.long)
    if prompt_audio is None:
        prompt_tokens = torch.empty(1, 0, device=device, dtype=torch.long)
        prompt_mel = torch.empty(1, 0, 80, device=device)
        embedding = torch.zeros(1, 192, device=device)
    else:
        if model_dir is None:
            model_dir = getattr(speech_tokenizer, 'model_dir', None)
        if model_dir is None:
            raise ValueError('Provide model_dir for the speaker encoder checkpoint.')
        prompt_tokens, prompt_mel, embedding = prompt_features(prompt_audio, speech_tokenizer, model_dir, device)
    prompt_text_tokens = torch.tensor([text_tokenizer.encode(prompt_text) if prompt_text else []], dtype=torch.long, device=device)
    speech = model.generate(text_tokens, max_length=max_length, temperature=1.0, top_k=50,
                            prompt_speech_token=prompt_tokens, prompt_text_tokens=prompt_text_tokens,
                            min_length=min(max_length, int(text_tokens.shape[1]) * 2))
    speech = torch.as_tensor(speech, device=device, dtype=torch.long).reshape(1, -1)
    if speech.numel() == 0 or speech.min() < 0 or speech.max() >= 6561:
        raise ValueError('LM must generate nonempty speech tokens in [0, 6560], excluding EOS.')
    def length(n):
        return torch.tensor([n], device=device, dtype=torch.int32)
    mel, _ = flow_model.inference(token=speech, token_len=length(speech.shape[1]),
        prompt_token=prompt_tokens, prompt_token_len=length(prompt_tokens.shape[1]),
        prompt_feat=prompt_mel, prompt_feat_len=length(prompt_mel.shape[1]), embedding=embedding,
        streaming=False, finalize=True)
    audio, _ = vocoder.inference(speech_feat=mel, cache_source=torch.zeros(1, 1, 0, device=device))
    audio = audio.flatten().cpu()
    if not torch.isfinite(audio).all() or not len(audio):
        raise ValueError('Audio synthesis returned an invalid waveform.')
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio.numpy(), 24000)
    return audio, 24000


def run_voice_cloning_evaluation(student_id, trained_model, text_tokenizer, speech_tokenizer,
        flow_model, vocoder, prompt_audio, prompt_text, fixed_test_set_path=None,
        results_dir='./results', pretrained_dir='./pretrained_models', device='cuda',
        max_examples=None, model_dir=None, restart=False):
    """Write each completed prediction immediately; interrupted runs can resume."""
    import librosa
    from transformers import pipeline
    texts = json.loads((_ROOT / 'data/fixed_test_texts.json').read_text())
    path = Path(results_dir) / 'tts_submission.json'; path.parent.mkdir(parents=True, exist_ok=True)
    import hashlib
    from hw4_runtime import model_fingerprint
    voice = torch.as_tensor(prompt_audio).detach().cpu().float().contiguous().numpy()
    signature = hashlib.sha256((model_fingerprint(trained_model) + prompt_text + json.dumps(texts)).encode())
    signature.update(memoryview(voice).cast('B'))
    fingerprint = signature.hexdigest()
    payload = {'format_version': 1, 'student_id': str(student_id), 'model_and_prompt': fingerprint, 'transcriptions': {}}
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get('student_id') != str(student_id):
            raise ValueError('Existing TTS predictions belong to another student ID; move that file first.')
        if previous.get('model_and_prompt') == fingerprint and not restart:
            payload = previous
        else:
            path.replace(path.with_suffix('.previous.json'))
            print('Starting a fresh evaluation; previous predictions were archived.')
    spec = json.loads((_ROOT / 'model_manifest.json').read_text())['whisper']
    asr_path = os.environ.get('HW4_WHISPER_PATH', spec['repository'])
    asr = pipeline('automatic-speech-recognition', model=asr_path,
                   revision=None if Path(asr_path).is_dir() else spec['revision'],
                   model_kwargs={'cache_dir': str(Path(pretrained_dir) / 'huggingface')},
                   device=device, torch_dtype=torch.float16 if str(device).startswith('cuda') else torch.float32)
    attempted = 0
    for i, text in enumerate(texts):
        if payload['transcriptions'].get(str(i)):
            continue
        if max_examples is not None and attempted >= max_examples:
            break
        attempted += 1
        try:
            audio, sr = synthesize(text, trained_model, text_tokenizer, speech_tokenizer,
                flow_model, vocoder, device, prompt_audio, prompt_text,
                output_path=str(path.parent / 'tts_audio' / f'{i}.wav'), model_dir=model_dir)
            audio = librosa.resample(audio.numpy(), orig_sr=sr, target_sr=16000)
            result = asr({'array': audio, 'sampling_rate': 16000}, generate_kwargs={'language': 'english', 'task': 'transcribe'})
            payload['transcriptions'][str(i)] = result['text'].strip()
        except Exception as error:
            print(f'Example {i} failed: {error}')
            payload['transcriptions'][str(i)] = ''
        finally:
            temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(payload, indent=2)); temp.replace(path)
        print(f'Saved example {i}; {len(payload["transcriptions"])}/{len(texts)} attempted.')
    return str(path)

def load_text_tokenizer(model_dir: str):
    """
    Load pretrained text tokenizer (Qwen2 BPE)
    
    Uses transformers AutoTokenizer to load the Qwen2 tokenizer directly
    without importing CosyVoice Python modules.
    
    Args:
        model_dir: Path to CosyVoice2-0.5B directory
        
    Returns:
        text_tokenizer: Wrapper around HuggingFace tokenizer
    """
    from transformers import AutoTokenizer
    
    tok_dir = os.path.join(model_dir, "CosyVoice-BlankEN")
    
    # Load using HuggingFace transformers
    hf_tokenizer = AutoTokenizer.from_pretrained(
        tok_dir,
        local_files_only=True,
        trust_remote_code=False
    )
    
    # Add special tokens for CosyVoice2
    special_tokens = {
        'eos_token': '<|endoftext|>',
        'pad_token': '<|endoftext|>',
        'additional_special_tokens': [
            '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
            '[breath]', '<strong>', '</strong>', '[noise]',
            '[laughter]', '[cough]', '[clucking]', '[accent]',
            '[quick_breath]',
            "<laughter>", "</laughter>",
            "[hissing]", "[sigh]", "[vocalized-noise]",
            "[lipsmack]", "[mn]"
        ]
    }
    
    # Only add if not already present
    tokens_to_add = {}
    for key, value in special_tokens.items():
        if key == 'additional_special_tokens':
            # Check which tokens are not already in the tokenizer
            existing = set(hf_tokenizer.additional_special_tokens if hasattr(hf_tokenizer, 'additional_special_tokens') else [])
            new_tokens = [t for t in value if t not in existing]
            if new_tokens:
                tokens_to_add[key] = new_tokens
        elif not hasattr(hf_tokenizer, key) or getattr(hf_tokenizer, key) != value:
            tokens_to_add[key] = value
    
    if tokens_to_add:
        hf_tokenizer.add_special_tokens(tokens_to_add)
    
    # Create wrapper class
    class TextTokenizerWrapper:
        """Wrapper to provide consistent interface"""
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.vocab_size = len(tokenizer)
            self.pad_id = tokenizer.pad_token_id
            self.eos_id = tokenizer.eos_token_id
        
        def encode(self, text: str, **kwargs):
            """Encode text to token IDs"""
            tokens = self.tokenizer([text], return_tensors="pt")
            return tokens["input_ids"][0].cpu().tolist()
        
        def decode(self, tokens, **kwargs):
            """Decode token IDs to text"""
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, dtype=torch.int64)
            return self.tokenizer.batch_decode([tokens], skip_special_tokens=False)[0]
    
    text_tokenizer = TextTokenizerWrapper(hf_tokenizer)
    
    print(f"Text tokenizer loaded (Qwen2 BPE)")
    print(f"  Vocab size: {text_tokenizer.vocab_size}")
    
    return text_tokenizer

def load_speech_tokenizer(model_dir: str, device='cuda'):
    """
    Load pretrained speech tokenizer (VQ-VAE ONNX)
    
    Args:
        model_dir: Path to CosyVoice2-0.5B directory
        device: Device for inference
        
    Returns:
        speech_tokenizer: Frozen pretrained tokenizer
    """
    import onnxruntime
    
    speech_tokenizer_path = os.path.join(model_dir, 'speech_tokenizer_v2.onnx')
    
    # ONNX configuration (same as CosyVoice)
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    
    # Short prompts run quickly on CPU and avoid Colab CUDA/cuDNN/ORT ABI mismatches.
    providers = ["CPUExecutionProvider"]
    
    session = onnxruntime.InferenceSession(
        speech_tokenizer_path,
        sess_options=option,
        providers=providers
    )
    
    class SpeechTokenizer:
        """Black-box speech tokenizer wrapper
        
        IMPORTANT: Speech tokenizer uses Whisper's mel-spectrogram parameters,
        NOT CosyVoice2's flow model parameters!
        """
        def __init__(self, session):
            self.session = session
            self.vocab_size = 6561
            self.eos_id = 6561
            
            # Whisper mel-spectrogram config (for speech tokenizer)
            # These are DIFFERENT from flow model's mel-spec!
            self.sample_rate = 16000  # Whisper uses 16kHz, not 24kHz!
            self.n_fft = 400          # Whisper n_fft
            self.hop_length = 160     # Whisper hop_length
            self.n_mels = 128         # 128 mel bands for speech tokenizer
        
        @torch.no_grad()
        def encode(self, audio: torch.Tensor, sr: int = 24000) -> torch.Tensor:
            """
            Convert audio to speech tokens using Whisper mel-spectrogram
            
            Args:
                audio: [1, T] or [T] audio waveform @ 24kHz (CosyVoice2)
                sr: sample rate
            
            Returns:
                tokens: [T_token] discrete speech tokens
            """
            import whisper
            
            if audio.dim() > 1:
                audio = audio.squeeze(0)  # [T]
            
            # Resample to 16kHz for Whisper
            if sr != 16000:
                import torchaudio
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio.unsqueeze(0) if audio.dim() == 1 else audio).squeeze()
            
            # Use Whisper's log_mel_spectrogram (exactly like CosyVoice)
            mel_spec = whisper.log_mel_spectrogram(audio, n_mels=128)  # [128, T]
            
            # Prepare ONNX inputs: [1, 128, T]
            feats = mel_spec.unsqueeze(0).cpu().numpy()  # [1, 128, T]
            feats_length = np.array([feats.shape[2]], dtype=np.int32)
            
            # Run ONNX model
            tokens = self.session.run(
                ['indices'], 
                {
                    'feats': feats,
                    'feats_length': feats_length
                }
            )[0]
            
            # tokens shape: [1, T_token] -> [T_token]
            return torch.tensor(tokens, dtype=torch.long).flatten()
    
    speech_tokenizer = SpeechTokenizer(session)
    speech_tokenizer.model_dir = str(model_dir)
    
    # Speech tokenizer loaded
    
    return speech_tokenizer

def load_flow_model(model_dir: str, device='cuda'):
    """
    Load pretrained Flow Matching model from CosyVoice2
    
    Args:
        model_dir: Path to CosyVoice2-0.5B directory
        device: Device for inference
        
    Returns:
        flow_model: Frozen pretrained flow model
    """
    import sys
    # Add CosyVoice to path temporarily for importing models
    # Try multiple possible locations
    possible_paths = [
        str(_ROOT / 'vendor/CosyVoice'),  # Direct path from hw4
        os.path.join(os.path.dirname(__file__), 'CosyVoice'),  # Relative to this file
    ]
    
    cosyvoice_path = None
    for path in possible_paths:
        if os.path.exists(path):
            cosyvoice_path = path
            break
    
    if cosyvoice_path and cosyvoice_path not in sys.path:
        sys.path.insert(0, cosyvoice_path)
    
    # Also add Matcha-TTS to path (required by flow_matching.py)
    matcha_path = os.path.join(cosyvoice_path, 'third_party/Matcha-TTS') if cosyvoice_path else None
    if matcha_path and os.path.exists(matcha_path) and matcha_path not in sys.path:
        sys.path.insert(0, matcha_path)
    
    # Import CosyVoice2 flow model components
    from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
    from cosyvoice.transformer.upsample_encoder import UpsampleConformerEncoder
    from cosyvoice.flow.flow_matching import CausalConditionalCFM
    from cosyvoice.flow.decoder import CausalConditionalDecoder
    from omegaconf import DictConfig
        
    # Create encoder with CosyVoice2 config
    encoder = UpsampleConformerEncoder(
        output_size=512,
        attention_heads=8,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        normalize_before=True,
        input_layer='linear',
        pos_enc_layer_type='rel_pos_espnet',
        selfattention_layer_type='rel_selfattn',
        input_size=512,
        use_cnn_module=False,
        macaron_style=False,
        static_chunk_size=25
    )
    
    # Create estimator (the actual neural network for flow matching)
    estimator = CausalConditionalDecoder(
        in_channels=320,  # 240 (cond) + 80 (mel)
        out_channels=80,
        channels=[256],
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=12,
        num_heads=8,
        act_fn="gelu"
    )
    
    # Create decoder (CFM) config
    cfm_params = DictConfig({
        'sigma_min': 1e-06,
        'solver': 'euler',
        't_scheduler': 'cosine',
        'training_cfg_rate': 0.2,
        'inference_cfg_rate': 0.7,
        'reg_loss_type': 'l1'
    })
    
    # Create decoder with estimator
    decoder = CausalConditionalCFM(
        in_channels=240,
        n_spks=1,
        spk_emb_dim=80,
        cfm_params=cfm_params,
        estimator=estimator  # Pass the estimator!
    )
    
    # Initialize flow model with proper config
    flow_model = CausalMaskedDiffWithXvec(
        input_size=512,
        output_size=80,
        spk_embed_dim=192,
        output_type='mel',
        vocab_size=6561,
        input_frame_rate=25,
        only_mask_loss=True,
        token_mel_ratio=2,
        pre_lookahead_len=3,
        encoder=encoder,
        decoder=decoder
    )
    
    # Load pretrained weights
    flow_path = os.path.join(model_dir, 'flow.pt')
    flow_state_dict = torch.load(flow_path, map_location='cpu', weights_only=True)
    flow_model.load_state_dict(flow_state_dict, strict=True)  # Use strict=True for compatibility
    
    # Move to device and set to eval mode
    flow_model = flow_model.to(device).eval()
    
    # Freeze all parameters
    for param in flow_model.parameters():
        param.requires_grad = False
    
    # Flow model loaded
    print(f"  - Checkpoint: flow.pt")
    print(f"  - Device: {device}")
    
    return flow_model

def load_vocoder(model_dir: str, device='cuda'):
    """
    Load pretrained HiFi-GAN vocoder from CosyVoice2
    
    Args:
        model_dir: Path to CosyVoice2-0.5B directory
        device: Device for inference
        
    Returns:
        vocoder: Frozen pretrained vocoder
    """
    import sys
    # Add CosyVoice to path temporarily for importing models
    # Try multiple possible locations
    possible_paths = [
        str(_ROOT / 'vendor/CosyVoice'),  # Direct path from hw4
        os.path.join(os.path.dirname(__file__), 'CosyVoice'),  # Relative to this file
    ]
    
    cosyvoice_path = None
    for path in possible_paths:
        if os.path.exists(path):
            cosyvoice_path = path
            break
    
    if cosyvoice_path and cosyvoice_path not in sys.path:
        sys.path.insert(0, cosyvoice_path)
    
    try:
        # Import CosyVoice2 vocoder and F0Predictor
        from cosyvoice.hifigan.generator import HiFTGenerator
        from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor
        
        # Create F0 predictor
        f0_predictor = ConvRNNF0Predictor(
            num_class=1,
            in_channels=80,
            cond_channels=512
        )
        
        # Initialize vocoder with CosyVoice2 config
        vocoder = HiFTGenerator(
            in_channels=80,
            base_channels=512,
            nb_harmonics=8,
            sampling_rate=24000,
            nsf_alpha=0.1,
            nsf_sigma=0.003,
            nsf_voiced_threshold=10,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            istft_params={'n_fft': 16, 'hop_len': 4},
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            lrelu_slope=0.1,
            audio_limit=0.99,
            f0_predictor=f0_predictor  # Add the f0_predictor
        )
        
        # Load pretrained weights
        vocoder_path = os.path.join(model_dir, 'hift.pt')
        hift_checkpoint = torch.load(vocoder_path, map_location='cpu', weights_only=True)
        # Remove 'generator.' prefix from keys if present
        hift_state_dict = {k.replace('generator.', ''): v for k, v in hift_checkpoint.items()}
        vocoder.load_state_dict(hift_state_dict, strict=True)  # Use strict=True for compatibility with f0_predictor weights
        
        # Move to device and set to eval mode
        vocoder = vocoder.to(device).eval()
        
        # Freeze all parameters
        for param in vocoder.parameters():
            param.requires_grad = False
        
        # Vocoder loaded
        print(f"  - Checkpoint: hift.pt")
        print(f"  - Device: {device}")
        
        return vocoder
        
    except ImportError as e:
        raise RuntimeError("Install requirements-tts.txt before loading the vocoder") from e


def generate_audio_from_tokens(speech_tokens, flow_model, vocoder, device):
    """
    Helper function: Convert speech tokens to audio using pretrained flow model and vocoder
    
    Args:
        speech_tokens: list or tensor of speech token IDs
        flow_model: Pretrained flow model
        vocoder: Pretrained vocoder
        device: torch device
        
    Returns:
        audio: [T_audio] tensor
        mel: [1, 80, T_mel] tensor
    """
    import torch
    
    # Convert to tensor if needed
    if isinstance(speech_tokens, list):
        speech_token_tensor = torch.tensor([speech_tokens], dtype=torch.long).to(device)
    else:
        # Already tensor, ensure proper shape [1, T]
        if speech_tokens.dim() == 1:
            speech_token_tensor = speech_tokens.unsqueeze(0).to(device)
        else:
            speech_token_tensor = speech_tokens.to(device)
    
    speech_token_len = torch.tensor([speech_token_tensor.shape[1]], dtype=torch.int32).to(device)
    
    with torch.no_grad():
        # No prompt mode - use empty tensors
        prompt_token = torch.zeros(1, 0, dtype=torch.long).to(device)
        prompt_token_len = torch.tensor([0], dtype=torch.int32).to(device)
        prompt_feat = torch.zeros(1, 0, 80, dtype=torch.float32).to(device)
        prompt_feat_len = torch.tensor([0], dtype=torch.int32).to(device)
        
        # Neutral zero speaker embedding (192-dim for CosyVoice2).
        embedding = torch.zeros(1, 192, device=device)
        
        # Flow model: tokens → mel
        mel, _ = flow_model.inference(
            token=speech_token_tensor,
            token_len=speech_token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            streaming=False,
            finalize=True
        )
        
        # Vocoder: mel → audio
        audio, _ = vocoder.inference(mel)
        
        # Squeeze to [T_audio]
        while audio.dim() > 1:
            audio = audio.squeeze(0)
    
    return audio, mel
