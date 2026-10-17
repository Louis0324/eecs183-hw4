# Frozen TTS inference code

The CosyVoice and Matcha-TTS source snapshots come from the original HW4
`hw4_util.zip`, downloaded on 2026-10-16. They are kept locally so setup does not
depend on cloning moving repositories or initializing Git submodules.

- CosyVoice: https://github.com/FunAudioLLM/CosyVoice (Apache 2.0; bundled LICENSE).
- Matcha-TTS: https://github.com/shivammehta25/Matcha-TTS (MIT; bundled LICENSE).
- The nested HiFi-GAN license is preserved.

HW4 changes: removed unused training-model imports from CosyVoice's class registry;
removed eager training utilities from Matcha's utils initializer; replaced Matcha's
Lightning-specific logger wrapper with Python logging for single-process inference.
Model layers and checkpoint shapes are unchanged. The original copyright and
license notices remain in the source files. `hw4_util.py` loads checkpoints strictly
and reports errors instead of substituting random audio.
