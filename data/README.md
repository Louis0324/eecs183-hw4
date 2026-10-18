# HW4 datasets

The notebooks automatically download these files from the public
[course Hugging Face repository](https://huggingface.co/datasets/Louis0324/eecs183-hw4),
using the fixed `hw4-2026-v4` release. No account or token is needed. Data files are
separate from the small `hw4-student.zip` support bundle; each notebook downloads
only what it uses, checks SHA-256 hashes, and extracts the needed data. Cached
files and successfully extracted datasets are reused when cells are rerun.
To use a local A/B/C data ZIP, set `DATA_ARCHIVE` in that notebook's data cell.

| Part | File | Download | Contents |
|---|---|---:|---|
| A | hw4-data-a.zip | 226.4 MiB | Shared HuBERT features for phoneme and MFCC probing, all 13 layers |
| A | hw4-speakers.zip | 214.6 MiB | Original 20-speaker, 1,000-recording audio dataset |
| B | hw4-data-b.zip | 141.4 MiB | 581 train, 89 validation, 95 test utterances |
| C | hw4-data-c.zip | 118.1 MiB | All 354,780 original TTS training examples and 9,756 validation examples |
| C | hw4-tts-starter.pt | 406.7 MiB | Original architecture's two-epoch weights; loaded automatically for normal training |

A has 8,000/1,754/1,893 train/validation/test frames at each layer. Phone and MFCC
targets share the same features, avoiding duplicate downloads. B has about two
hours of training audio and ten minutes each of validation/test audio. Speakers
are disjoint across these splits. B test transcripts are withheld.

A/B use a fixed subset of the public
[LibriSpeech alignments dataset](https://huggingface.co/datasets/gilkeyio/librispeech-alignments),
revision `0daa1eb43dda38ee6ce752e785555380e5628f5c`. The archive contains attribution,
split IDs, extraction settings, and seed 183. Phoneme labels are forced-aligner
annotations, not manually verified ground truth.

C stores text and speech tokens. It includes the full original training cache;
no training examples were subsampled. Validation excludes the 200 final test
texts (201 matching records).

Additional model downloads: HuBERT about 360 MiB (A speaker analysis and B);
CosyVoice inference components about 1,013 MiB (C; only about 4 MiB of tokenizer
files are needed to start training); Whisper about 1,543 MiB for C's final WER
evaluation. Dependencies also require downloads. These models come from their
original Hugging Face repositories at fixed revisions.
A fresh Colab runtime may need to download its files again; optionally keep your
own cache on Drive across sessions. A's probing tasks use cached features and need no raw audio.
