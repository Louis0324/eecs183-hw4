# EECS 183/283A Homework 4: Speech Processing

Complete the TODOs in `hw4-a.ipynb`, `hw4-b.ipynb`, and `hw4-c.ipynb`. Use Google
Colab with a GPU runtime. The notebooks explain the interfaces, tensor shapes,
experiments, and report questions. You do not need all of LibriSpeech or LibriTTS audio.

| Notebook | Open in Google Colab |
|---|---|
| [A: Probing HuBERT](hw4-a.ipynb) | [Open A](https://colab.research.google.com/github/Louis0324/eecs183-hw4/blob/main/hw4-a.ipynb) |
| [B: ASR](hw4-b.ipynb) | [Open B](https://colab.research.google.com/github/Louis0324/eecs183-hw4/blob/main/hw4-b.ipynb) |
| [C: TTS](hw4-c.ipynb) | [Open C](https://colab.research.google.com/github/Louis0324/eecs183-hw4/blob/main/hw4-c.ipynb) |

Save your own editable copy with **File → Save a copy in Drive** before working.
The course materials download from Hugging Face; your personal Drive is only for
saving your work and optional training checkpoints.

## Start on Colab

1. Open the notebook and select **Runtime → Change runtime type → GPU**.
2. Run setup. It automatically downloads the small **hw4-student.zip** support
   bundle from the public [course Hugging Face repository](https://huggingface.co/datasets/Louis0324/eecs183-hw4).
   No Hugging Face account, token, or TA Drive access is needed. Setup preserves
   Colab's matched PyTorch/torchaudio installation and installs other dependencies.
3. Run the data cell. It downloads **only that part's dataset**, verifies its
   checksum, and extracts it. A/B/C are about **226/141/118 MiB**. A's speaker
   recordings download separately when needed (**215 MiB**). See
   [data/README.md](data/README.md) for counts and additional model downloads.
4. Implement the TODOs, then use `SMOKE_TEST=True` to check your implementation.
   Set it back to `False` for experiments. A fresh starter intentionally stops at
   `NotImplementedError`; the staff smoke test is a separate notebook.
5. Save checkpoints and results to Drive or download them regularly. Colab local
   storage is temporary. B and C include a `RESUME` switch for saved checkpoints.
   After mounting Drive, you can set `RESULTS` in setup to a Drive directory such
   as `/content/drive/MyDrive/hw4/results`. The submission cell also collects
   predictions and plots from that directory.

For C, the model-loading cell automatically downloads **hw4-tts-starter.pt**
(about **407 MiB**) from the same public repository. It contains only the original
TTS model's weights after two reference epochs. The exact model constructor is supplied for compatibility; implement
sequence construction, masking, loss, generation, and training yourself. The
default warm start uses one additional student epoch, with the full original
dataset still available. `SMOKE_TEST=True` does not need the starter download.
The staff implementation measured 23.55% WER with this starter and 14.56% after
three reference epochs, using the same 200 texts and voice prompt. Further
training can improve quality; these measurements do not guarantee the outcome
of a student run. Performance thresholds are listed below.

Course downloads use the fixed `hw4-2026-v2` release. Data/checkpoints are checked
against SHA-256 hashes; downloads are cached within the runtime, and rerunning a
completed data cell reuses its extracted files. After Colab deletes a runtime,
its local cache is lost and those files must download again. No full repository
clone is needed. An optional offline copy can be supplied through `DATA_ARCHIVE`;
for C, `STARTER_CHECKPOINT` can point to an existing local checkpoint file.

The three notebooks may use separate Colab sessions. A's phone/MFCC tensors are
precomputed, B uses about two hours of training audio, and C retains the entire
original **354,780-example tokenized training cache**.

## Grading

| Part | Implementation | Model performance | Report | Total |
|---|---:|---:|---:|---:|
| A: probing and speaker embeddings | 3 | 12 | 9 | 24 |
| B: CTC ASR | 12 | 18 | 3 | 33 |
| C: speech-token TTS | 16 | 24 | 3 | 43 |

The automated component is **85 points**. Staff grade only the shared report PDF
for the remaining **15 points**, plus up to **10 bonus points** for the personal
speaker experiment. Correct implementations earn credit even if a model has not
converged. Performance carries at least half of each notebook total, including
report points. A awards 6 points each for phoneme accuracy and MFCC R², recomputed
from saved predictions over all 13 layers; its notebook lists per-layer targets
and the linear partial-credit formula. Missing/invalid layers affect only their
own share. Speaker visualization retains code/report credit without adding a
classifier experiment. ASR full performance credit permits WER ≤50%, compared with a 28.9%
reference using the revised ASR settings. Inclusive ASR cutoffs
50/60/70/80/90/99% award 18/15/12/9/6/3 points. TTS full performance credit requires
WER **strictly below 15%**, compared with 14.56% for the original
12-layer/768-dimensional/12-head reference. Its strict upper cutoffs of
15/17/19/21/22.5/23.5% award 24/20/16/12/8/4 points; WER ≥23.5% earns zero performance
points. This gives the two-epoch starter (23.55%) zero performance points, including
the rounded 24% baseline. The 16 TTS implementation points are graded independently.

For TTS, the exact model constructor, checkpoint loader, optimizer defaults,
scheduler, and outer epoch/checkpoint loop are supplied. Students implement the
dataset/collator, sequence construction, causal attention, loss, generation, and
the per-batch training and validation loops. AdamW starts at learning rate 2e-4
and weight decay 0.01, with 10% warmup followed by cosine decay. Use those defaults
as a starting point; tuning is optional and should use validation results. Keep
the supplied model constructor unchanged to preserve checkpoint compatibility.

Implementation checks use small offline examples, not your trained checkpoint.
They check shapes, masking, gradients, losses, decoding, and training behavior.
Keep the required names and signatures. Put extra imports inside your helper
functions: the grader extracts function/class definitions without running setup,
downloads, training, or other top-level notebook code.

Use [report_template.txt](report_template.txt) for one **report.pdf** with exactly
five required sections, worth 3 points each:

- **A1:** phoneme-accuracy plot and interpretation.
- **A2:** MFCC R² plot and comparison with phoneme probing.
- **A3:** speaker t-SNE panels and interpretation.
- **B1:** one short CTC decoding/valid-length/checkpoint-selection question.
- **C1:** one short teacher-forcing/masking/WER question.

Each notebook begins with a completion checklist showing what to implement,
what scaffolding is provided, what to run, and which small files to save.
The notebooks give the exact report prompts. Put every conceptual answer in the PDF;
notebook markdown answers are not manually graded. Use the fixed section IDs,
matching figure captions, and selectable text (not screenshots of written answers).
About 3–5 sentences per section are enough, plus the requested figures/sequences;
B1/C1 need no extra training or experiments. Include training settings once in the
template's brief experiment summary. Write `Not completed` for unfinished work.
Report credit rewards explanations and available evidence, not a high model score.
The optional **BONUS** section covers the existing own-voice experiment.

## Submit to Gradescope

Run the final cell to capture your current notebook edits, include available
predictions/plots, strip large notebook outputs from the submitted copy, and
download `submission.zip`. This works even if some questions are unfinished.

**Default: submit one combined ZIP containing one shared `report.pdf`.**
The PDF covers **A1, A2, A3, B1, C1** (and BONUS if attempted). Use
`report_template.txt`, keep all five headings, and write `Not completed` only for
answers you did not finish. You may answer B1/C1 without training those models.
Export one PDF with selectable text from Google Docs, Word, or LaTeX. If you wrote
separate reports, combine them into this document before uploading; the helper
**does not merge PDFs**. Notebook PDFs are not required.

**Typical A → B → C workflow**

1. **After A:** run its final cell. Download `submission.zip` and rename it
   `hw4-a.zip` on your computer. At the report prompt, upload your current combined
   PDF or cancel if it is not ready.
2. **After B:** leave `include_previous=True` (B's default). At the **earlier ZIP**
   prompt, upload `hw4-a.zip`. At the separate **report** prompt, upload an updated
   combined PDF, or cancel to keep the report already in the ZIP. The cell adds
   your current B notebook and available B predictions. Save the download as
   `hw4-ab.zip`.
3. **After C:** leave `include_previous=True` (C's default). Upload `hw4-ab.zip`
   at the earlier-ZIP prompt, then your final combined `report.pdf` at the report
   prompt. The cell adds C and downloads the final `submission.zip`. Upload
   **that one ZIP** to the HW4 Programming Assignment on Gradescope.

**Stopping early or working in another order**

| Work you want graded | What to do |
|---|---|
| A + B only | Finish step 2 with your final report, then submit `hw4-ab.zip`. You do **not** need to open or run C. |
| A only, or part of A | Submit A's ZIP with the report answers you completed. |
| Only B or C | Cancel the earlier-ZIP prompt; the cell still packages the current notebook and available results. |
| Separate A and B ZIPs, finishing C | Select both A and B ZIPs together at C's earlier-ZIP prompt, then upload the shared report at the next prompt. |
| Another order, or updating earlier work | Set `include_previous=True` in whichever notebook you package last and import your latest combined ZIP. Current notebook edits and available local results replace their earlier copies. |

Use the **latest** ZIP for each part; do not select both an old and a new version
of the same part. Keep `include_report=True`: cancelling its upload preserves an
existing report; setting it to `False` deliberately leaves the report out.
Missing code, predictions, or a PDF do not block grading of other completed work.
Missing C does not reduce A/B credit. Save each ZIP before disconnecting Colab.

**Before uploading:** read the printed notebook summary, file list, and warnings.
Check that all notebooks you want graded and their available prediction files
are included. Open `report.pdf` inside the ZIP to confirm it is your latest report.
Warnings for intentionally unfinished questions are expected; fix warnings about
work you intended to include. Every Gradescope resubmission must contain **all**
work you want graded; earlier attempts are not combined with the newest one.

**Packaging after a runtime reset:** you can run just the final cell; it downloads
the small support bundle from Hugging Face only if its helper is missing. No GPU, dataset download,
dependency installation, or training is needed for packaging. Set
`include_previous=True` and import your saved submission ZIP to recover earlier
results. If newer results exist only on Drive, mount it and restore your `RESULTS`
path first; the helper cannot recover files from a deleted runtime.

The ZIP contains the available `hw4-a.ipynb`, `hw4-b.ipynb`, `hw4-c.ipynb`,
`report.pdf`, and prediction files under `results/`. It excludes data, checkpoints,
pretrained weights, audio, and instructor answer keys. The staff submission with
all three notebooks, complete prediction files, and A's plots is about **3.1 MiB
before the report PDF**. Keep PDF figures reasonably compressed.

A performance uses `results/a/phoneme_layer_0.json` through layer 12 and the
corresponding `mfcc_layer_*.json`, saved from `test_predictions`. A summary metric
alone earns no performance credit. B uses `results/asr_submission.txt` (95 test
IDs); C uses `results/tts_submission.json` (200 test IDs). Missing layers or
predictions affect only their related credit; missing ASR/TTS predictions count
as deletions in corpus WER. Exporters resume matching predictions and archive them
when the model or voice prompt changes, so reevaluating a new model does not
retain stale results. An entirely empty submission receives zero without crashing.

Locally, put your edited notebooks, `results/`, and the shared `report.pdf` in one
folder and run `python make_submission.py --root YOUR_WORK --report report.pdf`.
Use `--no-report` only when deliberately submitting without a PDF.

**Alternative: separate part ZIPs in one upload.** The autograder also accepts
`hw4-a.zip`, `hw4-b.zip`, and `hw4-c.zip` selected **together in the same submission
attempt**, with one shared `report.pdf` included once (inside one ZIP or alongside
them). Each ZIP should contain only its part's current work. Identical copies
count once; conflicting copies of a file are flagged and omitted while other work
is still graded. This alternative still requires one combined PDF, not separate
A/B/C reports. Do not submit the parts as successive attempts.
Gradescope's [programming submission instructions](https://www.gradescope.com/help#submitting-code)
allow multiple files in one upload; [its grading guide](https://guides.gradescope.com/hc/en-us/articles/22066635961357-Grading-a-Programming-Assignment)
explains that the latest submission determines the autograder score.
