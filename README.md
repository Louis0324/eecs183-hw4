# EECS 183/283A Homework 4: Speech Processing

Complete Parts A, B, and C in Google Colab. The notebooks contain the coding TODOs,
report questions, and grading rubrics.

## 1. Start in Colab

| Part | Notebook | Open in Colab |
|---|---|---|
| A | [Probing HuBERT](hw4-a.ipynb) | [Open A](https://colab.research.google.com/github/Louis0324/eecs183-hw4/blob/main/hw4-a.ipynb) |
| B | [Automatic speech recognition](hw4-b.ipynb) | [Open B](https://colab.research.google.com/github/Louis0324/eecs183-hw4/blob/main/hw4-b.ipynb) |
| C | [Text-to-speech](hw4-c.ipynb) | [Open C](https://colab.research.google.com/github/Louis0324/eecs183-hw4/blob/main/hw4-c.ipynb) |

1. Choose **File → Save a copy in Drive**, then **Runtime → Change runtime type → GPU**.
2. Run the setup and data cells. Materials download automatically from
   [Hugging Face](https://huggingface.co/datasets/Louis0324/eecs183-hw4); no account or manual download is needed.
   See [download sizes](data/README.md).
3. Complete the TODOs. Set `SMOKE_TEST=True` in setup for a quick check. For full
   experiments, set it to `False`, choose a fresh `RESULTS` folder, and rerun from setup.
   Run evaluation/export cells before packaging; performance points use saved test predictions.

Part C includes a starter checkpoint and defaults to **one additional training epoch**.
**Colab storage is temporary:** save your notebooks, checkpoints, and results to
Drive or download them before your runtime ends.

## 2. Write one report

Use [report_template.txt](report_template.txt) to create **one `report.pdf` for all
three parts**. Put conceptual answers and requested figures in the PDF, using the
section headings **A1, A2, A3, B1, C1**. Aim for 3–5 sentences total per section,
plus figures/sequences. Keep text selectable; mark unfinished items `Not completed`.

**Grading:** 85 points from the autograder (code and model performance) + 15 points
for the report. The optional speaker experiment offers up to 10 bonus points.
Detailed rubrics are in each notebook.

## 3. Submit one combined ZIP to Gradescope

Run the **final cell** in each notebook to package your work:

1. **After A:** save the downloaded ZIP as `hw4-a.zip`.
2. **After B:** upload `hw4-a.zip` when asked for an earlier ZIP. Save the new
   download as `hw4-ab.zip`.
3. **After C:** upload `hw4-ab.zip` at the earlier-ZIP prompt and your final
   `report.pdf` at the report prompt. Submit the resulting **`submission.zip`**
   to Gradescope.

At earlier report prompts, upload your current combined PDF or cancel if it is
not ready. Cancelling keeps any PDF already imported or saved by the packager.

**Incomplete work is accepted.** If you finish only A or A+B, submit your latest
ZIP with your report. For B or C alone, cancel the earlier-ZIP prompt.
Correctly completed questions are graded independently.

The ZIP includes your notebooks, available predictions/plots, and report.
**Do not submit checkpoints, datasets, or audio.** Check the printed file list and
open the report before uploading. Every resubmission must include **all** work you
want graded; previous submissions are replaced.

For other completion orders or a runtime reset, see **Prepare your submission**
at the end of any notebook.
