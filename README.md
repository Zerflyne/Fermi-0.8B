<p align="center"><img src="docs/fermi-banner.svg" alt="FERMI" width="600"></p>

# FERMI-0.8B

**FERMI** (Fast Embedded Reasoning for Machine Inference) is a small open model that reads a **state** (any text or
JSON: an email, a chat message, a log, a board position, a product listing…) and answers **typed questions** about it
with **probabilities**. It never generates text: every answer is a distribution you can threshold, sort or average.

| question type | you give | you get |
|---|---|---|
| `noul` (yes/no) | a question, optional criteria for yes and no | `p_yes` |
| `choice` | a question and 2–26 named options | a probability per option |
| `score` | a question and 2–26 ordered levels | a probability per level + the expected level |

One state can carry many questions; each is scored independently. This is version **0.1**, a first prototype:
Qwen3.5-0.8B + a LoRA adapter + a small decision head, trained for about 6 hours on one RTX 4060.

**Bigger sibling:** [FERMI-2B](https://github.com/Zerflyne/FERMI-2B) (MiniCPM5-2B base, one full epoch): 85.3% agreement
with the teacher instead of 81.6%, same code and demos. Use it if you have ~5 GB of GPU memory.

## Desktop app (easiest)

The [FERMI app](https://github.com/Zerflyne/FERMI) runs FERMI-0.8B on Linux, Windows and macOS, on a graphics card (Vulkan,
Metal) or on the CPU, with no Python: install it, pick FERMI-0.8B on the first start and it downloads
[`gguf/FERMI-0.8B-Q8_0.gguf`](https://huggingface.co/Zerflyne/FERMI-0.8B/blob/main/gguf/FERMI-0.8B-Q8_0.gguf) by itself. That file
is the base model with FERMI's LoRA merged in, quantized to Q8_0, with the decision head stored in its metadata; it is
made for the app's llama.cpp runtime (it is not a chat model). The app has a playground, folder analysis with charts and
a local JSON API.

## Quick start

```bash
git clone https://github.com/Zerflyne/Fermi-0.8B && cd Fermi-0.8B
pip install -r requirements.txt          # CPU; for NVIDIA GPUs: requirements-gpu.txt
python examples/quickstart.py
```

```python
from fermi import Fermi

m = Fermi.load()                         # GPU if available, else CPU; downloads Qwen/Qwen3.5-0.8B once
m.classify(
    state={"subject": "Charged twice!!", "body": "I was charged twice for March. I need the money back before Friday."},
    questions={
        "refund":  {"type": "noul",   "instructions": "Is the customer asking for a refund?"},
        "topic":   {"type": "choice", "instructions": "What is the main topic?",
                    "criteria": {"billing": "Payments, charges", "technical": "Bugs, errors", "other": "Anything else"}},
        "urgency": {"type": "score",  "instructions": "How urgent is it?", "criteria": ["Low", "Medium", "High"]},
    })
```

```json
{
 "refund":  {"type": "noul", "answer": "yes", "p_yes": 0.8693},
 "topic":   {"type": "choice", "answer": "billing", "probabilities": {"billing": 0.9934, "technical": 0.0007, "other": 0.0059}},
 "urgency": {"type": "score", "answer": 2, "answer_text": "High", "probabilities": [0.1122, 0.1179, 0.7699],
             "expected": 1.658, "expected_01": 0.8288}
}
```

For `noul`, `criteria` is optional: `{"true": "what counts as yes", "false": "what counts as no"}`.
For `score`, list the levels from lowest to highest.

## Interactive demos

```bash
python server.py            # then open http://127.0.0.1:8765
```

- **Playground**: write any state, build your own questions, see every distribution.
- **Live moderation**: FERMI checks a chat message while you type (acceptable? kind? severity?).
- **Inbox triage**: eight emails, three questions each; FERMI sorts them by expected urgency.
- **Twenty questions**: a secret card that FERMI can read and you can't; ask yes/no questions and read the odds.
- **Tic-tac-toe**: play against FERMI; each move is a `choice` over the empty cells, shown as a heat map.
  It plays badly, and that is instructive: FERMI is a classifier, not a game engine.

Games that show how much a classifier depends on what you tell it. Most have a **hints** switch: with hints, the page
computes plain facts about each option (this column blocks X, this move hits the wall…) and FERMI still has to read them
and pick; without hints, FERMI gets only the raw board.

- **Snake**: FERMI steers in real time, one "which way?" question per step.
- **Connect four**: play against FERMI and see the probability of every column.
- **Pong**: FERMI holds a paddle at 60 frames per second, answering "up, stay or down?" as fast as it can.
- **Blackjack**: FERMI decides hit or stand; the textbook basic strategy scores how often it agrees with the math.
- **Rock, paper, scissors**: FERMI predicts your next move from the history and plays the counter; try the pattern bots.

Choice questions in the games are asked with the options in several orders and averaged (`fermiChoice` in
`demos/fermi.js`): small classifiers have a position bias, and this removes most of it.

The server also exposes the model as a JSON API: `POST /api/classify` with `{"state": ..., "questions": {...}}`.

## Hardware

| | memory | speed (measured) |
|---|---|---|
| NVIDIA GPU, bfloat16, with `flash-linear-attention` | ~2 GB VRAM | not measured yet (much faster than CPU) |
| CPU, float32 (8 threads of a desktop CPU) | ~5.5 GB RAM | ~0.5–1 s per question; 24 questions in ~15 s |
| Apple MPS | — | should work through the PyTorch path, not tested yet |

`Fermi.load(device="cpu")` forces the CPU. `Fermi.load(base="/path/to/Qwen3.5-0.8B")` (or `FERMI_BASE=...`) uses a local
copy of the base model instead of downloading it. On CPU, `OMP_NUM_THREADS` limits how many cores it takes.

## Results

Evaluation on **5,651 questions about 1,078 states the model never saw** (split by state). "Teacher" is the model that
produced the training labels (see below); agreement = same most-likely answer.

| questions | n | agreement with the teacher | cross-entropy vs teacher |
|---|---|---|---|
| all | 5,651 | **81.6%** | 0.623 |
| yes/no | 2,389 | 88.7% | 0.365 |
| choice | 1,684 | 81.0% | 0.705 |
| score | 1,578 | 71.6% | 0.927 |

On a separate set of 200 reference questions answered by an external commercial classifier (Jev), FERMI-0.8B gives
the same answer **76%** of the time; the 27B teacher itself agrees **81%**. For score questions, an answer one level
off counts as a miss here, but it is usually a near miss (the distributions are close).

### Checkpoints

| folder | training step | eval agreement with the teacher* | agreement on the 200 reference questions* |
|---|---|---|---|
| `checkpoints/step-200` | 200 / 783 | 77.3% | 76.5% |
| `checkpoints/step-400` | 400 / 783 | 81.1% | 77.0% |
| `checkpoints/step-600` | 600 / 783 | 81.0% | 77.0% |
| `checkpoints/final` | 783 / 783 | 81.6% (full eval set) | 76.0% |

\* intermediate checkpoints were measured on a 2,538-question subset of the evaluation set.
Each folder holds `adapter.safetensors` (LoRA, bfloat16, ~42 MB), `head.safetensors` and `config.json` (with its
evaluation). Load one with `Fermi.load("checkpoints/step-600")` or `python server.py --checkpoint checkpoints/step-600`.

## How it works

**Input.** State and question are written into one prompt; every option becomes one line (`A. option: description`).
A yes/no question is two lines, `yes` and `no`. Long states are cut to fit 2,048 tokens (start and end are kept).

**Head.** FERMI reads the last hidden state of every option line and of the final token, which has seen all the
options: `score_i = v · gelu(A·h_option_i + B·h_end + type)`. A softmax over the options gives the answer
distribution. Options are scored by position, so the number of options is not fixed.

**Base and adapter.** Qwen3.5-0.8B (hybrid linear/full attention) is frozen. A LoRA adapter, rank 32 and alpha 64,
sits on the attention, linear-attention and MLP projections of every layer (20.4M parameters), plus the head (1.05M). At load time the LoRA is merged
into the weights, so inference costs the same as the base model.

**Data.** 48,942 states built from open datasets (emails, chats, reviews, code reviews, papers, recipes, resumes,
server metrics, chess and connect-four positions and more, in several languages), with 252,990 generated questions:
41% yes/no, 31% choice, 28% score.

**Labels.** Soft labels from **Qwen3.8-27B** (NVFP4), asked each question with a clean context: it saw only the state
and the question, and was constrained to one answer token. The label is its renormalized next-token probability.
Choice and score questions were asked twice, the second time with the options in reverse order, and the two
distributions were averaged to cancel position bias. Yes/no labels got a +0.5 logit shift toward "yes", because the
teacher leaned toward "no" on reference questions.

**Training.** Soft-label cross-entropy (a proper scoring rule, so probabilities stay meaningful); AdamW, learning
rate 2e-4 for the LoRA and 1e-3 for the head, cosine schedule; about 124k questions (half an epoch). It took 5.7 hours
on an RTX 4060 8 GB.

## Limitations

- **Distilled.** FERMI imitates a 27B teacher and inherits its mistakes; it is not ground truth.
- **Half an epoch.** v0.1 saw half of the training data once. [FERMI-2B](https://github.com/Zerflyne/FERMI-2B) saw all of it.
- **Calibration.** On unseen states, when FERMI is confident it is usually right, but at the end of training it
  became somewhat overconfident (expected calibration error vs the teacher's top answer: 0.068). Treat mid-range
  probabilities as "unsure", not as exact frequencies.
- **Weak at games and precise reasoning.** Board positions, arithmetic and multi-step logic are beyond a 0.8B
  classifier; see the tic-tac-toe demo.
- **Phrasing matters.** Questions that name what the state describes ("Is the product damaged?") work better than
  vague ones ("Is it bad?").
- Not for high-stakes decisions about people (medical, legal, hiring, credit) without human review.

## License

Apache-2.0, like the base model [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B).
