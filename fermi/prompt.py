"""How FERMI sees a state and a question: exactly the format used in training.

One example = one question:
    <system> State: <state> <question header> A. option\n B. option\n ... <end>
The decision head reads the hidden state of the last token of every option line and of the very last token
(which has seen all the options). A yes/no question ("noul") is two options: yes / no.
"""
import json

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_LEN = 2048          # tokens per example: a longer state is shortened (head + tail), as in training

SYSTEM = ("<|im_start|>system\nRead the state, then answer the question about it. "
          "Judge only from what the state shows.<|im_end|>\n<|im_start|>user\nState:\n")
END = "<|im_end|>\n<|im_start|>assistant\n"
TYPES = {"noul": 0, "choice": 1, "score": 2}


def state_text(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def question_parts(q):
    """(header, [option lines]) as text."""
    t = q["type"]
    if t == "noul":
        c = q.get("criteria") or {}
        head = "\n\nQuestion (yes or no): " + q["instructions"] + "\n"
        lines = ["A. yes" + (": " + c["true"] if c.get("true") else "") + "\n",
                 "B. no" + (": " + c["false"] if c.get("false") else "") + "\n"]
    elif t == "choice":
        head = "\n\nQuestion (pick one option): " + q["instructions"] + "\n"
        lines = ["%s. %s%s\n" % (LETTERS[i], k, (": " + d) if d else "") for i, (k, d) in enumerate(q["criteria"].items())]
    elif t == "score":
        head = "\n\nQuestion (pick one level of the scale): " + q["instructions"] + "\n"
        lines = ["%s. %s\n" % (LETTERS[i], lv) for i, lv in enumerate(q["criteria"])]
    else:
        raise ValueError("unknown question type %r (noul, choice or score)" % t)
    return head, lines


def check_question(qid, q):
    t = q.get("type")
    if t not in TYPES:
        raise ValueError("%s: type must be noul, choice or score" % qid)
    if not q.get("instructions"):
        raise ValueError("%s: instructions missing" % qid)
    if t == "choice" and (not isinstance(q.get("criteria"), dict) or not 2 <= len(q["criteria"]) <= len(LETTERS)):
        raise ValueError("%s: choice needs criteria as {option: description} with 2-%d options" % (qid, len(LETTERS)))
    if t == "score" and (not isinstance(q.get("criteria"), list) or not 2 <= len(q["criteria"]) <= len(LETTERS)):
        raise ValueError("%s: score needs criteria as a list of 2-%d levels, lowest first" % (qid, len(LETTERS)))


class Builder:
    """Token ids and option positions for (state, question), with a tokenizer that has encode(text) -> ids."""

    def __init__(self, encode):
        self.encode = encode
        self.sys = encode(SYSTEM)
        self.end = encode(END)
        self.ellipsis = encode(" […] ")[:3]

    def build(self, state_ids, q):
        head, lines = question_parts(q)
        qh = self.encode(head)
        opts = [self.encode(x) for x in lines]
        rest = len(self.sys) + len(qh) + sum(map(len, opts)) + len(self.end)
        room = MAX_LEN - rest
        if room < 64:
            raise ValueError("question too long (%d tokens without the state)" % rest)
        st = state_ids
        if len(st) > room:
            a = int(room * 0.7) - 3
            b = room - a - 3
            st = st[:a] + self.ellipsis + st[-b:]
        ids = self.sys + st + qh
        pos = []
        for o in opts:
            ids = ids + o
            pos.append(len(ids) - 1)
        return ids + self.end, pos
