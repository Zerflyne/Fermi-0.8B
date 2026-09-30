"""FERMI: a small open classifier that answers typed questions about a state with calibrated probabilities.

    from fermi import Fermi
    m = Fermi.load()                         # GPU if available, otherwise CPU
    m.classify(state, {"q1": {"type": "noul", "instructions": "..."}})

Base model: Qwen/Qwen3.5-0.8B (downloaded from Hugging Face, or a local folder). On top of it: a LoRA adapter
(merged into the weights at load time) and a small decision head that scores every option.
"""
import inspect
import json
import os
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prompt import TYPES, Builder, check_question, state_text

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "..", "checkpoints", "final")
PAD = 248044


class Head(nn.Module):
    """score_i = v . gelu(A h_option_i + B h_end + type)"""

    def __init__(self, hidden, d=512):
        super().__init__()
        self.a = nn.Linear(hidden, d)
        self.b = nn.Linear(hidden, d, bias=False)
        self.tipo = nn.Embedding(3, d)
        self.v = nn.Linear(d, 1)

    def forward(self, h_opt, h_end, kind):
        return self.v(F.gelu(self.a(h_opt) + self.b(h_end) + self.tipo(kind))).squeeze(-1)


def _pick_device(device):
    if device not in (None, "auto"):
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _kernels(device):
    """Fast linear-attention kernels (flash-linear-attention, Triton) on CUDA; the PyTorch reference elsewhere."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    fla = False
    if device.type == "cuda":
        try:
            from fla.modules.convolution import causal_conv1d

            def conv(hidden_states, weight, bias=None, activation=None, **kw):
                y, _ = causal_conv1d(x=hidden_states.transpose(1, 2), weight=weight, bias=bias, activation=activation)
                return y.transpose(1, 2)
            M.causal_conv1d_fn = conv
            fla = True
        except ImportError:
            pass
    else:
        # transformers picks the fla kernel whenever the package is installed, even for CPU tensors
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn"):
            if hasattr(M, name):
                setattr(M, name, inspect.unwrap(getattr(M, name)))
    return fla


class Fermi:
    def __init__(self, text_model, head, tokenizer, device, dtype, config):
        self.model = text_model
        self.head = head
        self.tok = tokenizer
        self.device = device
        self.dtype = dtype
        self.config = config
        self.builder = Builder(lambda s: tokenizer(s, add_special_tokens=False)["input_ids"])
        self.lock = threading.Lock()

    @classmethod
    def load(cls, checkpoint=DEFAULT_CKPT, base=None, device="auto", dtype=None, verbose=True):
        """checkpoint: folder with adapter.safetensors, head.safetensors, config.json.
        base: Hugging Face id or local folder of Qwen3.5-0.8B (default: the one in config.json).
        dtype: default bfloat16 on GPU, float32 on CPU."""
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = json.load(open(os.path.join(checkpoint, "config.json")))
        base = base or os.environ.get("FERMI_BASE") or cfg["base_model"]
        dev = _pick_device(device)
        fla = _kernels(dev)
        if dtype is None:
            dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        if verbose:
            print("FERMI: base %s on %s (%s)%s" % (base, dev, str(dtype).replace("torch.", ""),
                                                   ", fast kernels" if fla else ""))
        tok = AutoTokenizer.from_pretrained(base)
        full = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
        text = full.model if hasattr(full, "model") and hasattr(full.model, "layers") else full.get_decoder()
        del full
        lcfg = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], target_modules=cfg["lora_targets"],
                          lora_dropout=0.0, bias="none")
        text = get_peft_model(text, lcfg)
        adapter = {k: v.to(torch.float32) for k, v in load_file(os.path.join(checkpoint, "adapter.safetensors")).items()}
        res = set_peft_model_state_dict(text, adapter)
        missing = [k for k in getattr(res, "unexpected_keys", []) or []]
        if missing:
            raise RuntimeError("adapter keys not used: %s" % missing[:5])
        text = text.merge_and_unload()           # the LoRA goes into the weights: no overhead at inference
        text.to(dev).eval()
        head = Head(text.config.hidden_size, cfg.get("head_dim", 512))
        head.load_state_dict(load_file(os.path.join(checkpoint, "head.safetensors")))
        head.to(dev).eval()
        return cls(text, head, tok, dev, dtype, cfg)

    @torch.no_grad()
    def classify(self, state, questions, batch_tokens=None):
        """state: text or JSON-like object. questions: {id: {"type": "noul"|"choice"|"score", "instructions": str,
        "criteria": ...}}; criteria: noul {"true": ..., "false": ...} (optional), choice {option: description},
        score [lowest level, ..., highest level].

        Returns {id: answer}: noul {"answer": "yes"|"no", "p_yes": p}; choice {"answer": option,
        "probabilities": {option: p}}; score {"answer": level index, "probabilities": [p per level],
        "expected": mean level index, "expected_01": expected rescaled to 0-1}."""
        for qid, q in questions.items():
            check_question(qid, q)
        st = self.builder.encode(state_text(state))
        items = []
        for qid, q in questions.items():
            ids, pos = self.builder.build(st, q)
            items.append((qid, q, ids, pos))
        budget = batch_tokens or (16384 if self.device.type == "cuda" else 4096)
        out = {}
        with self.lock:
            for group in _groups(items, budget):
                for (qid, q, _, _), p in zip(group, self._run(group)):
                    out[qid] = _answer(q, p)
        return {qid: out[qid] for qid in questions}      # in the order of the questions

    def _run(self, group):
        L = max(len(x[2]) for x in group)
        ids = torch.full((len(group), L), PAD, dtype=torch.long)
        for i, (_, _, t, _) in enumerate(group):
            ids[i, :len(t)] = torch.tensor(t)
        ids = ids.to(self.device)
        use_autocast = self.device.type == "cuda" and self.dtype != torch.float32
        with torch.autocast("cuda", dtype=self.dtype, enabled=use_autocast):
            h = self.model(input_ids=ids, use_cache=False).last_hidden_state
        res = []
        for i, (_, q, t, pos) in enumerate(group):
            hp = h[i, pos].float()
            he = h[i, len(t) - 1].float().expand_as(hp)
            kind = torch.full((len(pos),), TYPES[q["type"]], dtype=torch.long, device=self.device)
            s = self.head.float()(hp, he, kind)
            res.append(torch.softmax(s, -1).cpu().tolist())
        return res


def _groups(items, budget):
    """Batches of similar length within a token budget (rows x longest)."""
    items = sorted(items, key=lambda x: len(x[2]))
    cur = []
    for it in items:
        if cur and (len(cur) + 1) * len(it[2]) > budget:
            yield cur
            cur = []
        cur.append(it)
    if cur:
        yield cur


def _answer(q, p):
    t = q["type"]
    if t == "noul":
        return {"type": t, "answer": "yes" if p[0] >= 0.5 else "no", "p_yes": round(p[0], 4)}
    if t == "choice":
        keys = list(q["criteria"])
        best = max(range(len(p)), key=lambda i: p[i])
        return {"type": t, "answer": keys[best], "probabilities": {k: round(x, 4) for k, x in zip(keys, p)}}
    best = max(range(len(p)), key=lambda i: p[i])
    exp = sum(i * x for i, x in enumerate(p))
    return {"type": t, "answer": best, "answer_text": q["criteria"][best], "probabilities": [round(x, 4) for x in p],
            "expected": round(exp, 3), "expected_01": round(exp / (len(p) - 1), 4)}
