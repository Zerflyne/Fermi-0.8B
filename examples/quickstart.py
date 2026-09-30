"""FERMI in five lines: python examples/quickstart.py  (add --device cpu to force the CPU)"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fermi import Fermi  # noqa: E402

device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "auto"
m = Fermi.load(device=device)

state = {
    "from": "customer@example.com",
    "subject": "Order arrived broken",
    "body": "The lamp I ordered arrived with the glass shattered. I'd like a replacement, not a refund. Photos attached.",
}
questions = {
    "wants_refund": {"type": "noul", "instructions": "Does the customer ask for a refund?",
                     "criteria": {"true": "Asks for money back", "false": "Does not ask for money back"}},
    "request": {"type": "choice", "instructions": "What does the customer want?",
                "criteria": {"replacement": "A new item", "refund": "Money back", "information": "An answer or explanation",
                             "other": "Something else"}},
    "urgency": {"type": "score", "instructions": "How urgent is the case?", "criteria": ["Low", "Medium", "High"]},
}
print(json.dumps(m.classify(state, questions), indent=1, ensure_ascii=False))
