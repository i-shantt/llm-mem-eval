"""LLM judge for answer correctness, plus the machinery to validate it.

An unvalidated judge is worthless here. The audit of LoCoMo found its judge
accepts up to 63% of intentionally wrong answers, and that benchmark's numbers
are still cited. So this module ships three things together: the judge, an
exporter that produces a hand-labelling worksheet, and an agreement calculation
against those hand labels.

Rule for this repo: no judged number is reported without its judge/human
agreement alongside it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

JUDGE_PROMPT = """You are grading whether a model's answer to a question about a \
past conversation is correct.

Question: {question}
Correct answer: {gold}
Model's answer: {pred}

The model's answer is correct if it conveys the same information as the correct \
answer. Ignore differences in wording, extra detail, or politeness. It is \
incorrect if it states different facts, contradicts the correct answer, or fails \
to answer.

Reply with exactly one word: CORRECT or INCORRECT."""

ABSTAIN_PROMPT = """You are grading whether a model correctly declined to answer.

Question: {question}
Model's answer: {pred}

This question cannot be answered from the conversation history. The model is \
CORRECT only if it says it does not know, cannot find the information, or that \
the information was never discussed. It is INCORRECT if it invents an answer.

Reply with exactly one word: CORRECT or INCORRECT."""

_YES = re.compile(r"\bcorrect\b", re.I)
_NO = re.compile(r"\bincorrect\b", re.I)


def parse_verdict(text: str) -> bool | None:
    """True=correct, False=incorrect, None=unparseable.

    Checks INCORRECT first, since 'INCORRECT' contains 'CORRECT'.
    """
    if _NO.search(text):
        return False
    if _YES.search(text):
        return True
    return None


@dataclass
class JudgedAnswer:
    question_id: str
    question: str
    gold: str
    pred: str
    is_abstention: bool
    verdict: bool | None
    raw_verdict: str


def judge_answer(
    backend,
    question: str,
    gold: str,
    pred: str,
    is_abstention: bool = False,
) -> tuple[bool | None, str]:
    template = ABSTAIN_PROMPT if is_abstention else JUDGE_PROMPT
    prompt = template.format(question=question, gold=gold, pred=pred)
    gen = backend.generate(prompt, max_new_tokens=8)
    return parse_verdict(gen.text), gen.text


def export_labelling_worksheet(
    judged: list[JudgedAnswer], out_path: str | Path, limit: int = 100
) -> Path:
    """Write a JSONL worksheet for a human to label.

    Each line gets a `human_label` field set to null. Fill it with true/false.
    The judge's own verdict is deliberately omitted so the labeller is not
    anchored by it.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for j in judged[:limit]:
            f.write(json.dumps({
                "question_id": j.question_id,
                "question": j.question,
                "correct_answer": j.gold,
                "model_answer": j.pred,
                "is_abstention": j.is_abstention,
                "human_label": None,
                "_instructions": "set human_label to true if model_answer "
                                 "conveys the same information as "
                                 "correct_answer, else false",
            }) + "\n")
    return out


def judge_agreement(
    judged: list[JudgedAnswer], worksheet_path: str | Path
) -> dict:
    """Compare judge verdicts against human labels.

    Reports raw agreement and Cohen's kappa. Kappa matters because these classes
    are imbalanced -- if 80% of answers are correct, a judge that always says
    CORRECT gets 80% agreement and is useless.
    """
    human: dict[str, bool] = {}
    with Path(worksheet_path).open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("human_label") is not None:
                human[row["question_id"]] = bool(row["human_label"])

    by_id = {j.question_id: j for j in judged}
    pairs = [
        (human[qid], by_id[qid].verdict)
        for qid in human
        if qid in by_id and by_id[qid].verdict is not None
    ]
    if not pairs:
        return {"n_labelled": len(human), "n_comparable": 0,
                "error": "no overlapping labelled examples with parseable verdicts"}

    n = len(pairs)
    agree = sum(1 for h, j in pairs if h == j)

    # Cohen's kappa
    both_true = sum(1 for h, j in pairs if h and j)
    both_false = sum(1 for h, j in pairs if not h and not j)
    h_true = sum(1 for h, _ in pairs if h)
    j_true = sum(1 for _, j in pairs if j)
    po = (both_true + both_false) / n
    pe = (h_true / n) * (j_true / n) + (1 - h_true / n) * (1 - j_true / n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    # The failure mode the LoCoMo audit found: judge accepting wrong answers.
    human_wrong = [(h, j) for h, j in pairs if not h]
    false_accept = (
        sum(1 for _, j in human_wrong if j) / len(human_wrong)
        if human_wrong else None
    )

    return {
        "n_labelled": len(human),
        "n_comparable": n,
        "raw_agreement": agree / n,
        "cohens_kappa": kappa,
        "false_accept_rate": false_accept,
        "false_accept_note": "fraction of human-judged-WRONG answers the judge "
                             "accepted; the LoCoMo audit found up to 0.63",
        "human_correct_rate": h_true / n,
        "judge_correct_rate": j_true / n,
    }
