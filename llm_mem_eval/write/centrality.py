"""Keep the most *central* sentences: a third point on the same curve.

`ExtractiveSelectionPolicy` spends a token budget on the first sentences of every
turn, or on the last ones. Both are positional rules, and together they answer
one question -- whether answers cluster at the start of a message. Neither
answers the question a real extraction step is trying to answer, which is
whether the *representative* content of a turn can be identified at all.

This policy varies exactly one thing against lead-k and tail-k: which sentence
of a turn survives. Budget, granularity, round-robin breadth, the skip-don't-stop
rule and the re-join into original order are all identical, deliberately. So a
gap between this arm and lead-k at the same budget is attributable to selection
and to nothing else, which is the only reason to add a third selector rather
than a third budget.

**Centrality is computed within a turn, not across the conversation.** Ranking
every sentence in the haystack globally and taking the top ones would change two
things at once -- selection *and* breadth -- and the resulting store would be
some turns in full and others absent, which is what `TruncatedVerbatimPolicy`
already measures. Per-turn holds the round-robin schedule fixed at lead-k's:
every turn is offered its first pick before any turn gets a second.

That fixes the *schedule*, not the realised breadth, and the difference decides
which budgets this arm can be read at. A turn's most central sentence is
systematically longer than its first -- centrality rewards a sentence for
resembling its neighbours, and short sentences share fewer words to resemble
them with. Measured on the eligible subset, this policy keeps 40 tokens per
record against lead-k's 26 at a 5% budget, and 39 against 26 at 10%.

Above a threshold that does not matter, because the budget covers at least one
sentence from every turn either way and both rules land on all 497 records:

    budget   lead-k records   this policy   tokens/record (lead-k -> this)
      5%          207             135              26 -> 40
     10%          411             269              26 -> 39
     25%          497             497              52 -> 53
     50%          497             497             105 -> 105

So **only the 25% and 50% rows are a controlled comparison.** At 5% and 10% the
budget runs out before every turn is reached, longer picks reach fewer turns,
and the arms differ in coverage as well as in selection -- which is the axis
`TruncatedVerbatimPolicy` already isolates and the dominant effect in this
repo's survival results. Reading a low-budget gap as evidence about selection
would be attributing a coverage effect to the wrong cause.

IDF is estimated over every sentence in the conversation even though similarity
is only ever computed within a turn. A turn holds about four sentences, and IDF
over four documents is noise; the conversation gives a few thousand, which is
enough for common words to actually discount.

**Where this necessarily agrees with lead-k.** 16% of turns are a single
sentence, and every selector keeps that one. Below three sentences there is very
little for centrality to say. The measured ceiling on how far this arm can move
is therefore well under 100% of turns -- median 4 sentences, 64% with three or
more -- and a small gap should be read against that ceiling rather than as
evidence that selection does not matter.

When a turn's sentences share no vocabulary the similarity matrix is all zeros,
every sentence scores alike, and the stable sort falls back to original order --
i.e. to lead-k exactly. That is the honest degenerate case: with no signal, this
policy is lead-k rather than something arbitrary.

**What it measured, stated as the null result it is.** At the two controlled
budgets the paired difference against lead-k is -0.045 at 25% and -0.027 at 50%,
both with 95% intervals crossing zero (McNemar p 0.38 and 0.45). Centrality
selection is therefore *not distinguishable* from taking the first sentence.
That is not the same as showing the two are equivalent: the survival subset is
110 questions after the two-token restriction, which puts the detection floor
around 8 points, and a real effect smaller than that would not show up here. The
arm's contribution is a bound, not a winner.

No LLM call, so this is priced like the other extractive arms: CPU only.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from llm_mem_eval.cost import CostLedger, count_tokens
from llm_mem_eval.data.loader import Example, MemoryUnit
from llm_mem_eval.write.base import renumber
from llm_mem_eval.write.extractive import split_sentences

_WORD_RE = re.compile(r"[a-z0-9']+")

# Continuous LexRank on a row-normalised similarity matrix. 0.85 is PageRank's
# conventional damping; the ranking is insensitive to it here because the
# matrices are tiny, and it is pinned rather than tuned so no arm is chosen by
# its own hyperparameter.
_DAMPING = 0.85
_POWER_ITERATIONS = 40
_CONVERGENCE = 1e-8


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def idf_over(sentences: list[str]) -> dict[str, float]:
    """Smoothed IDF over the conversation's sentences."""
    n = len(sentences)
    df: Counter = Counter()
    for s in sentences:
        for w in set(tokenize(s)):
            df[w] += 1
    return {w: math.log(n / (1 + c)) + 1.0 for w, c in df.items()}


def tfidf_vector(sentence: str, idf: dict[str, float]) -> dict[str, float]:
    """L2-normalised, so a dot product is a cosine."""
    tf = Counter(tokenize(sentence))
    vec = {w: c * idf.get(w, 1.0) for w, c in tf.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm == 0:
        return {}
    return {w: v / norm for w, v in vec.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(w, 0.0) for w, v in a.items())


def lexrank_order(sentences: list[str], idf: dict[str, float]) -> list[int]:
    """Sentence indices, most central first. Ties keep original order."""
    n = len(sentences)
    if n <= 1:
        return list(range(n))

    vecs = [tfidf_vector(s, idf) for s in sentences]
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = cosine(vecs[i], vecs[j])
            sim[i][j] = sim[j][i] = s

    # Row-normalise into a stochastic matrix. A row that sums to zero -- a
    # sentence sharing no vocabulary with any other -- becomes uniform, which
    # keeps the chain irreducible instead of trapping mass at a dangling node.
    for i in range(n):
        total = sum(sim[i])
        if total <= 0:
            sim[i] = [1.0 / n] * n
        else:
            sim[i] = [v / total for v in sim[i]]

    scores = [1.0 / n] * n
    for _ in range(_POWER_ITERATIONS):
        nxt = [
            (1.0 - _DAMPING) / n
            + _DAMPING * sum(scores[j] * sim[j][i] for j in range(n))
            for i in range(n)
        ]
        delta = sum(abs(a - b) for a, b in zip(nxt, scores))
        scores = nxt
        if delta < _CONVERGENCE:
            break

    # Stable sort on the negated score: equal scores keep original order, so a
    # turn with no usable signal degenerates to lead-k rather than to an
    # arbitrary permutation.
    return sorted(range(n), key=lambda i: -scores[i])


class CentralitySelectionPolicy:
    """LexRank sentences per turn, round-robin, to a token budget."""

    def __init__(self, fraction: float, granularity: str = "turn"):
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
        self.fraction = fraction
        self.granularity = granularity
        self.name = "lexrank_" + f"{fraction:.0%}".replace("%", "pct")

    def config(self) -> dict:
        return {
            "policy": "centrality_lexrank",
            "fraction": self.fraction,
            "granularity": self.granularity,
            "selector": "round-robin LexRank sentences, centrality within a turn",
            "similarity": "tf-idf cosine, idf over the conversation's sentences",
            "damping": _DAMPING,
            "comparable_to": (
                "leadk/tailk at the same fraction: identical budget, breadth and "
                "round-robin, differing only in which sentence of a turn is kept."
            ),
        }

    def build(self, ex: Example, ledger: CostLedger) -> list[MemoryUnit]:
        with ledger.timer("write"):
            units = ex.units(self.granularity)
            budget = self.fraction * sum(count_tokens(u.text) for u in units)

            sents = [split_sentences(u.text) for u in units]
            idf = idf_over([s for group in sents for s in group])
            orders = [lexrank_order(group, idf) for group in sents]

            kept: list[list[int]] = [[] for _ in units]
            used = 0
            depth = 0
            max_depth = max((len(s) for s in sents), default=0)

            # Identical to ExtractiveSelectionPolicy's loop, including
            # skip-don't-stop, so the two arms differ only in `orders`.
            while depth < max_depth and used < budget:
                for i, order in enumerate(orders):
                    if depth >= len(order):
                        continue
                    j = order[depth]
                    t = count_tokens(sents[i][j])
                    if used + t > budget:
                        # Skip rather than stop: one long sentence should not
                        # strand the remaining budget.
                        continue
                    kept[i].append(j)
                    used += t
                depth += 1

            out = [
                MemoryUnit(
                    unit_id=u.unit_id,
                    # Original order, so the only difference from lead-k is
                    # which sentences survive, not how the record reads.
                    text=" ".join(s[j] for j in sorted(k)),
                    session_id=u.session_id,
                    session_date=u.session_date,
                    session_index=u.session_index,
                    is_evidence=False,
                    roles=u.roles,
                    provenance=(u.unit_id,),
                )
                for u, s, k in zip(units, sents, kept) if k
            ]
            return renumber(out)
