"""Re-derive the constants scripts/model_write_cost.py pins, from mem0 itself.

model_write_cost.py hardcodes three numbers read out of mem0's source at tag
v2.0.18 -- the extraction system prompt's token count and two batch sizes that
v3 does not expose as configuration. Hardcoding is deliberate: that script has
to run offline, in CI, and in a clone that has never installed mem0. But a
pinned number with no way to re-derive it is exactly the kind of unfalsifiable
input this repo criticises elsewhere, so this script closes the loop.

It needs the optional extra:

    pip install -e ".[mem0]"
    python scripts/verify_write_cost_inputs.py

Without it the script exits 0 with an explanation rather than failing, so it is
safe to call from CI, where mem0 is deliberately not installed.

Exit codes: 0 verified or skipped, 1 a constant no longer matches its source.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_mem_eval.cost import count_tokens  # noqa: E402
from scripts.model_write_cost import (  # noqa: E402
    EXISTING_MEMORIES_TOP_K,
    LAST_K_MESSAGES,
    MEM0_VERSION,
    SYSTEM_PROMPT_TOKENS,
)

# The comment that separates the v3 ADD-only pipeline from the two-call
# extract/update loop the 2025 paper describes.
V3_MARKER = "V3 PHASED BATCH PIPELINE"


def main() -> int:
    try:
        import mem0
        from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT
        from mem0.memory import main as mem0_main
    except ImportError as e:
        print(f"mem0 is not installed ({e}), so the pinned constants cannot be "
              f"re-derived here. This is the expected state in CI and in a "
              f"clone that has not opted into the extra.\n\n"
              f"  pip install -e \".[mem0]\"\n\n"
              f"model_write_cost.py runs regardless; it is these constants "
              f"that stay unchecked.")
        return 0

    installed = getattr(mem0, "__version__", "unknown")
    failures = []

    # The version gate is not pedantry. v3 rewrote the extraction path, so a
    # prompt token count from any other release is measuring different code.
    if installed != MEM0_VERSION:
        print(f"WARNING: mem0 {installed} installed, constants were read from "
              f"{MEM0_VERSION}. Differences below may be real changes upstream "
              f"rather than errors here.")

    derived = count_tokens(ADDITIVE_EXTRACTION_PROMPT)
    ok = derived == SYSTEM_PROMPT_TOKENS
    print(f"{'ok ' if ok else 'FAIL'}  SYSTEM_PROMPT_TOKENS  "
          f"pinned {SYSTEM_PROMPT_TOKENS:,}  derived {derived:,}"
          f"  (ADDITIVE_EXTRACTION_PROMPT, cl100k_base)")
    if not ok:
        failures.append("SYSTEM_PROMPT_TOKENS")

    # The wheel being the v3 rewrite is what makes the whole write-cost model
    # about the shipped algorithm rather than the 2025 paper's, so the marker
    # is asserted rather than assumed.
    src = inspect.getsource(mem0_main).splitlines()
    marker = next((i for i, ln in enumerate(src) if V3_MARKER in ln), None)
    ok = marker is not None
    print(f"{'ok ' if ok else 'FAIL'}  v3 pipeline           "
          f"{V3_MARKER!r} "
          + (f"at mem0/memory/main.py:{marker + 1}" if ok else "NOT FOUND"))
    if not ok:
        failures.append("V3_MARKER")

    # "One LLM call per add(), regardless of message count" is the claim the
    # whole write-cost model rests on: it is what makes the cost a function of
    # caller batching rather than of how many facts the extractor emits. It is
    # also the one structural claim here that can be checked mechanically, by
    # counting the generation calls in the v3 block, so it is checked rather
    # than left as a comment.
    if ok:
        end = next((i for i, ln in enumerate(src[marker + 1:], marker + 1)
                    if re.match(r"    (async )?def |^class ", ln)), len(src))
        calls = [(i, ln) for i, ln in enumerate(src[marker:end], marker + 1)
                 if "generate_response(" in ln]
        one_call = len(calls) == 1
        print(f"{'ok ' if one_call else 'FAIL'}  one LLM call/add()    "
              f"{len(calls)} generate_response call(s) in the v3 block")
        for i, ln in calls:
            print(f"  mem0/memory/main.py:{i}: {ln.strip()}")
        if not one_call:
            failures.append("ONE_CALL_PER_ADD")

    # LAST_K_MESSAGES and EXISTING_MEMORIES_TOP_K are bare literals inside a
    # method body, so there is nothing to import and no assertion that would
    # mean anything -- "is 10 somewhere in this file" passes on almost any
    # file. Rather than dress that up as a check, print the lines that carry
    # them, scoped to the v3 block so the output is the handful of lines a
    # reviewer actually has to read.
    if ok:
        print(f"\nContext for LAST_K_MESSAGES={LAST_K_MESSAGES} and "
              f"EXISTING_MEMORIES_TOP_K={EXISTING_MEMORIES_TOP_K}, literals "
              f"in the v3 block that cannot be imported. Not asserted -- "
              f"read them:")
        window = [(i, ln) for i, ln in enumerate(src[marker:marker + 40],
                                                 marker + 1)
                  if re.search(r"\[-\d+:\]|(?:limit|top_k)=\d+", ln)]
        for i, ln in window:
            print(f"  mem0/memory/main.py:{i}: {ln.strip()}")
        if not window:
            print("  none in the 40 lines after the marker -- v3's slicing "
                  "may have moved; re-read it before quoting the model again.")

    if failures:
        print(f"\n{len(failures)} constant(s) no longer match mem0 "
              f"{installed}: {', '.join(failures)}. Update "
              f"scripts/model_write_cost.py and regenerate "
              f"results/write_cost_model.json before quoting it again.")
        return 1

    print(f"\nSYSTEM_PROMPT_TOKENS re-derived from mem0 {installed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
