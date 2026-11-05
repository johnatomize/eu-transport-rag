# The faithfulness judge against the owner's labels

Built by `make tables` from the judge's calibration runs in `runs/judge/` and the owner's labels in `data/faithfulness_labels.v1.jsonl` (#82, #83). The owner labelled 30 of F's dev answers: 28 supported and 2 not. A judge agrees on an answer when it calls it supported or not supported as the owner did; an answer it couldn't give a usable verdict on counts as a disagreement. *Same first sentence* counts, among the unsupported answers the judge caught, those where its first unsupported sentence is the owner's.

**The rule** (docs/DECISIONS.md, 2026-11-05): Haiku 5.5 is the judge if it agrees on at least 27 of 30 and catches every unsupported answer; otherwise Opus 5.5. With only 2 unsupported answers, catching them is a thin test.

No calibration run yet.
