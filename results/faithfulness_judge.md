# The faithfulness judge against the owner's labels

Built by `make tables` from the judge's calibration runs in `runs/judge/` and the owner's labels in `data/faithfulness_labels.v1.jsonl` (#82, #83). The owner labelled 30 of F's dev answers: 28 supported and 2 not. A judge agrees on an answer when it calls it supported or not supported as the owner did; an answer it couldn't give a usable verdict on counts as a disagreement. *Same first sentence* counts, among the unsupported answers the judge caught, those where its first unsupported sentence is the owner's.

**The rule** (docs/DECISIONS.md, 2026-11-05): Haiku 5.5 is the judge if it agrees on at least 27 of 30 and catches every unsupported answer; otherwise Opus 5.5. With only 2 unsupported answers, catching them is a thin test.

| Judge | Instructions | Run | Agreement | On supported answers | On unsupported answers | Same first sentence | Cost (US$) | Meets the rule |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| claude-haiku-5-5 | `b6b94f34888a` | `calibration-2026-11-05T20-47-52Z-claude-haiku-5-5.json` | 26/30 | 24/28 | 2/2 | 2/2 | 0.0139 | no |
| claude-opus-5-5 | `b6b94f34888a` | `calibration-2026-11-05T20-49-11Z-claude-opus-5-5.json` | 28/30 | 26/28 | 2/2 | 2/2 | 0.5693 | yes |

## Disagreements

### claude-haiku-5-5, instructions `b6b94f34888a`

Run `calibration-2026-11-05T20-47-52Z-claude-haiku-5-5.json`.

- **q040**, answer by claude-haiku-5-5. Owner: supported.
  Judge: not supported.
  Sentence 1, “The AETR applies instead of Regulation 561/2006 to international road transport operations that are undertaken partly outside the EU/EEA-type areas covered by the Regulation.”: “The cited Article 2(3) says the AETR applies instead of the Regulation to international road transport undertaken in part outside the areas in paragraph 2; the answer's phrase 'EU/EEA-type areas' is a loose paraphrase of 'areas mentioned in paragraph 2', which the cited text does not describe as EU/EEA, so the characterisation of those areas is not supported.”

- **q051**, answer by claude-haiku-5-5. Owner: supported.
  Judge: not supported.
  Sentence 4, “This note predates the 2020 Mobility Package, which changed the rules, so it is older than the current act.”: “The node is dated October 2012 and described as before the 2020 Mobility Package, but the passage itself says nothing about the Mobility Package or about whether it changed the rules; the claim that the package changed these rules is not in the cited text.”

- **q068**, answer by claude-haiku-5-5. Owner: supported.
  Judge: not supported.
  Sentence 4, “The 2002/15 Directive text is the original, never-amended version, and the other two sources are consolidated versions, so none of these is an older note that conflicts with the acts.”: “The cited nodes show the 2002/15 text as original and never amended and the other two as consolidated versions, but nothing in the cited text says anything about whether any source conflicts with or is an older note than the acts; the claim about the status of the sources is only partly stated by the place labels and the conclusion about conflicts is not in the cited text.”

- **q095**, answer by claude-haiku-5-5. Owner: supported.
  Judge: not supported.
  Sentence 3, “For a digital tachograph it means your driver card, plus any manual records and printouts from that period, plus record sheets for the same period when you drove an analogue vehicle.”: “The cited text covers manual records and printouts for a digital tachograph (Article 36(2)(ii)), but says nothing in the cited nodes about carrying the driver card or about record sheets for analogue vehicles driven alongside a digital tachograph, so those claims are not supported.”

### claude-opus-5-5, instructions `b6b94f34888a`

Run `calibration-2026-11-05T20-49-11Z-claude-opus-5-5.json`.

- **q040**, answer by claude-haiku-5-5. Owner: supported.
  Judge: not supported.
  Sentence 1, “The AETR applies instead of Regulation 561/2006 to international road transport operations that are undertaken partly outside the EU/EEA-type areas covered by the Regulation.”: “Article 2(3) supports the AETR applying instead of the Regulation to international operations undertaken partly outside the areas in paragraph 2, but the cited text never says what those areas are, so calling them 'EU/EEA-type areas' adds a definition the cited nodes don't give.”

- **q095**, answer by claude-haiku-5-5. Owner: supported.
  Judge: not supported.
  Sentence 3, “For a digital tachograph it means your driver card, plus any manual records and printouts from that period, plus record sheets for the same period when you drove an analogue vehicle.”: “The cited Article 36(2)(ii) covers only manual records and printouts; the driver card and the record sheets for analogue driving are not in any cited node.”
