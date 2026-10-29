# Decisions

Choices made along the way, with the reason and the alternatives considered. Newest first. Open questions are `decision-needed` issues; the answer is recorded here or in the design once the owner decides.

## 2026-10-29: gold nodes that are list items keep their lead-in out of the gold

**Decision:** a gold node stays the most specific node that holds the answer, even when it is a point in a list that means little alone, such as 2002/15 Art. 3(a)(1)(ii), "loading and unloading;", or 561/2006 Art. 13(1)(r). The sentences that introduce it come from the text around it instead: the review page shows them above the node, and a chunk that holds a list item without the paragraph it sits in carries them in its prefix (#33). Decided by the owner during the gold answer check (#17).

**Why:** 34 gold nodes in 39 questions are points in a list, and the owner flagged only two of those questions. Fixing the context in chunking treats all of them the same way, and keeps the gold exact for retrieval evaluation in D.

**Alternatives:** citing the parent paragraph as gold, which for 561/2006 Art. 13(1) means points (a) to (r) and blurs which point answers; adding a second node that gives context, such as 165/2014 Art. 34(5)(b)(ii) for loading and unloading, which supports the answer only indirectly.

## 2026-10-28: the question set and its dev/test split

**Decision:** `data/questions.v1.jsonl` holds 100 questions, split 60 dev and 40 test by a stratified seeded shuffle, and the split is frozen from the commit that adds them (#16, plan approved by the owner on the issue).

- **Mix:**

  | Type | Questions |
  | --- | ---: |
  | `single_fact` | 20 |
  | `definition` | 10 |
  | `exception` | 16 |
  | `multi_article` | 14 |
  | `cross_act` | 10 |
  | `unanswerable` | 16 |
  | `version_trap` | 14 |

  75 are paraphrased, in a driver's or fleet manager's words, and 25 are verbatim, in the act's wording. Every type has both.
- **How it was assigned:** the questions were grouped into strata by type and phrasing, in the order of `TYPES` and `PHRASINGS` in `src/rag/questions.py`. Each stratum was sorted by ID and shuffled with one generator, `rng = random.Random(16)`, shared by all strata in that order (`rng.shuffle(stratum)` for each), and the shuffled strata were laid end to end. Of every five positions, the second and fourth went to test. This gives exactly 60/40 overall and close to 60/40 within each type, and nobody chose which question went where. The three samples from #15 were included; their earlier `dev` was a placeholder.
- **Frozen:** questions added later go to dev. A test question found wrong in the owner's check (#17) is corrected in place, not moved. The validator fails if any type has no questions in either split, and a test checks the test split against `tests/snapshots/test_split.txt` and that every question after q100 is in dev.
- **Key facts** are written with digits, a decimal point and the unit: `9 hours`, `4.5 hours`, `7.5 tonnes`, `100 km`, even where the act writes "nine hours" or "7,5 tonnes". The only exceptions are `twice` and the times of day `00.00 on Monday` and `24.00 on Sunday`. One form for every fact keeps exact match predictable. Answers will still write numbers in other ways ("nine hours", "45-minute break"), so answer evaluation in G will have to normalise numbers before matching.
- **Why 60/40:** the test split is used once, for checkpoint G's final table. With 40 questions the 95% interval on a proportion is about ±15 points, against ±14 with 50, so the extra dev questions for tuning in D–G and for calibrating the judge cost little.
- **Version traps** cite the current text as gold and give the old answer in `notes`. Each old answer was checked against an earlier text fetched from CELLAR, outside the corpus, and the note names it:

  | Old text | Changes it was checked for |
  | --- | --- |
  | `02006R0561-20150302` | the 2020 changes to 561/2006 |
  | `02006R0561-20200820` | the 2024 changes |
  | `32014R0165`, the original act | the 2020 changes to 165/2014 |
  | `32006R0561`, the original act | the one change made by 165/2014 in 2014 |

  The traps come from:
  - **Reg. 2020/1054:** eleven traps.
  - **Reg. 165/2014:** one trap. The tradesman's exclusion became an exclusion for everyone, with a wider radius, and Reg. 2020/1054 extended it.
  - **Reg. 2024/1258:** two traps, coach drivers' split break and the 12-day derogation. Five have a guidance note in the corpus that gives the old answer:

  | Note | Old answer |
  | --- | --- |
  | guidance-1 (2007) | Art. 12 |
  | guidance-6 (2017) | rest on a ferry, two traps |
  | clarification-3 (2007) | 50 km where the act now says 100 km |
  | clarification-7 (2016) and guidance-5 (2013) | 28 days of records where 165/2014 now says 56 |

  These are the questions that test whether old notes cause wrong answers (see "guidance notes that predate the Mobility Package" below).
- **Unanswerable questions** are plausible questions from three groups: details the acts leave to Member States, neighbouring EU rules outside the corpus, and near misses, where the corpus touches the topic without answering it. Each was confirmed absent by a keyword search over the parsed text, and its `notes` gives the terms searched, any near misses found and, when known, where the answer actually lives. Phase 2 subjects (weights and dimensions, emissions, CO2, vehicle safety) were avoided, so the questions stay unanswerable when phase 2 joins.

**Why:** a split nobody chose by hand can't drift easier or harder, and freezing it before D has seen any result keeps the test split unseen until G.

**Alternatives:** 50/50, which narrows the test interval by about one point but leaves less for tuning; assigning by a hash of the ID, which is stable when questions are added but isn't stratified, so a small type could end up all in one split.

## 2026-10-28: the question schema

**Decision:** each line of `data/questions.v1.jsonl` is one question with the fields `id`, `question`, `answer`, `key_facts`, `gold_node_ids`, `type`, `phrasing`, `split`, `checked` and an optional `notes` (#15, plan approved by the owner). `src/rag/questions.py` validates it against the parsed corpus, and `make check` runs the validator.

- **IDs** are `q` and three digits, and are never reused or renumbered: a deleted question leaves a gap, so run logs stay comparable.
- **`key_facts`** lists the values that answer evaluation checks by exact match, written as an answer should give them (`"90 hours"`, `"45 minutes"`), and each must appear word for word in the gold answer. Without it, G would have to pull the numbers out of the gold answer sentence, which is fragile (design problem 4). It can be empty when the answer has no number or duration, as for most definitions and conditions, which G checks by normalised match.
- **`gold_node_ids`** are the most specific nodes that hold the answer. The validator rejects a whole act, chapter or note, and a node listed together with one of its ancestors. How a retrieved chunk matches them, for example through an ancestor, is for retrieval evaluation in D to decide.
- **Gold nodes may not lie in an amending article** (561/2006 Art. 26 and 27, 165/2014 Art. 45). Their wording has since been replaced, so it can't be the evidence for a current answer (design problem 1). The validator rejects them and doesn't just warn.
- **Rules per type:** an `unanswerable` question has no answer, key facts or gold nodes, and every other type has an answer and at least one gold node. A `multi_article` question has gold nodes in two articles or annexes or more; a guidance page counts as neither. A `cross_act` question has gold nodes in two acts or more; a guidance note doesn't count as an act. A `version_trap` question has non-empty `notes`.
- **Version traps** cite the current node as gold and put what changed and the old answer in `notes`, as free text. There is no field such as `trap_node_ids` for the old evidence.
- **No `lang` field** until checkpoint L. L can add one that defaults to `en`, without changing any existing row. Node IDs carry no language (see "document IDs" below), so a Swedish question can cite the same gold nodes.

**Why:** the schema fixes what retrieval and answer evaluation measure against, so the validator checks it against the current parser on every `make check`, not against a saved list of IDs.

**Alternatives:** pydantic for the schema, which adds a dependency for ten fields; a warning instead of an error for amending articles; a machine-readable field for a version trap's old evidence.

## 2026-10-28: guidance notes that predate the Mobility Package stay in, with their date

**Decision:** all 17 guidance and clarification notes stay in the corpus. Each has a `published` date and a `published_source` in `corpus/manifest.yaml`, and the parser puts the date on the note node (#26, option 1, chosen by the owner).

**The problem:** the acts are pinned to their consolidated versions of 31.12.2024, but 15 of the 17 notes are older than the Mobility Package (Reg. 2020/1054, 31 July 2020). Some may have been overtaken. Guidance-1 interprets 561/2006 Art. 12 without the second subparagraph that 2020/1054 added (driving up to one or two hours longer to reach home), and 11 of the notes cite Reg. 3821/85, which 165/2014 replaced. Like amending acts (design problem 1), an old note can bring outdated wording into retrieval.

**Where the dates come from:** no note prints its own publication date, so each date names its source.
- `letter`: clarification notes 1, 3, 4 and 5 record a Commission letter whose date is printed on page 1 ("letter of 12 November 2007"). That letter is when the position was taken. Clarification-1 gives only the month (`1993-02`).
- `pdf`: for the other 13, the creation date in the PDF's metadata. The note existed by then but may be older: clarification notes 2 and 6 carry 2012-10-08, the date the whole series was converted. Guidance-10 cites Reg. 2020/1054, so its date of 2022 is plausible.
- `commission-page`: for a publication date found on the Commission's page. No note uses it. The page, checked on 2026-10-28, gives only the date each file was uploaded: 20 July 2021 for 15 notes, even clarification-1 from 1993, which looks like the day the notes moved to the site, and 8 March 2022 for guidance-9 and guidance-10. Those dates would put every note after the Mobility Package. The page's links are the URLs in the manifest.

Only guidance-9 and guidance-10 (2022) are newer than the Mobility Package, and a test checks that.

**Why:** the dates are cheap, nothing still valid is lost (such as guidance-7 on "each period of 24 hours"), and answers can say how old the guidance is. Version-trap questions in the evaluation set (#16) then measure whether old notes actually cause wrong answers. If they do, the next step is option 2: a hand-checked note on each note that has been overtaken, which chunking (checkpoint D) can use to leave it out or rank it lower.

**When to check:** no issue tracks this yet. When checkpoint G's issues are created, one of them gets a criterion: look at the version-trap results and, if a pre-2020 note led to a wrong answer, open an issue for option 2. D's retrieval results give an earlier hint (whether an old note ranks above the act), but only G shows whether it changes the answer.

**Alternatives:** option 2 now, which means reading every note against the current text (a legal judgement for the owner to check) before knowing whether old notes cause wrong answers, and most notes are incomplete rather than wrong; leaving the pre-2020 notes out (option 3), which drops 15 of 17 notes, all the guidance on driving and rest times, and the material for version-trap questions.

## 2026-10-28: guidance notes are read with pdfminer.six, one node per page

**Decision:** each guidance or clarification note becomes a note node (`guidance-7`) with one node per page under it (`guidance-7:p3`), so that an answer can cite the page. The page number is the page's position in the PDF, counted from 1; where a note prints page numbers (guidance-10), they agree. `src/rag/pdftext.py` extracts the text with **pdfminer.six** (#13, plan approved by the owner on the issue).

- **Cleaning:** footnote markers are dropped, because "165/2014¹" would otherwise read `165/20141`. A marker is a digit at most three quarters the size of the other characters on its line: 6.5–8 pt next to 10–12 pt text. The footnote's own number at the foot of the page goes the same way, and its text stays. A text box in the top or bottom 15% of the page that repeats on every page of a note (a running header or footer) is kept only on page 1; in clarification-7 that is the header and the disclaimer. A box in that margin holding only the page number is dropped. Text inside figures, such as the labels of guidance-7's diagrams, is read like any other text. Whitespace is collapsed, and a word split over two lines with a hyphen is joined with the hyphen kept.
- **Legal basis:** the field naming the provisions a note interprets ("Article: 8 (2) and (5) of Regulation (EC) No 561/2006") is kept on the note in a form the reference scanner reads. Its references are scanned as part of page 1. References are scanned in the page text as written, so each can be found in it (the notes write "Article 13 (1) (d)", with spaces, and the scanner now reads that too). When the field names exactly one act in the corpus, that act is the one the note interprets, and references in the note that name no act point into it. A reference that names no act is left unresolved rather than guessed when the first act named after it in its sentence is a different one, as in "Article 14 (4) a of Regulation (EEC) No 3821/85". The page text itself is never rewritten.
- **Not extracted, and listed by `make parse`:** images, which are the timeline diagrams of guidance-7 pp. 2–6, the figures of guidance-10 pp. 3–4 and one image on guidance-4; and pages drawn as a diagram, table or form, meaning more than 100 drawn lines and boxes: guidance-7 pp. 5–6, guidance-9 p. 3 and guidance-10 p. 5. On those pages the text may not follow the layout. On guidance-10 p. 5, the table of nation codes comes out as a column of codes followed by a column of countries, so which code belongs to which country is lost. It can be rebuilt with an explicit override, like 2002/15's, if a question needs it.

**Why pages:** paragraph boundaries in a PDF are guesses from the layout and could move with a new library version, which would change IDs. Pages come from the pinned file itself. Chunking (checkpoint D) splits pages further, and each chunk keeps its page's ID for citations.

**Why pdfminer.six:** all 17 notes were extracted with three libraries and compared word by word (2026-10-27).
- pypdf splits words in the middle ("author ize", "diffi culties"), which would hurt keyword search and the exact-substring check on quotes.
- pypdfium2 is clean but glues footnote markers to the next word ("2020/10542sets", "codes4to").
- pdfminer.six gets word boundaries right and reports each character's font size, which is what makes footnote markers detectable.

It is MIT-licensed pure Python, but it depends on `cryptography` (for encrypted PDFs), which brings `cffi` and binary wheels. Every page of every note has a text layer, so no OCR is needed.

**Alternatives:** a node per labelled section (`guidance-1:approach`), which reads well, but the labels differ between notes and a section has no page number. Paragraph nodes (`guidance-7:p1.3`) are less stable for the reason above.

## 2026-10-27: node IDs, and what the parser leaves out

**Decision:** `src/rag/parse.py` gives every node an ID `<document ID>:<path>` (#12, plan approved by the owner on the issue).

- **Articles:** numbered and lettered enumerations go in brackets, the way the acts cite themselves: `561/2006:art6(3)`, `561/2006:art8(6)(a)`, `561/2006:art4(e)`, `165/2014:art34(5)(b)(iii)`, `2002/15:art3(a)(1)`. Chapters are nodes (`561/2006:chII`) but aren't part of article IDs.
- **Annexes:** headings and decimal numbers are joined with dots and lettered points go in brackets: `165/2014:anxI.III.c.4`, `165/2014:anxI.III.a.3.1`, `165/2014:anxI.I(a)`.
- **Not nodes:** subparagraphs and indents (`—`) stay in their parent's text, so "paragraph 2, first subparagraph" resolves to `art8(2)`.
- **Left out:** the 16 recitals of 2002/15. The consolidated acts have none, and recitals aren't binding. They can be added later as `2002/15:recN` without changing any other ID.
- **Flagged:** articles that amend another act (561/2006 Art. 26–27 and 165/2014 Art. 45) are parsed normally, with `amending: true`. Their text holds wording that was later replaced: Art. 45 has the 2014 wording of 561/2006 Art. 3(aa), which the Mobility Package has since replaced. Chunking (checkpoint D) decides whether to index them. The default proposed there is to leave them out.

**Why:** gold answers and citations cite these IDs, so they follow the structure a reader sees in the act and survive changes around them. Leaving chapters out of article IDs means an ID survives a chapter being restructured. Brackets for article enumerations make the IDs read like citations. Annexes are numbered by headings and decimals ("point 4 of Part VI"), so dots fit them better.

**How it is held stable:** `tests/snapshots/node_ids.txt` lists every ID, and a test compares the parse with it, so any change to an ID shows up in a pull request diff. A duplicate ID is an error, never given a silent suffix. Anything the parser doesn't understand is listed by `make parse` and checked against a reviewed list in the tests.

**Also decided:** the parser uses only the standard library (`xml.etree` for the XHTML, `html.parser` for the HTML 4 of 2002/15), so it adds no dependency. 2002/15's flat HTML gets its structure from the labels. One sentence can't be placed that way, so the parser has an explicit override for it, which is safe because the file is pinned by hash.

**Alternatives:** dots throughout (`art8.6.a`), which doesn't read like a citation; subparagraphs as nodes (`art8(6)-2`), which the acts cite rarely and which shift when a subparagraph is inserted; dropping amending articles in the parser, which would hide a choice that retrieval can measure instead.

## 2026-10-27: document IDs, and pinned versions

**Decision:** each document in `corpus/manifest.yaml` has a short ID that prefixes all its node IDs: `561/2006`, `165/2014` and `2002/15` for the acts, and `guidance-1`…`guidance-10` and `clarification-1`…`clarification-7` for the Commission's notes (#11). The notes keep the Commission's own numbering, which has two separate series. IDs carry no version and no language; the manifest pins the version, and the file name carries the language.

**Why:** node IDs end up in the gold answers, which the owner checks by hand, so they must survive changes around them. The Swedish texts (checkpoint L) need the same IDs as the English ones. EU drafting inserts new paragraphs as `8a` rather than renumbering, so most IDs would also survive a newer consolidated version.

**Trade-off:** moving to a newer consolidated version (re-pinning, with `scripts/fetch_corpus.py --pin`) is a deliberate step: compare the parsed versions, then re-check the gold answers whose nodes changed. A re-pin isn't expected during this project, so there is no automatic check for newer versions.

**Alternatives:** the version in the ID (`561/2006@20241231:art6(3)`), which breaks every ID on a re-pin; a type letter (`R561/2006`), which isn't needed because the acts in the design don't collide.

## 2026-10-27: Docker runs in Colima

**Decision:** the local Docker engine is Colima, installed with Homebrew (`brew install colima docker docker-compose`), and isn't started at login (`colima start` when needed).

**Why:** it is free, has no GUI, and installs the same way as the rest of the toolchain. Compose (#7) is all this project needs from Docker until checkpoint H.

**Alternatives:** Docker Desktop, the standard GUI app (free for personal use, heavier).

## 2026-10-27: raw corpus files are committed to git

**Decision:** the documents that `scripts/fetch_corpus.py` downloads are committed under `corpus/raw/` (#9).

**Why:** tests and the CI retrieval quality gate then run offline and are reproducible, and the corpus survives CELLAR changing or dropping a document. Both phases together are about 10–13 MB (measured 2026-10-27: phase 1 about 1–3 MB including the guidance PDFs, phase 2 about 9.5 MB), well below the 50 MB limit set in #9 for committing.

**Trade-off:** every re-pinned consolidated version stays in git history for good. The largest is Reg. 2018/858 at 5.9 MB, so a few updates over the project's life are fine, but frequent re-pinning is not.

**Alternatives:** a GitHub release asset (keeps the repo small, but needs a new release whenever the manifest changes), or not storing the files at all (CI depends on CELLAR being up, and a removed document breaks the rebuild).

## 2026-10-27: gitleaks runs as a local pre-commit hook

**Decision:** `.pre-commit-config.yaml` defines a `repo: local` hook that runs the installed `gitleaks` on staged changes, rather than the hook published in the gitleaks repository.

**Why:** the published hook builds gitleaks from source with Go, which means downloading a Go toolchain on first use. The local hook needs only the `gitleaks` binary and no network when committing.

**Trade-off:** every clone needs gitleaks installed (`brew install gitleaks`), and the version isn't pinned in the repo. CI (checkpoint J) will install a pinned version.
