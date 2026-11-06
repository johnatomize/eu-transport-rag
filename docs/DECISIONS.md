# Decisions

Choices made along the way, with the reason and the alternatives considered. Newest first. Open questions are `decision-needed` issues; the answer is recorded here or in the design once the owner decides.

## 2026-11-06: the finalists for the test split

**Decision** (owner, 2026-11-06, on #85's proposal; for #86): three versions answer the 40 test questions once, each with Opus 5.5 through the batch service and judged as on dev: Haiku 5.5 reranking hybrid search's top 20 paragraph chunks, top 5; hybrid search over article chunks, top 3; and the base, hybrid search over paragraph chunks, top 5.

- **Haiku reranking:** on dev it refused 3 answerable questions against the base's 8 and cited 87% of gold passages against 80%, at the base's cost. Four of the five questions it won back have no key facts, so the owner's hand grades on the test split (#88) are what can show whether those answers are right.
- **Article chunks:** the fewest retrieval misses on dev (4 against 7, with windows) and the best automatic correctness (0.87), with fewer unsupported answers than windows (6 against 9).
- **The base:** the most faithful on dev (40 of 42 supported), and the setup the other two are compared with.

**Alternatives:** 200-word windows in place of article chunks (the same correctness and misses, but 9 unsupported answers); top 10 in place of the base, which would leave the test table without the setup the design compares from; top 3, the weakest on dev.

## 2026-11-06: the spending cap rises to US$100, with a limit per paid issue

**Decision** (owner, 2026-11-06, on #85's estimate): the project's cap in the spend log rises from US$20 to US$100. Answers, judging and Haiku reranking are paid from the owner's US$100 of Anthropic credits, so the total may pass the original US$20. Each paid issue still gets its own limit, approved by the owner after its dry run, as a guard against a run that gets stuck or repeats: US$20 for #85's dev answer and judging runs. Before each run in an issue, its spending so far plus that run's most-it-can-cost must stay within the issue's limit; otherwise the work stops and the owner is asked. The paid tools keep checking the project cap, as before.

**Alternatives:** keeping US$20 for the whole project, which #85's cautious estimate (US$14.23 expected, before the test run) would nearly use up.

## 2026-11-06: Haiku 5.5 reranks the answer run, in place of Qwen

**Decision** (owner, 2026-11-06, after #99's comparison; changes the reranked version of #78 and the owner's note on #85 of the same day): the reranked dev answer run in #85 uses Claude Haiku 5.5 reading hybrid search's top 20 paragraph chunks together (`hybrid-paragraph-haiku20`), not Qwen3-Reranker 0.6B. Only the best reranker from #99 goes into the answer runs, and a reranker that lowers recall is not worth answering with. #85 still has five new Opus 5.5 runs: top 3, top 10, article chunks, 200-word windows and Haiku reranking.

- **Why Haiku:** on the 50 answerable dev questions it raised recall@5 from 0.76 to 0.80 and recall@3 from 0.58 to 0.78, lost no question to search alone, and won back all of Qwen's two-passage losses. Its top 5 is no longer than search's, so answering from it costs no more, and reranking adds about US$0.00066 a question, about 3% of an Opus answer.
- **No new paid reranking:** Haiku's orders for the 60 dev questions are already cached in `reranks/` from #99, so the answer run reads them and pays only for answers and judging.

**Alternatives:** keeping Qwen and adding Haiku as a sixth run (about US$1.90 more, to confirm in answers that Qwen's weaker passage lists matter); Haiku and Jev in place of Qwen (Jev matched Haiku's recall@5 but shows the model 15% more text, lost q060 to search alone and would need a second outside provider in the service); keeping Qwen alone, as planned in #78.

## 2026-11-06: ways to improve reranking, and how they are compared

**Decision** (#99; the five candidates are the owner's, the settings below were fixed before any was measured): each candidate changes one thing from Qwen3-Reranker 0.6B reordering hybrid search's top 20 paragraph chunks, the setup the answer runs rerank (#85), and is measured on the dev split as a retrieval run. With 50 answerable questions, trying many settings and keeping the best would only fit the dev split, so each candidate has one setting.

- **Blend** (`hybrid-paragraph-rerank20-blend`): the top 20 are ordered by reciprocal rank fusion of search's order and the reranker's, each giving 1 / (60 + rank), the constant hybrid search already uses (`Rerank(blend=…)` in `src/rag/retrieve.py`). Ties keep search's order. No model run: Qwen's cached scores are reused.
- **Another instruction** (`-rerank20-needed`): Qwen is asked "Given a question about EU road transport law, judge whether the answer relies on this passage. A question may need several passages, such as a rule, its exceptions and the definitions it uses; each of them counts" (`other_instructions` in `config/rerank.yaml`). Its scores share Qwen's cache file, since each key holds its instruction, and the main instruction's keys are unchanged. Scoring the 1,200 pairs on the Mac (OK'd by the owner on 2026-11-06) is logged in `runs/rerank-local/`, the source of the local time per question.
- **Cited passages after reranking** (`-rerank20-refs`): cross-reference expansion (2026-11-01) wraps the reranked list, so each reranked hit is followed by the chunks it cites, inside the top k as before. Every hit's references are followed, since a reranker's score of 0 doesn't mean nothing matched.
- **Jev** (`hybrid-paragraph-jev20`), TypeSafe's model, pinned to `jev-1.14.0`: one request a question, with the question as the state and one yes-or-no question (a Noul) a passage, "Does a correct answer to `question` rely on this passage…?", with the passage in that question's own text and criteria for yes and no (`src/rag/hosted_rerank.py`). Its probability of yes is the score. Nouls in one request can't see each other, so each is cached per pair, like Qwen's. US$0.042 per million input tokens, output free (TypeSafe's models page, read 2026-11-06); the key is `TYPESAFE_API_KEY` in `.env`. Plain HTTP with Python's own library, so no new dependency. Only a 429 is tried again; a server error, a timeout or a dropped connection isn't, since TypeSafe doesn't say whether it was billed. A reply from any model but the pinned one is refused and kept in the log, never cached. The most a run can cost is taken at twice the estimated tokens, since Jev's tokenizer isn't published.
- **Haiku 5.5 reading the top 20 together** (`hybrid-paragraph-haiku20`): one call a question, the passages numbered in search's order, asking for the numbers of those a correct answer relies on, most important first, every passage a question needs before any that is only related. Low effort and at most 2,000 output tokens (`config/rerank-hosted.yaml`), since it must stay quick enough for every question. Listed passages come first in its order and the rest follow in search's order. A reply that can't be used (cut off, refused, a number repeated or out of range) leaves search's order, as a service would, and is cached with its reason so it isn't paid for again. Replies are cached after every call, also when the run log can't be written. Each call's estimate allows 1,500 tokens on top of the text for the API's own input, from the first run's bill (about 1,350 a call), and a prompt that may pass `max_prompt_tokens` is refused.
- **Paid runs** are dry runs until `make rerank WITH=jev|haiku RUN=1`, which checks prices read in the last week, the most the run can cost against the cap (counting `runs/rerank/` until the spend log names each log), and the key. Every call is logged as soon as it is paid for. Their answers are committed in `reranks/`, so the tables rebuild for free and nothing is paid for twice. A change of model or wording changes the reranker's name and its cache file, so old and new scores never mix.
- **Comparison:** `results/retrieval.md` gives every reranked run's metrics, its question-by-question wins and losses at recall@5 against hybrid search alone and against Qwen's top 20, with an exact sign test, and what a question cost each reranker, from its scoring logs. What a deployed service would have to carry or call is in `PROGRESS.md`.

**Alternatives:** tuning the blend constant or the instructions on the dev split, which with 50 questions would fit noise; one Jev request a passage, as TypeSafe's reranking example does, which sends the question 20 times and costs more; giving Haiku the passages in a shuffled order, which would hide search's view that blending shows helps; Qwen3-Reranker 4B, left out unless the owner adds it (a larger download).

## 2026-11-06: how the reranker reorders search results

**Decision** (#84): `Rerank` in `src/rag/retrieve.py` wraps any retriever: Qwen3-Reranker 0.6B (#77) reads the question with each of the wrapped retriever's top N passages, and those N are sorted by its score; the passages beyond N keep their order. The model, its settings and its scores are in `src/rag/rerank.py` and `config/rerank.yaml`.

- **The score:** the model is asked, in the form of its model card, whether the passage meets the question and an instruction, and its score is the log-odds of "yes" (its logit for "yes" minus that for "no"). That orders passages as the card's probability does, but two very likely passages don't both round to 1.0 and tie. Ties keep the wrapped retriever's order. Each pair is scored alone, without padding, so a score never depends on its batch, and an input over the model's 8,192 tokens is refused rather than cut short.
- **The instruction** is the embedding model's, word for word: "Given a question about EU road transport law, retrieve the passages of legislation and Commission guidance that answer it". The passage is the chunk's indexed text, heading path included.
- **Pinned and cached, as the embedding model is:** revision `e61197ed4502…` on Hugging Face (Apache 2.0), float32 on the Mac's GPU. Every score is kept in `reranks/<model setting>.jsonl`, keyed by the model, revision, device, instruction, question and passage, and committed, so `make retrieval-runs` and `make tables` never run the model and a missing score stops a run before any log is written. `make rerank` reports what is missing; `make rerank RUN=1` scores it on the Mac.
- **N = 20 and 50, over hybrid search on paragraph chunks**, the setup #78 reranks: two runs, `hybrid-paragraph-rerank20` and `-rerank50`, whose logs name the reranker setting and N. Before reranking, the top 20 hold every gold passage for 41 of the 50 answerable dev questions and the top 50 for 48, against 38 for the top 5: that is all a reranker can gain. Scoring the top 50 for the 60 dev questions is 3,000 pairs, and covers N = 20 too.

- **The answer runs rerank the top 20** (#85): the top 50 put the same passages in the top 5, and reranking 20 passages a question is about 2.5 times faster, which matters for the service.
- **Cache keys are stored as `pair`**, not `key` as in the embedding cache: the secret scan's generic-API-key rule reads a long hash after the word "key" as a secret, and renaming the field avoids it without widening the scan's exception.

**Alternatives:** batches of pairs, faster but with scores that shift slightly with what they are padded alongside; the probability as the score; reranking every strategy, left for later since #78 compares reranking on paragraph chunks only; computing scores inside `make retrieval-runs`, which would make it need the model's libraries.

## 2026-11-06: the Faithful column in the answer results

**Decision** (#95): `results/answers.md` gives each scored answer run a **Faithful** column, the share of its answers the chosen faithfulness judge found supported, from the judge's whole runs in `runs/judge/run-*.json` (`src/rag/tables.py`).

- **Which judge counts:** only runs by the chosen judge, `judge_model` in `config/generation.yaml` with the instructions the code holds now (`judge_prompt_version`), the pair that was calibrated in #83. A run by another model or with other instructions is listed under the table as not counted, with the reason, so a paid run is never silently dropped. If the instructions change, every earlier run moves to that list until it is judged again, since the new instructions haven't been measured against the owner's labels.
- **Over what:** only answers that passed the check and aren't refusals are judged. The share is taken over the answers the judge gave a usable verdict on, with its Wilson 95% interval, and the cell says how many: "0.93 (0.82–0.98), 42 judged". An answer the judge gave no usable verdict on is shown apart ("1 without a verdict"), not counted as supported or as unsupported.
- **Matching:** a judge run names the answer run it judged by its file name, so `make judge LOG=…` takes the bare file name, not a path. Each judgement must be of an answer of that run that passed the check, only once, with that answer's text, compared word for word so a change in how sentences are split doesn't matter.
- **Unfinished:** a judge run that is missing any judgeable answer of its run, such as a call that got no reply, is listed with what stopped it. Judging the answer run again finishes it: the finished run counts and the unfinished one stays listed, with its cost still in the spend log. With no finished run, the cell shows "unfinished". Two finished runs of one answer run are refused, as two answer runs of one setup are; once both are in the spend log, the one not to count is moved out of `runs/judge/`. A run with no judge run shows "not judged". The Runs table names each run's judge log and what judging it cost.

**Alternatives:** counting a missing verdict as unsupported, which would mix the judge's failures into the answers' faithfulness; counting judge runs by any model, which would put an uncalibrated judge's numbers beside the calibrated one's; a separate faithfulness table, which the owner's request for a column in the same table doesn't need.

## 2026-11-06: runs through the batch service

**Decision** (#79; what batch runs show for speed was answered by the owner in a working session): `make answers BATCH=1` and `make judge … BATCH=1` send a run's calls as one batch through Anthropic's Message Batches API, which answers within 24 hours, mostly within one, at half the price (`src/rag/batch.py`). Each request is exactly the call a direct run sends; only the way it travels changes.

- **Same checks, batch prices:** the dry run prints the cost at the batch prices in `config/generation.yaml`, and a paid run needs `RUN=1`, prices read in the last week and a most-possible cost, every answer at its full output limit, within the cap.
- **The batch's ID is in the run log before anything else happens:** the log is written before the batch is created, and again as soon as the API returns its ID. The log's `batch` holds the ID, a status (`sending`, `submitted`, `collected`, `not_sent` or `unknown`) and, for each call, its `custom_id` (its place in the run and its question, such as `007-q012`, unique even when one question's answer is judged twice) and a fingerprint of the exact request.
- **Picked up, never sent twice:** a run stopped while it waits leaves its batch `submitted`. Running the same command again finds the log whose requests have the same fingerprints and waits for that batch instead of sending another, with no price or cap check, since nothing new is paid. Creating the batch is tried again only after "too many requests" (429), which creates nothing: a server error, a timeout, a dropped connection or Ctrl-C while the request is out may have stored the batch without its ID reaching the log, so the status becomes `unknown`, and the same command refuses to send it again until someone has looked on the Console's Batches page and, finding none, set the status to `not_sent` by hand. A direct run of the same calls is refused too while their batch may still be paid for. Reading a batch's progress and results costs nothing, so a dropped connection there is simply tried again.
- **Spend:** until its results are in its log, a batch is counted at the most it can cost, so a run started meanwhile can't pass the cap.
- **Results** are matched to calls by `custom_id`, in any order, and checked against the batch before any is recorded. A reply is recorded as a direct run's, at the batch price. A request that errored, expired or was canceled, or has no result, is listed in `failed` with the reason and scored as no reply, never as an answer. Such requests aren't billed.
- **No time per answer (owner's choice):** a batch's replies have no time of their own, as an answer that took four seconds and one that queued for forty minutes look alike, so `seconds` is empty and the results table shows "–" for a batch run's speed. Speed comes from direct runs: F's Opus and Haiku runs. A variant can still be timed later with a few direct questions.
- **The run log keeps a direct run's form,** plus `batch`, so scoring and tables don't care how a run was sent.

**Alternatives:** timing ten questions of each variant directly, about US$2 more; running every variant directly, about US$4–5 more, which leaves less of the cap for the test run; picking a batch up by naming its log, which a mistyped name could turn into a second batch; prompt caching inside the batch, not worth it for a 600-token instruction.

## 2026-11-05: Opus 5.5 judges faithfulness

**Decision** (#83, by the owner's rule of the same day below): Opus 5.5 is the faithfulness judge, `judge_model` in `config/generation.yaml`, with the instructions of version `b6b94f34888a`.

- **Measured:** on the owner's 30 labelled answers, Opus agreed on 28 (26 of the 28 supported, both unsupported) and Haiku on 26 (24 and both). Both found the owner's first unsupported sentence in q001 and q022. Haiku misses the rule's 27 by one, so Opus is the judge. Full results and every disagreement are in `results/faithfulness_judge.md`.
- **The two answers both judges call unsupported against the owner's label** (q040, q095, both Haiku's) add details the cited nodes don't hold: what the AETR's "areas" are, and the driver card and analogue record sheets among what must be carried. The labels stay as the owner made them: changing a label after seeing the judges' verdicts would make the agreement depend on them.
- **Cost:** judging the 30 cost US$0.57 with Opus and US$0.014 with Haiku; at Opus's rate a dev run's answers cost about US$0.80–0.95 to judge, half in batches.

**Alternatives:** Haiku, about 40 times cheaper but under the agreement the owner set; changing the instructions and measuring again, which would tune the judge on the labels it is measured against.

## 2026-11-05: the faithfulness judge

**Decision** (#83, plan agreed with the owner in chat): how `src/rag/judge.py` judges whether every claim in an answer follows from the text it cites, and how it is measured against the owner's labels (#82).

- **What the judge reads** is what the owner read on the labelling page: the question, the answer split into numbered sentences by the same function as the labels, and for each cited node its place and version, the headings above it, the text that introduces it, its full text and the words the answer quoted from it. Each node appears once, with all its quotes. It never sees the gold answer or the owner's labels and notes.
- **What it is told "supported" means** is the owner's definition: every claim follows from the cited nodes' full text, not only the quotes; simple arithmetic on numbers that are all in the cited text counts; correctness and completeness aren't judged; saying the passages don't cover something is not a claim. Question, answer and cited text are material to check, never instructions, and text that would open or close one of the prompt's tags is made harmless, as in the answer prompt.
- **What it gives back:** for every sentence, its number, a one-sentence reason, and *supported*, *not supported* or *no claim*, as structured output. A reply that misses a sentence, judges one twice, or is cut off gives no verdict. An answer is supported when no sentence is not supported, and its first unsupported sentence is the first that is.
- **Agreement** is counted on the answer's verdict, supported or not, as the owner labelled it. A missing or unusable verdict counts as a disagreement. The report gives agreement overall, on the supported and on the unsupported answers, and, for information only, how often the judge's first unsupported sentence is the owner's. `make tables` writes it to `results/faithfulness_judge.md` from the calibration run logs and the labels, with every disagreement and the judge's reasons.
- **Keeping it honest:** the judge's instructions are fixed and fingerprinted (`judge_prompt_version`) in every run log. Before calibration, the judge may be tried out only on answers the owner didn't label (`make judge … ONLY=…` refuses the labelled ones), to catch broken wiring, never to tune it towards anyone's verdicts. If the instructions change after a calibration run, both runs stay in the table.
- **Settings:** the judge runs through the answer client, with the same effort (medium), output limit (4,000 tokens), prices, cap check, retries and run logs (`runs/judge/`, counted by the spend check). Batches follow once #79 is done.

**Alternatives:** judging the answer as a whole, which would not show which sentence goes beyond its sources or let the owner check the judge's reasons; giving the judge only the quoted words, which would mark supported claims as unsupported when the rest of the node holds them, and judge differently from the owner; counting agreement on the first unsupported sentence too, which two unsupported labels can't measure.

## 2026-11-05: choosing the faithfulness judge when few labels are unsupported

**Decision** (owner, answering #93; changes the judge rule of 2026-11-04 for #76): the owner's labels (#82) mark 28 of the 30 answers supported. The two that aren't are both Haiku's: q001, which names a 56-hour weekly limit the cited text doesn't contain, and q022, which uses the Annex D country codes without citing Annex D. A judge that called every answer supported would agree on 28 of 30 and pass the old rule.

- **The rule:** Haiku 5.5 is the judge if it agrees with the owner on at least 27 of 30 answers *and* marks both q001's and q022's Haiku answers unsupported. Otherwise Opus 5.5 is the judge.
- **What is reported:** both judges' agreement overall, on the 28 supported answers and on the 2 unsupported ones, each given separately. The results say that two unsupported answers are a thin test of whether a judge catches them.

**Alternatives:** keeping the old rule, which can't tell a useful judge from one that always says yes; labelling more answers by the same fixed draw, about an hour of the owner's reading per 15 questions for perhaps one or two more unsupported answers; picking likely-unsupported answers by hand, which would make the agreement rate depend on the choice.

## 2026-11-04: which answers the owner labels for faithfulness, and what "supported" means

**Decision** (owner, for #82): the owner labels 30 of F's dev answers on a review page, and the faithfulness judge (#83) is measured against those labels. Code in `src/rag/faithfulness.py` and `scripts/faithfulness_labels.py`.

- **What "supported" means.** An answer is supported when every claim in it follows from the full text of the nodes it cites, not only the words it quotes. Simple arithmetic on cited numbers counts as supported when every number in the sum is in the cited text: in F, Opus's "at most 34 hours" for q001 follows from the quoted 90 hours over two weeks and 56 in one. Whether the answer is correct or complete is not part of the label, and saying that the passages don't cover something is not a claim. When an answer is not supported, the owner marks its first unsupported sentence.
- **Which answers** (fixed before anyone read them for this): 15 dev questions that both of F's runs, Opus 5.5 and Haiku 5.5, answered with a reply that passed the check, and both answers to each. The owner reads each question once and compares two answers on it. The 15 are spread over the answerable types by fixed quotas: 4 single-fact, 3 exceptions, 3 version traps, 2 multi-article, 2 definitions and 1 cross-act. They are drawn with the seed `faithfulness-labels-v1`, one draw per type, from the 39 questions that qualify. q001 is always included. The draw gives q001, q002, q012, q021, q022, q023, q037, q040, q045, q050, q051, q068, q088, q095 and q097.
- **Blind.** The page calls a question's two answers A and B, in an order drawn with the same seed, and never names the model, so that neither model is read more kindly.
- **Sentences** are split by one function, used by the page and later by the judge, so "the first unsupported sentence" means the same to both. It splits at line breaks, and after a full stop, question mark or exclamation mark followed by a capital, a digit or an opening quote or bracket. It doesn't split after "e.g." and similar abbreviations, after "Art." or "No." when a number follows, inside decimals, or after a list item's own number at the start of a line.
- **Labels belong to one build of the page.** The page keeps labels under a fingerprint of the questions, the letters and the answers it shows, and puts it in its export. Labels saved from a page built from other answers are refused.
- **The labels** go in `data/faithfulness_labels.v1.jsonl`, one answer per line, with the run log's file name, the verdict, the first unsupported sentence's number and text, and the owner's note. `make check` passes while the file doesn't exist, and once it does, every chosen answer must be labelled exactly once and each sentence must match the answer.
- **Kept from the judge.** The labels and the owner's notes never go into the judge's prompt or its examples. While the judge is developed, only answers outside these 30 are used, so the agreement measured in #83 is honest.

**Alternatives:** 30 answers drawn without pairing, which means reading 30 questions instead of 15; choosing answers by hand to include tricky ones, which would make the agreement rate depend on that choice; showing model names, which saves nothing and risks a kinder reading of one model; strict literal support, under which the correct sum in q001 would count against the answer although the system is meant to help a driver apply the rules.

## 2026-11-04: how two answer runs are compared

**Decision** (#81): `results/answers.md` compares runs question by question, with an exact two-sided sign test, in `src/rag/answer_eval.py` and `src/rag/tables.py`.

- **A question is right** when it has key facts and the answer is correct, or it is unanswerable and was refused. Answerable questions without key facts are left out, since code can't tell whether their answers are right (decision of the same day on what G measures); on the test split the owner's grades (#88) can add them later. A question whose call got no reply in either run is left out too: a failed or expired request says nothing about which version is better, and it is counted in its own column.
- **Each run is compared with the base setup** (Opus 5.5, hybrid search over paragraph chunks, top 5), since each dev version changes one thing from it; if it was run more than once, the latest run is the base. A split without the base run compares every pair.
- **The sign test** counts the questions only one run gets right, both ways, and gives the chance of a split at least that lopsided if neither run were better. Questions both get right or both get wrong say nothing and are only shown.
- **Many comparisons:** a note under the table gives their number and the Bonferroni threshold (0.05 divided by it), so one small p among several isn't over-read. No correction is applied to the p values themselves.
- **Runs are named by their settings**, such as "opus-5-5, hybrid paragraph top 5", with effort, output limit and prompt version added when two runs would otherwise share a name. Two finished runs with all the same settings on one split are refused, so a rerun must replace the earlier log rather than sit beside it. A run that hasn't answered every question of its split is listed but not scored.

**Alternatives:** comparing every pair of the seven dev runs (the base, five versions and Haiku), 21 tests of which the six against the base answer the design's questions; counting correctness alone, which would leave out whether a version still refuses the 10 unanswerable questions; McNemar's chi-squared test, which approximates the same sign test badly at the small counts seen here.

## 2026-11-04: how an answer run is scored

**Decision** (#80): the rules `src/rag/answer_eval.py` follows to score an answer run log, without calling a model.

- **What the reader saw:** each question ends as an answer that passed the check, a refusal that passed it, "could not answer reliably" (the reply failed the check), or no reply (the call failed, which happens when a request in a half-price batch fails or expires; a direct run stops at its first failed call). A run that has nothing for some question of its split is refused, so an unfinished run is never compared with finished ones.
- **Correctness** is scored on answerable questions with key facts: correct when the answer passed the check and states every key fact. A refusal, an unreliable answer or a missing reply counts as not correct.
- **A citation finds a gold node when it cites that node or a node inside it**, and not when it cites a node above it. The check has already made sure each quote is word for word in the cited node's own text. So a quote from point (b) of 561/2006 Art. 8(6) lies inside a gold Art. 8(6), while a citation of Art. 13(1) for a gold point 13(1)(r) quotes the paragraph's own words, its lead-in, which the gold leaves out (decision of 2026-10-29). This is retrieval evaluation's rule, applied to the quoted words.
- **Precision** is the share of an answer's distinct cited nodes that find a gold node, averaged over the answerable questions that were answered; a node quoted twice counts once. **Recall** is the share of gold nodes found, averaged over every answerable question, so a refusal finds none. Precision reads strictly: the gold is the smallest node that holds the answer, so citing the next paragraph for context, as Opus often does, lowers it even when that paragraph is relevant. Whether a citation supports its claim is the faithfulness judge's job (#83).
- **Refusal accuracy** is the share of unanswerable questions refused; **false refusals** the share of answerable questions refused. "Could not answer reliably" and no reply are their own shares, counted as neither.
- **Tracing:** an answerable question that is not correct, refused, unreliable or without a reply is a *retrieval miss* when none of its gold nodes was among the chunks the model was shown (a chunk holds a gold node when it holds it or a node inside it), and a *generation error* when at least one was. An unanswerable question answered is a generation error. An answer to a question without key facts is not traced, since code can't tell whether it is right; the owner's hand grades of the test answers to those questions (#88) will be read in with the tables. A question whose call got no reply is not traced either, since the log doesn't say what search found.
- **Cost, tokens and seconds** are means over the questions that got a reply, unreliable ones included, since they were paid for.
- **Intervals:** 95%, Wilson for shares and a seeded percentile bootstrap over questions for means, reusing retrieval evaluation's code; every measure is given overall, by phrasing and by type. A test run is scored only with the explicit flag.

**Alternatives:** counting a citation of the node above a gold node as finding it, which would credit a quote of a lead-in; recall over answered questions only, which would let a system that refuses more look better; counting each quote rather than each cited node in precision.

## 2026-11-04: matching key facts

**Decision** (#29): `src/rag/keyfacts.py` decides by code whether an answer states a key fact. The answer and the fact are both broken into words and numbers, with numbers and units spelled one way, and the fact's words must appear in the answer in order, as whole words.

- **What is forgiven:** number words up to the hundreds ("seven calendar days", "fifty-six hours", "four and a half hours", "a hundred km"), with "half an hour" as `0.5 hours` and "an hour and a half" as `1.5 hours`, never `1 hour`; "a" or "an" before a unit ("within an hour"), except after "once", "twice", "times" or another unit, where it is a rate ("twice a week", "9 hours a day"); the act's decimal comma (`7,5 tonnes`), a thousands comma and `½`; a number joined to its unit ("a 45-minute break"); units in the singular or plural, short ("9h", "45 mins") or spelled out ("kilometres" for `km`); times with a colon (`00:00`); dates written month first or with an ordinal ("July 1st, 2026"), while an ordinal elsewhere ("the 4th week") is not a count; leading and trailing zeros. Dashes, quote marks and invisible characters are handled as in the check of every answer.
- **What is not:** a different unit ("9 minutes" for `9 hours`), a number that only contains the fact's digits ("19 hours", "4.5 hours" for `5 hours`), and any other word. "six hours in a row" doesn't state `6 consecutive hours`, and "four weekly rests" doesn't state `4 weekly rest periods`. Forgiving synonyms would need a list that grows with each answer seen, and the gold facts are frozen.
- **On F's two dev runs** (#66), a plain substring match found every key fact in 27 of the 56 answered questions that have key facts; this match finds them in 51. Of the five left, two answers really leave figures out (Opus says the weekly working-time limits are "not in these passages", and Haiku gives the one-hour extension without the two-hour one), and three say them in other words: "four weekly rests" (both models, q091) and "six hours in a row" (Opus, q060). Those three count as not correct, so correctness errs low, never high.

**Alternatives:** a plain substring match, which missed 24 of those 51 answers; matching only a fact's number and unit and ignoring its other words, which would accept "4 weeks" for `4 consecutive weeks`; a model comparing the answer with the gold answer, which the owner ruled out for correctness (decision of the same day above).

## 2026-11-04: what checkpoint G measures and which versions it compares

**Decision** (owner, answering #75, #76, #77 and #78):

- **Correctness without key facts is graded by the owner, on the final test run only** (#75). On every run, an answer to a question with key facts is correct when it states each key fact, matched by code after numbers are normalised (#29). The 19 answerable dev questions and 10 answerable test questions without key facts get no correctness score on the dev runs; the table relies on citation recall and faithfulness for them. On the test split, the owner grades those 10 questions' answers from each finalist by hand (about 30 verdicts), against the frozen gold answers, and the grades are stored in the repository beside the run logs, so `make tables` reads them like any log. A model never grades correctness, as the design requires.
- **The faithfulness judge is chosen by measurement** (#76): Haiku 5.5 and Opus 5.5 both judge the owner's ~30 hand-labelled answers (#82), for under US$1 together. Haiku is used if it agrees with the owner on at least 27 of 30, otherwise Opus. Both agreement rates are reported. (Tightened on 2026-11-05, above: Haiku must also catch both unsupported answers.)
- **Reranking is done by Qwen3-Reranker 0.6B, run locally on the owner's Mac** (#77). The owner OK'd downloading it (about 1.2 GB) and running it. It is free and of the same family as the embedding model; Claude Haiku reranking can be added as an extra row for a few cents if it disappoints.
- **The dev comparison changes one thing at a time from the best setup** (#78): hybrid search over paragraph chunks with the top 5, already answered by Opus 5.5 in F. Five new Opus 5.5 runs on the 60 dev questions: top 3; top 10; article chunks; 200-word windows; and with reranking. The chunkings are compared at about the same amount of text (about 1,350 words: 10 paragraphs, 3 articles or 6 windows). F's Haiku run is an extra row. The best three then answer the 40 test questions once. Estimated at US$5–9 in total with judging, through the half-price batch service (#79).

**Alternatives:** a judge comparing every answer with its gold answer, which covers every question but lets a model grade correctness; the design's full grid of 18 versions, about US$15 even at half price, which would leave too little of the US$20 cap for the test run; a Bedrock rerank model, likely blocked by the project's Bedrock limits in Stockholm.

## 2026-11-03: asking one question with `make ask`

**Decision** (#65; running the local model for each question was OK'd by the owner in a working session): `make ask Q="…"` runs retrieval, the prompt, the model and the check for one typed question. It is a free dry run unless `RUN=1` is given, and paid questions follow the answer client's rules (decision of the same day below).

- **The best retrieval setup by default** (owner's choice): hybrid search over paragraph chunks, top 5, as in E's results. Vector search needs the question's vector. A question from the question set uses its cached one. Any other is embedded by Qwen3-Embedding 0.6B on the Mac's GPU, at the pinned revision. That adds about 5 seconds a question, while the model loads, and costs nothing. The command switches the Hugging Face hub offline before the model's libraries load, so it never downloads anything, however it is started; if the files are missing it fails and says so. `RETRIEVER=bm25` needs no model. The alternative, keyword search by default, is instant but put every needed passage in the top 5 for 30 of the 50 answerable dev questions, against 38 for hybrid.
- **A typed question's vector is not saved:** the committed embedding cache stays what E measured with, the corpus and the question set, rather than growing with every question tried. The vector is used once, in memory. A question for a paid embedding model (Titan, should it come back) is refused rather than paid for.
- **Each paid question has a log** in `runs/ask/ask-<time>-<model>.json`, in the same form as `make answers`' run logs, with the question's text added and the split given as `typed`. `rag.spend` counts its cost until a spend log row names the file, so a string of questions can't pass the cap unseen; the `ask-` prefix keeps a name from matching a dev run's log started in the same second, since the spend log names files without their folder. One row may name several files. A log file is created exclusively before anything is sent, so two questions in the same second can't overwrite each other's record (the same now holds for `make answers`). The logs are committed, like every run log, so every question paid for is in the repository's history, word for word.
- **What is printed:** the passages found (score, chunk and heading), the prompt's size and fingerprint, the estimate and what has been spent. After a paid call, it prints what a reader would be shown: the answer with each source's label, version and quote, the fixed refusal message, or "could not answer reliably" with three places to look. Below a failed answer it also prints the check's reasons, marked as not shown to a reader; this is a developer's tool, and the service in H shows only the reader's part.
- **Shared with `make answers`:** the cap and price checks, the estimate's wording and the run log's writer moved into `src/rag/generate.py`, and the choice of retriever into `rag.vector.make_retriever`, so both commands apply the same rules.

**Alternatives:** keyword search by default (above); saving typed questions' vectors to a second, git-ignored cache, which would save about 5 seconds on a repeated question but leave the typed questions on disk; one log per day for typed questions, which the spend log could name only once, so questions asked after the row was written would go uncounted.

## 2026-11-03: the answer client

**Decision** (#64): `src/rag/generate.py` sends each prompt to Claude through Anthropic's Messages API with the official Python SDK, and `make answers` runs it over the dev questions. Without `RUN=1` it sends nothing and needs no key.

- **The request:** the system prompt and one user message from `prompt.build_prompt`, the answer format as the reply's JSON schema (`output_config.format`), `effort: medium` and `max_tokens` from `config/generation.yaml`. Both models always think before answering, and the thinking is billed as output; effort is the only control, and medium is both models' default, written down so a change of default can't change a run. No prompt caching: the instructions are about 600 tokens, too short to be worth it now; G can weigh it with the Batches API.
- **The estimate:** input tokens from the prompt's text by the same rule as for embeddings (the higher of characters / 4 and words × 1.3), plus an allowance for the instructions the API adds for a JSON schema, then raised by 50%. Output can't be known before, so two figures are printed: the expected cost at 1,000 output tokens an answer plus the margin, and the most the run can cost, with every answer using all of `max_output_tokens`. A dev pass of 60 questions comes to US$0.064 expected and US$0.14 at most with Haiku 5.5, and US$2.57 and US$5.57 with Opus 5.5.
- **The cap check uses the most a run can cost**, on top of the spend log and every paid run (embedding or answer) whose log it doesn't name yet. So a run can't pass the cap even if every answer runs long. A run also refuses to start when the prices are more than a week old or a prompt may be over `max_prompt_tokens`.
- **`max_output_tokens: 4000`:** an answer is a few sentences and citations, and 4,000 leaves room for thinking. An answer cut off there fails the check and is logged as such. The limit keeps the most an Opus dev pass can cost near US$5.6; #66 shows how much the answers really use.
- **Retries:** the SDK's own retries are off, so every try is ours and in the log. Only refusals that are not billed are tried again: too many requests (429) and server errors (5xx, including 529, overloaded), at most 4 tries a call, after 5, 15 and 45 seconds. A timeout (300 s) or a dropped connection is not tried again, because the model may have answered and been paid for without the reply arriving, and a retry could pay twice beyond what the cap check allowed for. Nor is anything else (a bad key, an unknown model, a refused request). A call that still fails, an interruption or any other error stops the run: the answers so far are kept, the failed call is listed in the log's `failed`, the rest aren't sent, and nothing failed is counted as an answer. A timed-out call may still be billed without the API reporting its tokens, so each spend log row is checked against the Console's usage.
- **A refusal by the model** (`stop_reason: refusal`) fails like any other bad reply, with its category in the log. The API's fallback option, which would answer with another model, is not used: an answer from a model other than the one being measured would spoil the results.
- **The run log** (`runs/generation/<time>-<model>.json`, committed like the embedding run logs) is written before the first call, so a run that can't be logged never starts, and rewritten whole after every answer, so whatever stops a run, every reply paid for is on disk and the next run's cap check counts it. It holds, for each question, the retrieved chunk IDs, the raw reply, the check's verdict and reasons, the tokens the API counted, the cost at the configured price, the time and the tries; and for the run the model, effort, output limit, prices, prompt version (a fingerprint of the instructions and schema), retrieval setup and split. Its file name goes in the spend log row. Cache tokens, if any appear, are counted at no less than their price.
- **The key** is read from `ANTHROPIC_API_KEY`, in the environment or in `.env`, and only by a paid run. Only the dev split can be answered; the test split waits for G.

**Alternatives:** counting input tokens with the API's free token-counting endpoint, which is exact but needs the key and the network, so the dry run would no longer be offline; the SDK's built-in retries, which are fewer and invisible in the log; checking the cap against the expected cost, which a run of long answers could pass.

## 2026-11-03: the check of every answer

**Decision** (#63): `src/rag/validate.py` checks the model's reply against the chunks its prompt was built from, with code alone, before anything is shown. A reply fails when it doesn't fit the answer format, when a refusal cites anything or an answer cites nothing, when a cited ID isn't one of the `<node>` IDs the prompt marked, or when a quote isn't a continuous piece of that node's text as the retrieved chunks hold it. Every citation is checked and every failure listed, for the run log; a failed answer is shown as "could not answer reliably" (decision of the same day below).

- **What counts as the node's text:** the lines a chunk gives that node, joined by line breaks, exactly as its `<node>` element shows them. A window that holds only part of a node offers only that part; a paragraph with text before and after its own points offers two pieces, and a quote can't run from one into the other. A heading or lead-in line is not quotable, and neither is an article's ID when only its paragraphs were marked.
- **What is normalised before matching:** every run of whitespace (line breaks, tabs, no-break spaces) becomes one space, the quote's ends are trimmed, and curly or low quote marks and primes become straight ones, and hyphens, en and em dashes and the minus sign become "-". The acts write `‘carriage by road’` and list definitions after an em dash, and a model often writes them plainly, without changing a word. Characters a reader never sees (soft hyphens, zero-width spaces and joiners, direction marks, the byte order mark) are dropped from both sides, and the quote is put in Unicode's composed form, as the corpus is, so `é` written as `e` plus an accent still matches. Nothing else is forgiven: case, digits, a decimal point for the act's decimal comma (`3.5` for `3,5`), punctuation, and `...` or `…` standing for left-out words all fail.
- **A quote can't cut a word or number:** where a quote starts or ends with a letter or digit, the match must start or end on a word's edge, so `shall not exceed 5` is not a quote of `shall not exceed 56 hours`.
- **The reader sees the corpus's text:** a passing citation carries the quote as the corpus writes it, read back from the match, with its own quote marks and line breaks, never the model's copy. The same citation given twice is shown once.
- **What the model saw is what the check accepts:** the check reads the chunks, not the prompt. A test renders every chunk of all three chunkings and confirms that each `<node>` element's text is exactly a piece the check offers for that node, with nothing escaped, and that quoting it whole passes. If the corpus ever gains text the prompt must escape, that test fails rather than answers being refused quietly.

**Alternatives:** an exact match with no normalisation, which would refuse true quotes over a line break or a quote mark; also forgiving case or the ellipsis, which would let a model change or shorten the law's words and still pass; fuzzy matching with a similarity threshold, which would make "word for word" a matter of degree.

## 2026-11-03: the prompt and the answer format

**Decision** (#62; the three choices that change what a reader sees were answered by the owner in a working session): `src/rag/prompt.py` builds the prompt from any retriever's chunks, reads the model's reply, and holds the labels and fixed messages a reader is shown.

- **Citations name the most specific node** (owner's choice): "Article 6(3)", not "Article 6". The prompt splits each chunk's text into `<node id="561/2006:art6(3)">` elements, one per paragraph, point or guidance page, and a citation names one of those IDs and quotes that node's text. To make this possible each chunk now records which node owns each line of its body (`line_owners`); chunk IDs, text and node lists are unchanged. The rejected option, citing the retrieved chunk, would have left the reader to find the sentence in a whole article. The price is about 5–10% more prompt and one more way to fail: a right quote under the wrong paragraph is rejected.
- **The version comes from the code, never the model** (owner's choice). `citation_label` gives "Regulation (EC) No 561/2006, Article 6(3), consolidated version of 31.12.2024" from the manifest's pin; 2002/15, pinned to its original text, gives "original text of 11.3.2002, never amended", from a new manifest field `adopted`; a guidance page gives "Commission guidance note 6, page 2, published January 2017", and a note dated before the Mobility Package (Reg. 2020/1054, 31 July 2020) adds "before the 2020 Mobility Package changed the rules". A note's date is shown as month and year. For a note dated from its PDF that is when it existed by, so it may be older; a note known only by month and dated July 2020 gets the warning, as it may predate the package. Only an act that was never amended is pinned to its original text, which the manifest's comments now state. Every answer ends with the legal note "Consolidated texts are for information only; the legally binding text is the one published in the Official Journal." The prompt tells the model that the acts are consolidated versions and each note is as of its date, and that when an act and an older note disagree it follows the act and says the note is older.
- **Two fixed messages** (owner's choice). A question the passages don't answer gets `NOT_COVERED`, which names what the documents cover (a test fails when the manifest gains an act the message doesn't name), and never the model's own explanation, which nothing checks. An answer that fails a check gets `UNRELIABLE`, then "These passages may be relevant:" and the first three places the search found, named without version, so the reader has a next step. The draft is never shown. The wording is the agreed one even for the rarer failures, such as a reply that isn't in the answer format, where "didn't match the texts word for word" is loosely true.
- **Marking off chunks:** each chunk is a `<passage>` whose `source` names the document and version, with its heading path in `<heading>`, any lead-in in `<context>` (neither is quotable) and its text in `<node>` elements. Heading lines inside a window are `<heading>` lines between nodes. Text in a chunk or a question that would open or close any of the prompt's tags, in any case and with spaces or invisible characters (zero-width spaces, soft hyphens, direction marks) in between, has its `<` replaced by `&lt;`, so a passage can't end early or pose as another. Nothing else is changed, so every other quote still matches the corpus exactly; the corpus has no `<` at all today. The system prompt says that everything inside `<passages>` is material to quote and never an instruction.
- **Same input, same prompt:** the system prompt is a constant and the user message is the passages, in the order the retriever ranked them, then the question. Nothing varies between calls, which keeps runs repeatable and lets the API cache the system prompt.
- **The answer format:** the API's structured output is given `ANSWER_SCHEMA`, so the reply arrives as `{answerable, answer, citations[{node_id, quote}]}`. `parse_answer` still checks it, and rejects anything that isn't JSON, a missing, extra or repeated field, a field of the wrong type (`"true"` or `1` for `answerable`) and empty text in `answer`, `node_id` or `quote`. An empty quote would otherwise be found in any text. Whether the citations hold up is #63's.
- **What the checks can't do:** they prove each quote is real and comes from the cited node. They can't prove that the sentence around it draws the right conclusion; G measures that.

**Alternatives:** XML-escaping every `<`, `>` and `&` in the text, which would make the model quote `&lt;` and fail the exact match; a random delimiter per call, which breaks repeatable prompts; asking for JSON in the prompt alone, without the schema.

## 2026-11-03: answers come from Claude through Anthropic's API, not Bedrock

**Decision** (#61; answered by the owner in a working session): F and G send their prompts to Claude through Anthropic's own API, with an API key from the owner's Anthropic account, instead of through Bedrock's EU cross-region inference profile (decision of 2026-10-31 below). The model is chosen separately, in #61.

- **Why:** the owner found the project's Bedrock quota for Anthropic models at 0 on 2026-11-03, the same block E hit with Titan, which AWS would not raise without a usage history. The API works with the owner's key at once. Its standard limits for the lowest paid tier are 4,000 requests, 2,000,000 input tokens and 400,000 output tokens a minute for each of Opus 5.5, Sonnet 5.5 and Haiku 5.5 (Anthropic's rate-limit page, read 2026-11-03), far above what 60 questions need. A new account may start lower, in an evaluation tier; the account's own limits are on the Console's Limits page.
- **Spending:** the owner's Anthropic account holds about US$100 of promotional credits. As with AWS's credits, each cost is logged at list price and counts against the project cap of US$20, whichever pays. The AWS budget emails can't see this spend, so the owner sets a spend limit on the Console's Billing page before the first paid answer. A paid run is held to the cap by the same dry-run, cap and price-age checks as the embedding client.
- **Where questions are processed:** the API routes requests globally by default, and the only pin it offers is US-only (`inference_geo: "us"`, at 1.1 times the price); there is no EU-only option, which Bedrock's EU profile gave. The corpus and the evaluation questions are public, so nothing personal leaves the EU now. It matters for the deployed service (I), where people may type personal details into a question, and is weighed when I is planned.
- **Deployment (I):** the service holds the key as an AWS secret that only its own role can read, instead of an IAM permission on Bedrock models. GitHub still reaches AWS through OIDC; the Anthropic key is the project's one long-lived secret, kept in `.env` locally.
- **Bedrock stays possible:** the answer client gets a `provider` setting like the embedding client's, so Claude on Bedrock can come back if AWS raises the quota.

## 2026-11-03: Haiku 5.5 while building, Opus 5.5 for scored answers

**Decision** (#61; chosen by the owner in a working session): answers come from two Claude models, both set in `config/generation.yaml`. **Claude Haiku 5.5** (`claude-haiku-5-5`) answers while F is built and debugged. **Claude Opus 5.5** (`claude-opus-5-5`) writes every answer G scores, so it is the model the results table reports.

- **Prices** (US$ per million tokens, Anthropic's pricing page, read 2026-11-03): Haiku 5.5 0.10 input and 0.50 output, for prompts up to 100,000 tokens; Opus 5.5 4 and 20. The Message Batches API, which answers within hours, halves both. The model's thinking is billed as output.
- **Rough cost of one pass over the 60 dev questions:** about US$0.05 with Haiku and US$1.60–2.10 with Opus, or about half that as a batch. This assumes the top 5 chunks (842 words with paragraph chunks, 2,348 with article chunks), about 600 tokens of instructions and about 1,000 output tokens per answer, counting thinking. The output is the uncertain part, so #64's dry-run adds a margin and paid runs log the real counts. About 20 batch passes with Opus fit in the US$20 cap.
- **Why two models:** many debugging runs cost pennies with Haiku, while the scored answers come from the strongest of the three models compared. G can also run Haiku on the same questions and report whether the small model answers as well, as a measured row rather than a guess.
- **Limits:** the lowest paid tier allows each of the two models 4,000 requests, 2,000,000 input tokens and 400,000 output tokens a minute (Anthropic's rate-limit page, read 2026-11-03); the account's own tier is on the Console's Limits page.
- **Guard:** no prompt may be longer than 100,000 tokens (`max_prompt_tokens`), above which Haiku's price rises; F's prompts are about 2,000–4,000.

**Alternatives:** Sonnet 5.5 (US$2 and US$10), half Opus's price, which stays the fallback if Opus runs use more output than estimated; one model for both building and scoring.

## 2026-11-03: the secret scan allows SHA-256 cache keys in the embedding files, and nothing else

**Decision** (#50): `.gitleaks.toml` keeps every default gitleaks rule and adds one exception. In `embeddings/*.jsonl` and `runs/embeddings*/*.json`, a value of 16 to 64 lowercase hex digits and nothing else is not reported by the `generic-api-key` rule. A key is 64 digits; `gitleaks dir` reads a large file in pieces, and a key cut at a piece's edge reaches the rule shorter.

- **Why:** the embedding cache and its run logs name each text by its cache key, a SHA-256 of the text and the model setting, written as `"key": "…"`. The generic rule reads every one of them as an API key, so the first commit of real vectors was refused with 831 findings. The keys are hashes of public legislation and hide nothing.
- **How narrow:** one rule, one value shape, two kinds of file. A token made only of lowercase hex digits would also pass in those files, and nothing else writes to them. Checked with planted fakes: a token and an AWS access key in the same files are still caught, and the same hash in any other file is still reported. The hook stays on and `--no-verify` is still never used.
- **Alternative:** rename the field in the cache and the run logs to a word the rule ignores. That changes the file formats, including the paid Bedrock run log that is checked against the bill, to work around a scanner, and the next field with a hash would need the same trick.

## 2026-11-03: embeddings come from Qwen3-Embedding 0.6B, run locally

**Decision** (#57; the switch and the model were answered by the owner in a working session): checkpoint E turns chunks and questions into vectors with **Qwen3-Embedding 0.6B**, an open model (Apache 2.0) run on the owner's Mac with sentence-transformers. It replaces Titan Text Embeddings V2 (decision of 2026-10-31 below). Settings in `config/embedding.yaml`, code in `src/rag/embed.py`.

- **Why the switch:** the project's Bedrock limit for Titan V2 in Stockholm is 0, so no call can succeed. AWS support declined to raise it until the account has a usage history, and upgrading the project to the paid plan did not change the limit. The owner found no embedding model on Bedrock with a limit above 0. Running locally costs nothing and needs no AWS.
- **Why this model:** it reads 32,000 tokens at once, so the longest chunk (165/2014 Annex I, about 2,900 tokens) is embedded whole; many small models stop at 512. It is among the strongest small open models for search, it handles Swedish, which keeps checkpoint L open without a second model, and its vectors have 1,024 values like Titan's. The weights are 1.2 GB. Considered: BGE-M3 (also multilingual and well proven; 2.2 GB and slower) and Nomic Embed v1.5 (0.5 GB, the lightest to deploy; English only and weaker).
- **Pinned:** the model's files are pinned to Hugging Face commit `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, and it runs on the Mac's GPU (`device: mps`). The GPU and the CPU round differently in the last digits, so a machine without MPS refuses instead of falling back. The cache key and the cache file's name include the revision and the device, so a new upload or another device starts a new file and never mixes with or replaces a cached vector. Titan's keys are unchanged.
- **Questions get an instruction, chunks don't:** Qwen3-Embedding was trained with `Instruct: <task>\nQuery:` before each search query and nothing before the passages. The task is written for this corpus ("Given a question about EU road transport law, retrieve the passages of legislation and Commission guidance that answer it") rather than the model's generic web-search one. The prefix is part of the text the model is given, so it is part of the cache key, and both search backends add it before looking a question up.
- **Safety rails:** an input over 8,192 tokens is refused, because sentence-transformers would cut it short without saying so; every input must end with the end-of-text token, which the model reads its vector from (last-token pooling); and the tokens the library prepares for the model must be exactly the tokenized text, so a library update that wraps text in a template is caught. The model then runs on those checked tokens, not on a second tokenization. Texts go in one at a time, in float32, so a vector never depends on what it was batched with.
- **Running it:** `make embed` reports what is missing and loads nothing; `make embed RUN=1` runs the model (the flag was `SEND=1`). Its libraries (`torch`, `transformers`, `sentence-transformers`) are the optional dependency group `local-embed`, which only `make embed RUN=1` installs, so `make check` and CI stay light. The first run downloads the model into the Hugging Face cache. The run log, with the revision, device, library versions, tokens and time, goes to `runs/embeddings-local/`, outside the spend log.
- **What it means for later checkpoints:** F and G still generate answers with Claude on Bedrock, within the cap. In I, the deployed service has to embed each new question with the same model, so Qwen3-Embedding travels inside the service (about 1.2 GB of weights and 2–3 GB of memory on a CPU), or E's measurements would describe a different search. That cost is weighed when I is planned.
- **Titan stays possible:** the Bedrock client, the price check and the cap check are kept behind `provider: bedrock`, so Titan can come back as one more row in the table if AWS raises the limit.

## 2026-11-01: cross-reference expansion takes places inside the top k

**Decision** (#49; how added chunks count was answered by the owner in a working session): `Expansion` in `src/rag/retrieve.py` wraps any retriever (BM25, vector or hybrid). Going down its ranking, each hit is followed by the chunks its cross-references point to, until k chunks are in the list.

- **Inside k** (the owner's choice): an expanded top 5 is still 5 chunks, and a referenced chunk pushes the last one out. Every row of the results table then compares methods on the same number of chunks, and a gain means better chunks, not more of them. The alternatives were chunks added beyond k, where scores can only rise and part of the gain is just more text, and referenced chunks ranked by a score of their own, which adds a setting to tune.
- **Which references:** those of every node in a chunk's `node_ids`, and of the article text that introduces its first node, which is in its prefix, in the order the text cites them. References carry no position, so an article's references in its closing text count too wherever its lead-in does; three units in the corpus have references in text of their own above their children. They are followed one step (an added chunk's own references are not followed). Skipped: references to acts outside the corpus, unresolved ones, references to a chapter or to an annex made of parts (too broad to stand for one passage), targets in no chunk (amending articles), and targets the chunk already holds, even in part. A reference the parser resolved to the nearest ancestor of a node it lacks is followed to that ancestor. A guidance page's legal basis counts, so a note brings in the articles it interprets.
- **Which chunk stands for a target:** the one holding the most of it, counted in nodes of its subtree, the earliest on a tie. That is the article's chunk, the paragraph's chunk, or the window that covers most of it.
- **Hits with a score of 0** matched nothing, so their references are not followed, as in hybrid search.
- **Scores and marking:** an added chunk takes the score of the chunk that cites it, so scores never rise down the list, and `Hit.via` names that chunk. Run logs record it, and the table reports the share of top-5 places held by chunks a reference placed. That share includes chunks the search also found lower down, which a reference moved up, so it says how much of the top 5 expansion arranged, not how many chunks are new.
- **No limit per hit:** one hit citing "Articles 6 to 9" can fill the rest of a top 5. Capping it at one added chunk was tried with the variants below and made no difference beyond noise. The first k of a longer list are the list for k, so a log read 50 deep scores the same as a search for 5.
- **Context reported beside the scores:** every row of the results table now has "Words in top 5", the text the top 5 chunks hold together. Run logs carry each chunk's length for it.
- **Narrower variants were tried and not kept:** following only the first reference of each hit, or only references to definitions (as design problem 2 words it), changed recall@5 and recall@10 by one or two of the 50 dev questions either way against following them all. That is noise at this size, so the simplest rule stays, and E's vector and hybrid runs (#50) are measured with it. Tuning on the dev split to noise would only overfit it.

## 2026-11-01: vector search is exact, in one pgvector table, with an in-memory twin

**Decision** (#47; plan agreed with the owner in a working session): vector search has two backends behind `Retriever`, in `src/rag/vector.py`. `PgVectorSearch` searches one table, `chunk_vectors`, in the local Postgres + pgvector database. `VectorSearch` does the same computation in memory from the embedding cache.

- **One table** for every model setting and chunking strategy, one row per chunk: model setting (the cache file's name), strategy, position in document order, chunk ID, the embedded text and the vector. `make vector-index` replaces one strategy's rows in a single transaction, from cached vectors only, so running it again gives the same rows and a strategy with any vector missing is left as it was. A search refuses to start if the rows don't match the current chunks (ID, order, text and cached vector), so a stale index can't rank chunks or vectors that have changed, and each search checks the IDs it returns in case the table was reloaded since. The table is looked up in the first schema of the search path only, where it is created.
- **Exact search.** The database compares the question with every chunk and sorts; about 400 chunks take milliseconds. The vector column has no fixed size, so an approximate index (HNSW or IVFFlat) cannot even be built on it, and a test checks the query plan uses none. This is what the design asks for: local and deployed results are identical.
- **Same numbers in both backends.** Both use pgvector's cosine distance, 1 minus the similarity clamped to [−1, 1], and break ties in chunk order. The score is the similarity. pgvector adds up float32 products in an order its compiler picks, and the in-memory search adds in float64, so on real vectors the two agree to about 1e-6 and chunks closer than that may swap places. Vectors are stored as float32, so a gap that small carries no meaning. A test on 300 random 1,024-dimension vectors checks that only such swaps occur; a test on small whole-number vectors, where every sum is exact, checks that hits and scores are identical bit for bit, ties included. I's "both backends give identical rankings" check starts from these tests.
- **A similarity of exactly 0** reads as "matched nothing" in hybrid search, as BM25's 0 does, and gets no credit. A chunk at right angles to the question means as little, and RRF reads only the top 50 of about 400 chunks, where real embeddings of related legal text score well above 0.
- **Questions need a cached vector too**, like chunks; searching a question that has none raises an error naming `make embed`. Typed questions (`make search`) need a paid call per question, which is left for later.
- **Tests that need the database** are marked `db` and run with `make test-db`, each in a schema of its own that is dropped afterwards; `make test` and `make check` leave them out and stay offline.
- **Driver:** psycopg 3, sending vectors as pgvector text, which keeps float32 values exact; the `pgvector` Python package would add a dependency for no gain here.
- **Alternatives:** an HNSW index (approximate: results could differ from the in-memory backend, for no speed gain at this size); a table per strategy or model (more tables to create and drop, same queries); only the in-memory backend (the design asks for pgvector locally, and I compares the two).

## 2026-10-31: embedding vectors are cached in the repository

**Decision** (#46; where the cache lives was answered by the owner in a working session): every vector Bedrock returns is kept in `embeddings/` and committed. Code in `src/rag/embed.py`.

- **Why committed:** the results tables can then be regenerated from the repository alone, and J's quality gate measures search on every pull request without AWS access from GitHub and without paying. The cost is about 4.5 MB per set of vectors for the current 831 texts, and each new model or chunking adds a set that stays in the history. The vectors come from public EU legislation, so committing them reveals nothing if the repository is made public.
- **Alternative:** keep the cache on the developer's machine only. The repository stays small, but the tables could only be reproduced by paying for the vectors again, and the quality gate would have to call Bedrock from GitHub.
- **Key:** sha256 of the model ID, the vector size, the normalisation and the text. The design names model and text; the two settings are added because each changes the vector.
- **Format:** one JSON Lines file per model setting (`amazon.titan-embed-text-v2_0-1024-normalized.jsonl`), one line per text with its key, the token count Bedrock billed and the vector as base64 of little-endian float32. Float32 is the precision Titan computes in, and base64 is about a quarter of the size of JSON numbers. Lines are sorted by key and the file is rewritten whole through a temporary file, so the same vectors always give the same file and an interrupted write never leaves half a line. The text itself is not stored: it is rebuilt from the corpus, and the key ties each vector to it.
- **Paying:** nothing is sent without `SEND=1` (`RUN=1` since 2026-11-03). A paid run is held to the dry-run's estimate with its 50% margin, and refuses to start if that estimate on top of what has been spent would pass the approved cap, or if the price is more than 7 days old. What has been spent is the spend log plus every paid run whose call log in `runs/embeddings/` no spend-log row names yet, so two runs in a row can't both pass against a log that shows neither; the row is added by hand and names the call log's file. Throttling is retried by botocore, at most 6 tries a call, the first included. A call that still fails caches nothing and stops the run; what was already paid for is kept, also when the run is interrupted, and every call Bedrock answered is logged with its tokens, even if its vector is refused.

## 2026-10-31: hybrid search merges rankings by reciprocal rank fusion, k = 60, depth 50

**Decision** (#48): hybrid search (`RRF` in `src/rag/retrieve.py`) merges the rankings of BM25 and vector search by reciprocal rank fusion. Each retriever's top 50 chunks get 1 / (60 + rank) from it, and a chunk's score is the sum.

- **Why ranks, not scores:** BM25 scores (0 to about 30) and cosine similarities (−1 to 1) have no common scale, and weighting them would add a setting to tune on the dev split. RRF needs none, and is the method the design names.
- **The constant 60** comes from Cormack, Clarke and Büttcher (2009), who introduced RRF, and is the usual default. A smaller constant gives the top ranks more weight.
- **Depth 50**, or k if larger: deep enough that a chunk ranked 30th by one method and 1st by the other is still found, and deeper than any k the evaluation scores (10). Reading every chunk's rank would let a vector ranking, which ranks all chunks, give every chunk credit.
- **Only matches count.** BM25 fills its top k with chunks that hold no query word, score 0, in chunk order; those get no credit, or they would favour chunks early in the corpus. A chunk returned twice by one retriever counts once.
- **Ties** are broken by chunk order, as in BM25, so two runs give identical rankings. When fewer than k chunks get any credit, the rest of the top k follows in chunk order with score 0, as BM25 does.
- **Alternatives:** a weighted sum of normalised scores (one more setting to tune, and normalisation depends on the query); fusing more than two retrievers is supported but not planned.

## 2026-10-31: Bedrock runs in Stockholm (eu-north-1), and embedding cost is estimated before it is paid

**Decision** (#44): the project's Bedrock calls go to **eu-north-1 (Stockholm)**, and `make embed-dry-run` estimates the embedding cost before any paid run. Settings in `config/embedding.yaml`.

- **Availability** (AWS Bedrock documentation, read 2026-10-31): Titan Text Embeddings V2 runs in-region in every EU region, Stockholm and Frankfurt included. Claude's current models are not offered in-region in the EU, but through the EU cross-region inference profile, which can be called from Stockholm, Frankfurt, Ireland and Paris and routes requests only to EU regions. So embeddings stay in Stockholm and generation (F) stays in the EU.
- **Price** (AWS Price List API, offer `AmazonBedrock`, on-demand input tokens, read 2026-10-31): US$0.021 per million tokens in Stockholm, US$0.20 in Frankfurt. Stockholm is ten times cheaper for the same model, and it is the nearest region. Only these two were priced.
- **Tokens are estimated**, because Titan's tokenizer is not published: the higher of characters / 4 and words × 1.3 per input, plus a 50% margin on the total and on each input checked against the model's limit. A text that appears in more than one chunking is counted once, as the cache will pay for it once. Bedrock returns the real count with each call, and the paid run logs it.
- **The price is a dated setting**, not fetched on each run, so the dry-run works offline; it warns when the price is more than 7 days old, and the price is read again before a paid run.
- **First estimate:** 192,327 tokens for the 831 distinct texts of the three chunkings and all 100 questions, US$0.0040; US$0.0061 with the margin. The longest input, an article chunk of 165/2014 Annex I, is about 2,860 tokens, under the model's 8,192 even with the margin. Questions of both splits are counted: G embeds the test questions too, and embedding a question is not scoring it.
- **Vectors:** 1,024 dimensions, normalised, the model's defaults; cost does not depend on the size.

## 2026-10-31: embeddings come from Amazon Titan Text Embeddings V2 on Bedrock

*Replaced on 2026-11-03 by Qwen3-Embedding 0.6B, run locally (above): the project's Bedrock limit for Titan V2 is 0.*

**Decision** (answered by the owner in a working session): checkpoint E turns chunks and questions into vectors with Amazon Titan Text Embeddings V2 on AWS Bedrock.

- **Why:** answers are generated with Claude on Bedrock (F) and the service is deployed on AWS (I), so the embedding model measured in E is the one the deployed service uses, and the AWS account F needs anyway is set up now. The account setup therefore moves from I to E.
- **Cost:** the corpus in all three chunkings plus the 100 questions is about 0.2 million tokens. At the launch price of US$0.02 per million input tokens that is under one cent; the dry-run checks the price on the day before anything is spent.
- **Alternatives considered:** a small open model run locally (free, a 100–500 MB download; it would have to travel inside the deployed service, or be swapped for Bedrock there, which would leave E's measurements describing a different search), and both side by side (2–3 extra hours). Titan V2 is closed and runs only on Bedrock, so a local model would be a different model, not a cheaper copy. Cached vectors are keyed by model, so a local model can still be added later as one more row in the table.
- **Bilingual extension:** Cohere Embed Multilingual is reconsidered if checkpoint L is taken up.

## 2026-10-31: how retrieval evaluation scores a ranking

**Decision** (#32, answered by the owner): the rules `src/rag/retrieval_eval.py` follows (#35).

- **When a chunk finds a gold node:** when it holds any of that node's text. In terms of the chunk record (next entry), a chunk finds gold node G when its `node_ids` hold G or a descendant of G. An article chunk finds a gold paragraph, a paragraph chunk finds part of a gold article, and a window that touches a gold paragraph finds it, even if the key fact sits in the part it misses. Prefix text never counts.
- **Chunk length is reported beside the scores**, because this rule favours bigger chunks. Checkpoint G compares strategies at an equal context budget.
- **recall@k** counts a question as found only when every one of its gold nodes is found in the top k. Fractional recall, the share of gold nodes found, is reported beside it.
- **MRR@10** uses the rank of the first chunk that finds any gold node; a question with none in the top 10 scores 0.
- **nDCG@10** uses binary relevance and credits each gold node once, at the first chunk that finds it, so overlapping windows don't count a node twice. The ideal ranking holds one gold node at each of the first ranks. One chunk can find several gold nodes at once, such as an article chunk for two gold paragraphs, and beat that ideal, so nDCG is capped at 1.
- **Intervals:** 95% Wilson intervals for recall@k, and a seeded percentile bootstrap (10,000 resamples of questions) for MRR, nDCG and fractional recall.
- **Stats code:** written in this repo and tested against published values, instead of reused from `llm-inference-spike` as the design first said. The owner chose this to keep the project self-contained.
- **Questions counted:** unanswerable questions have no gold nodes, so they are left out of the retrieval metrics and the table gives their count; refusal is measured in G. Version traps are scored against their current gold nodes.
- **k values:** recall@1, @3, @5 and @10, MRR@10 and nDCG@10, matching the top-k values G tries.

**Why:** the lenient rule matches what generation needs: the passage with the answer is in front of the model. The strict rule, which needs the whole gold node inside the chunk, would mean a paragraph chunk could never find a gold article. Rescoring under another rule is free as long as run logs keep the full rankings, which the run log plan (#36) has to provide.

## 2026-10-31: run logs and results tables are committed

**Decision** (#36, plan approved by the owner on the issue): every retrieval run writes a log to `runs/retrieval/<name>.json`, and `make tables` builds `results/retrieval.md` from the logs alone. Both are committed. Format in `src/rag/runlog.py`.

- **A log holds** the configuration, the git commit, fingerprints of the question file and the corpus manifest, a fingerprint of every chunk searched, and for each dev question, unanswerable ones included, its top 50 chunks with scores rounded to 6 decimals. It also holds the node IDs of every chunk that appears, so a log can be scored again under another rule without chunking the corpus again.
- **Stale logs are refused:** `make tables` fails if the questions or the manifest changed since the run, or if the parser or a chunking strategy now gives other chunks. Retrieval code itself is not fingerprinted; the commit in each log names it.
- **Reruns are quiet:** `make retrieval-runs` leaves a log alone when only its time and commit would change.
- **`make check` fails** when `results/retrieval.md` doesn't match what `make tables` would write.
- **The recall@5 baseline** for J's quality gate is in `results/baseline.json`, one value per run.

**Why:** the logs are the evidence behind every number in the results, and from E on each run costs money, so it is never paid for twice. 50 chunks leave room for G's reranking of a longer list; three runs take about 450 KB.

**Alternatives:** logs kept out of git and rebuilt when needed; the full ranking (up to 412 chunks per question).

## 2026-10-31: BM25, our own code with a stemmer

**Decision** (#34): keyword retrieval is Okapi BM25 written in `src/rag/retrieve.py`, behind a `Retriever` interface (`search(query, k)`) that E's vector and hybrid search will implement too.

- **Parameters:** k1 = 1.5, b = 0.75, and the idf `log(1 + (N − df + 0.5) / (df + 0.5))`, which stays positive (Lucene's). Ties go to the earlier chunk, so two runs rank identically.
- **Tokens:** lowercased. References and numbers stay whole, because questions hinge on them:
  - An article reference (`6(3)`, `13(1)(r)`) is one token and adds its parts, so "Article 6(3)" still matches a heading "Article 6" and the label "(3)". A point's printed label, such as `(a)`, is a token too.
  - An act number (`561/2006`) is one token and adds its numbers. `8a` stays apart from `8`, and "Annex I" gives `annex-i`.
  - The act's decimal comma becomes a point, so `3,5 tonnes` matches `3.5 tonnes`. Dates and item numbers with three parts (`31.12.2024`, `1.5.2`) are one token.
- **Words:** English words are stemmed with the Snowball stemmer ("driving" and "drive" both become "drive"), and 45 function words such as "the" and "shall" are dropped. Numbers are never dropped.
- **What it indexes:** a chunk's prefix and body together, so heading words such as "Definitions" count.

**Why:** about 30 lines of our own keep ties and rounding under control, which a later test needs when it compares two backends' rankings. Stemming is the one piece not worth writing: `snowballstemmer` is small, pure Python and has no dependencies. Without it, paraphrased questions ("drive", "rested") would miss the act's wording ("driving", "rest") even more often, and the baseline would look weaker than keyword search really is.

**Alternatives:** `bm25s` or `rank-bm25`, which bring their own tokenisers and, for `bm25s`, numpy; no stemming.

## 2026-10-31: chunking strategies

**Decision** (#33, plan approved by the owner on the issue): three strategies in `src/rag/chunk.py`, each a function from the parsed corpus to a list of chunks.

- **The chunk record:** an ID, the strategy, the document, a prefix, a body, the IDs of every node whose own text the body holds at least in part (`node_ids`), and a guidance note's publication date. The indexed text is the prefix, a line break and the body.
- **Chunk IDs** are the strategy plus the node ID of the unit (`article:561/2006:art6`, `paragraph:561/2006:art4(e)`). Windows are numbered within each document (`window-200-50:561/2006:w12`); the size is part of the strategy name, so two window sizes never share an ID.
- **article:** one chunk per article. Annexes split into their top-level parts, so that 165/2014 Annex I (about 4,000 words) doesn't match every tachograph question. Guidance notes give one chunk per page.
- **paragraph:** an article splits into its children (paragraphs, or points where an article lists them directly); an article without children is one chunk, and text after the last child becomes a chunk with the article's ID. Each chunk's prefix carries its lead-in: the article's text before it, which stays out of every body, so a gold article is found through its children. Annex parts split the same way. Guidance notes give one chunk per page.
- **window-200-50:** windows of 200 words overlapping by 50, over each document's text in reading order. They cross article and page boundaries but never documents. A window carries the lead-in of the node it starts in.
- **Prefix:** the heading path of the chunk's unit, joined with ` › `, then any lead-in. A guidance note's date goes in its own field, not the prefix.
- **Left out:** amending articles (561/2006 Art. 26–27, 165/2014 Art. 45) and the act's title text. They stay in the parsed tree.
- **Words, not tokens,** until E chooses an embedding model; legal English runs at about 1.3 tokens per word.

**Why:** 200 words sits between a mean paragraph (69 words) and a mean article (275 words), so the three strategies differ; 25% overlap is the usual default and keeps most sentences whole in some window.

**Alternatives:** windows of 300 words with an overlap of 75; one chunk per annex.

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
