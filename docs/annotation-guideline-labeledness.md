# Labeledness annotation guideline

Version: v2
Status: approved
Applies to: `evals/golden/labeledness_v1.jsonl`

Amendment history:

* **v2** -- amended before any live verdict existed, at the point the first
  50-screen pass was discarded. Section 7.1 only: the artefact-term aid was
  generalised from 118 to 166 Preferred Terms and its published figure restated
  as a range. No verdict rule changed. See 7.2.
* **v1** -- approved before the draw. The sample manifest records v1 because
  that is the version the sample was *drawn* under; records carry the version
  they were *annotated* under, and after this amendment those differ.

This document defines the labels in the labeledness gold set. It is written and
approved before any annotation exists, and every annotation record carries the
guideline version it was made under. A guideline written after the fact
describes whatever the annotator happened to decide; this one has to be
falsifiable against the annotations rather than derived from them.

The gold set is curated by hand. No model participates in it: not in drawing the
sample, not in ordering it, not in pre-filling, suggesting, ranking or
highlighting. Every aid the harness offers is a published deterministic string
function over the label text.

---

## 1. The question

> For this drug, and this adverse event, does the label section shown already
> describe the event as something that happens to people who take this drug?

Three things are deliberately not being asked.

**Not whether the drug causes the event.** Disproportionality is
hypothesis-generating. The question is what the label says, not what is true.

**Not whether the event is plausible.** A pharmacologically obvious reaction
that the label does not mention is `not-labelled`. Judging plausibility instead
of text is the single most likely way for this gold set to go wrong, because it
is the judgement the annotator is best equipped to make and it is not the
judgement being asked for.

**Not whether some other version of this drug's label mentions it.** The verdict
is rendered against the one document shown, identified by its `set_id`. A
different manufacturer's label for the same molecule may say something else.
That is real variance and it is recorded, not averaged away.

---

## 2. The unit

One screen is one triple: a drug string, an adverse event Preferred Term, and
one label document.

The drug string is the FAERS string, verbatim, as the signal table keys it. It
is shown as-is, trailing period and all, because it is the join key back to the
signal run and normalizing it for display would hide which row a verdict
belongs to.

The event is a MedDRA Preferred Term as it arrives in the public FDA extract.
Labels are not written in MedDRA, so the PT will usually not appear verbatim in
the text. That is expected and is the reason sections 4 and 5 exist.

The document is one SPL, drawn uniformly at random, **independently for each
pair**, from the eligible documents of that drug string's query group. Eligible
means it carries adverse-reactions text and is reached from a drug string whose
label fetch was not truncated by the openFDA page cap.

Two screens on the same drug may therefore show different label versions, and
that is deliberate. Drawing once per drug would fix one manufacturer's label for
every question asked about that drug; drawing once per pair samples across the
versions instead, which is what makes the paragraph in section 1 about
cross-manufacturer variance true rather than aspirational. In the committed
sample, 300 pairs over 128 query groups drew 228 distinct documents.

The two presentations of a repeat are the exception, and they have to be: they
share a pair, so they share a document, byte for byte. A repeat that showed a
different label version would be a different question, and the agreement figure
would measure nothing.

---

## 3. The verdicts

Five, one keypress each. Three of them collapse to `labelled` when the metric is
computed; they are kept apart because they fail differently and because
collapsing later is free while splitting later is not.

| key | verdict | meaning |
|---|---|---|
| `l` | labelled, described | the label names this event, under any surface form |
| `b` | labelled, broader term | a genuinely broader term subsumes the event |
| `c` | labelled, class warning | named as a property of the drug class |
| `n` | not labelled | none of the above, after the protocol in section 6 |
| `u` | unclear | section 7 |

`l` covers synonymy. A term that denotes the same clinical entity under a
different surface form -- "hives" for `URTICARIA` -- is `l`, not `b`, because it
is not broader: it is the same finding in different words. There is deliberately
no sixth verdict separating verbatim matches from synonyms. Splitting them would
put a new judgement call ("is this a synonym or an inflection?") in front of the
annotator at the exact place consistency is most at risk, to recover a
distinction that costs nothing to recover later: the manifest holds the PT and
the exact text shown, so whether the PT appeared verbatim is derivable
mechanically after the fact, and the evaluation reports it as a split of `l`
without anyone having pressed a key for it.

No verdict is preselected. Pressing Enter alone does nothing and re-renders the
screen. There is no skip: skipping selects against hard cases, which are the
cases the adjudicator will fail on.

`z` retracts the verdict just given, for a miskey. It reaches back only within
the current sitting: a verdict from an earlier session cannot be retracted blind,
because it is not on screen to be reconsidered. To change one, retract nothing --
the way back to a screen is to re-annotate it, and the resume scan returns to any
screen with no live verdict.

### 3.1 `l` -- the label describes this event

The label describes this event, under any surface form, as something that
occurs.

`l` is not a claim about strings. It is a claim about clinical identity: the
label's term and the Preferred Term name **the same finding**. Whether the words
match is irrelevant, and mostly they do not -- fewer than 3 percent of pairs in
this frame carry the PT verbatim in the adverse-reactions text, so a rule that
required the string would put almost every real match into the wrong bucket.

Counts as `l`:

- The PT verbatim, in any case, in any list.
- A different inflection or word order of the same finding: "increased hepatic
  enzymes", "hepatic enzyme elevations" for `HEPATIC ENZYME INCREASED`.
- **A synonym or lay term for the same clinical entity**: "hives" for
  `URTICARIA`, "heart attack" for `MYOCARDIAL INFARCTION`, "low blood pressure"
  for `HYPOTENSION`, "fever" for `PYREXIA`. Hives is not a broader category that
  contains urticaria; it *is* urticaria, so this is `l` and never `b`.
- The event appearing in any of the four sections read by this project: adverse
  reactions, warnings, warnings and cautions, boxed warning. Which section it
  appeared in is recorded separately by the harness; it does not change the
  verdict.

Does not count as `l`:

- The event named only in a drug interaction, an overdose section, or a
  description of the underlying disease.
- The event named only as a reason to take the drug, only as something to
  monitor for, only as a contraindication, only as a dosing caution, or only in
  a description of the patient population. Every one of these is determinate and
  every one is `n`. Section 3.4 has the full table of roles; it is one rule, and
  these are the roles that are not "reported as a reaction".

Frequency does not matter, and neither does hedging. "Rare", "less than 1
percent", post-marketing spontaneous-report lists, and reports carrying an
explicit disclaimer that causation cannot be established are all `l`. A label
that reports the event at all reports it. Section 3.4 has the standard FDA
disclaimer quoted in full, and the measured share of this corpus that carries
it; the role the term plays decides the verdict, never how strongly the label
stands behind it.

Section 3.6 is the boundary between this verdict, `b`, and `n`. Read it before
annotating; it is where the consistency of this gold set is won or lost.

### 3.2 `b` -- labelled by a broader term

The label does not name this event, under any surface form, but names a term
that clinically **subsumes** it -- a genuinely broader category that contains it
-- such that a reader of the label would already expect this event to be
covered.

`b` is strictly for breadth. A term that denotes the same entity under a
different surface form is `l`, not `b`, however unlike the PT it looks. If the
label's term and the PT are the same finding, the fact that one is a lay word
and the other is MedDRA does not make one broader than the other.

Subsumption runs one way only: from the label's term down to the PT. The label
term must be the broader one.

- Label says "hepatic failure", PT is `HEPATIC NECROSIS`: `b`. Necrosis is
  within failure.
- Label says "hepatic necrosis", PT is `HEPATIC FAILURE`: **not** `b`. The label
  names something narrower. That is `n`.

Counts as `b`:

- A MedDRA-style higher-level relation stated in plain words: label says
  "serious skin reactions", PT is `STEVENS-JOHNSON SYNDROME`.
- A body-system statement that names the manifestation: "hepatotoxicity" for
  a specific hepatic injury PT.
- A named syndrome that includes the PT as a component: "anaphylaxis" for PT
  `BRONCHOSPASM` when the label describes bronchospasm as part of it.

Does not count as `b`:

- A body-system heading alone with no statement of harm. A section headed
  "Hepatic" that then lists only two specific enzymes does not cover every
  hepatic PT. The covering language has to be in the text, not in the heading.
- "Adverse reactions have been reported" and similar contentless breadth.
- Terms that overlap but do not subsume: "dizziness" does not cover `VERTIGO`,
  "rash" does not cover `URTICARIA`. Related is not broader. These are `n`.

When subsumption is genuinely arguable, it is `u`, not `b`. `b` means the
annotator would defend it.

### 3.3 `c` -- labelled by a class warning

The event is named, but attributed to the drug's pharmacological class rather
than to this drug specifically. The canonical case is the NSAID cardiovascular
thrombotic boxed warning, which names myocardial infarction and stroke as class
effects.

This is still `labelled`: the text is in this drug's label and a reader of the
label has been told. It is kept separate because it is weaker evidence that this
particular drug does it, and because a later analysis may reasonably want to
exclude it.

`c` requires that the text attributes the event to the class. If the label names
the event without class attribution, it is `l` or `b` even where the reaction is
known to be a class effect. The distinction is what the text says, not what is
pharmacologically true.

`c` is the only verdict that commonly requires leaving the adverse-reactions
section. The harness shows which other sections exist and reaching them is one
keypress.

### 3.4 `n` -- not labelled

The reading protocol in section 6 was completed and none of `l`, `b` or `c` was
found.

`n` is a positive finding, not a failure to find. It means the protocol ran to
completion. It is not "I could not find it in the time I had" -- there is no
time-limited `n`, because the protocol is bounded by construction and completing
it is always possible.

A label that mentions the term is not thereby a label that describes the event.
What decides it is **the role the term plays in the sentence**, and the three
cases below are named because they look indeterminate and are not. Each has a
determinate answer under the question in section 1, and each would otherwise sit
in `u` consuming annotation that cannot be scored.

**The test is the role, not whether causation is asserted.**

| the term appears as | verdict |
|---|---|
| something listed or reported as an adverse reaction, **however hedged** | `l` |
| a contraindication | `n` |
| an indication | `n` |
| a monitoring instruction | `n` |
| a dosing caution | `n` |
| a description of the patient population rather than of a reaction | `n` |

**Hedged reporting is `l`, and an explicit causality disclaimer changes
nothing.** This is the case most likely to be got wrong, because standard FDA
postmarketing language disclaims causation in so many words:

> Cases of X have been reported in patients receiving [drug]. Because these
> reactions are reported voluntarily from a population of uncertain size, it is
> not always possible to reliably estimate their frequency or establish a causal
> relationship to drug exposure.

Against PT `X`, that is **`l`**. The reaction is listed as an adverse reaction;
the second sentence is a regulatory statement about *evidential strength*, not a
statement that the event is absent from the label. Reading it as `n` would answer
a question nobody asked -- section 1 is explicit that the task is not to judge
whether the drug causes the event.

This is not an edge case. **6,263 of the 8,878 eligible documents, 70.5 percent,
carry that boilerplate or a fragment of it, and in every one of them it sits
inside the adverse-reactions section** -- which is to say, inside the section
being read, on roughly seven screens in ten. Postmarketing sections are also
where the harder non-lexical labelled cases live, so reading the disclaimer as
`n` would suppress the labelled rate hardest exactly where it is most
interesting.

**Indication-only is `n`.** The label names the event, but as what the drug is
for rather than as something it causes. An antiemetic label naming nausea under
indications, against PT `NAUSEA`, is `n`: the label does not describe nausea as
an adverse reaction, so the event is not labelled. This is not a close call, it
is the question being answered correctly.

**Monitoring-only is `n`.** "Monitor renal function periodically", with no
statement anywhere that renal impairment occurs, against PT `RENAL FAILURE`, is
`n`. An instruction to watch for something is not a report that it happens. If
the text also says the reaction is why the monitoring is advised, that is `l` --
but the saying has to be in the text.

**A term playing none of those roles is `n`.** Where the term appears but in
none of the roles in the table above -- neither reported as a reaction nor as
one of the named non-reaction roles -- the answer is `n` rather than `u`. That it
does not appear as a reaction *is* the finding, not an absence of one. `u` is for
text that cannot be read or questions that cannot be posed, not for answers that
come out negative.

`n` is expected to be the large majority of screens. The measured rate of
explicit lexical evidence over this frame is about 9 percent, so a run producing
very few `n` verdicts is a sign the guideline is being read too loosely, not a
sign the corpus is unusual.

### 3.5 `u` -- unclear

Section 7.

### 3.6 The boundary: same entity, broader, or merely related

This is the section that decides whether this gold set is consistent. `l`, `b`
and `n` are separated by one question asked of the label's term and the
Preferred Term:

> Do they name **the same finding** (`l`), a finding that **contains** it
> (`b`), or a **different finding** that merely overlaps or co-occurs with it
> (`n`)?

Every Preferred Term below is one that actually occurs in this sampling frame,
with its pair count in parentheses, so these are cases that can really come up
on a screen rather than constructed ones.

**Same entity -- `l`.** Different words, one finding.

| the label says | Preferred Term | why |
|---|---|---|
| "hives" | `URTICARIA` (156) | hives is urticaria; the words differ, the finding does not |
| "heart attack" | `MYOCARDIAL INFARCTION` (157) | lay term for the same event |
| "low blood pressure" | `HYPOTENSION` (157) | plain-language restatement |
| "fever" | `PYREXIA` (157) | lay term for the same sign |
| "nosebleed" | `EPISTAXIS` (157) | lay term for the same sign |
| "swelling of the hands and feet" | `OEDEMA PERIPHERAL` (157) | descriptive restatement of the same finding |

**Merely related, or narrower -- `n`.** These are the ones that look like the
row above and are not.

| the label says, and nothing more | Preferred Term | why |
|---|---|---|
| "rash" | `URTICARIA` (156) | "rash" is a nonspecific descriptor. A reader told only "rash" has not been told about hives |
| "dizziness" | `VERTIGO` (157) | overlapping, neither contains the other |
| "headache" | `MIGRAINE` (155) | headache does not entail migraine |
| "nausea" | `VOMITING` (157) | they co-occur; neither is the other |
| "aplastic anaemia" | `ANAEMIA` (157) | the label term is *narrower*; subsumption runs the wrong way |
| "elevated liver enzymes" | `HEPATIC FAILURE` (150) | narrower again; an enzyme rise is not liver failure |

The first row of each table is the same Preferred Term, `URTICARIA`, under two
different labels and with opposite verdicts. That pair is the boundary in its
sharpest form: "hives" is `l` because it *is* urticaria, "rash" is `n` because it
merely includes it among many things. If a screen feels like one of these, it is
worth deciding which of the two it is before pressing anything.

**Genuinely broader -- `b`.** The label's term is a category that contains the
PT.

| the label says | Preferred Term | why |
|---|---|---|
| "hepatic failure" | `HEPATIC NECROSIS` (94) | necrosis sits within failure |
| "serious skin reactions" | `STEVENS-JOHNSON SYNDROME` (144) | a named category containing the PT |
| "severe cutaneous adverse reactions" | `TOXIC EPIDERMAL NECROLYSIS` (118) | the same relation, one level up |
| "blood dyscrasias" | `AGRANULOCYTOSIS` (134) | a category term that covers the PT |

**The tiebreak.** If deciding between `l` and `b` requires an argument, it is
`b`; the label gave breadth rather than the thing. If deciding between `b` and
`n` requires an argument, it is `u`, because arguable subsumption is exactly what
`u` is for. Do not construct a chain of reasoning to reach `l`: `l` should be
recognition, not derivation.

---

## 4. Which sections are read

Four, in this order of precedence for the search:

1. `adverse_reactions` -- the primary section, always shown first.
2. `boxed_warning` -- short, and where class warnings usually live.
3. `warnings_and_cautions` -- the Physician Labeling Rule format.
4. `warnings` -- the older format.

A document in this sample always has adverse-reactions text; that is an
inclusion criterion. The other three may be absent, and the harness shows which
are present before any is opened.

Text found in any of the four supports `l`, `b` or `c`. The section it was found
in is recorded per annotation, so a later analysis can restrict to
adverse-reactions-only labelledness without re-annotating.

---

## 5. Search terms

The harness offers incremental search. It does not pre-highlight, because a
pre-computed highlight anchors the eye and turns the absence of a highlight into
evidence of absence, which it is not: fewer than 3 percent of pairs in this
frame have the PT verbatim in the adverse-reactions text.

The protocol below specifies what to search. Searching more is allowed;
searching less is not.

For a PT of one or more content words:

- The PT verbatim.
- Each content word of the PT, stemmed to its root, so that "elevations" is
  reached by searching "elevat". Words shorter than four characters, and the
  MedDRA filler words `NOS`, `not otherwise specified`, `disorder` and
  `disorders`, are not content words.
- Where the PT has an obvious lay synonym, that synonym. `PYREXIA` requires
  searching "fever".

---

## 6. The reading protocol

Two protocols. Which one applies is decided by the harness from the length of
the adverse-reactions text, before the screen is shown, and it is recorded on
the annotation.

The threshold is **12,000 characters** of adverse-reactions text. Measured over
the 8,878 eligible documents, the median is 6,450 characters and the 75th
percentile is 11,116, so the threshold sits just above the third quartile and
puts about 22 percent of eligible documents on the bounded protocol. It is set
where a full read stops fitting in the time a screen can have.

### 6.1 Full protocol -- adverse-reactions text at or under 12,000 characters

1. Read the adverse-reactions text.
2. Run the searches in section 5 across all present sections.
3. If a boxed warning is present, read it. It is short and it is where `c`
   lives.
4. Decide.

### 6.2 Bounded protocol -- adverse-reactions text over 12,000 characters

The section is not read end to end. Instead:

1. Run every search in section 5 across all present sections.
2. Read the subsection headings of the adverse-reactions text, and read any
   subsection whose heading is plausibly related to the event.
3. If a boxed warning is present, read it in full.
4. Decide.

**An `n` reached by completing the bounded protocol is a defensible `n` and is
recorded as `n`, not as `u`.** The bound is the point: an unbounded protocol on
a 188,000-character section does not produce a better verdict, it produces no
verdict. Which protocol applied is on every record, so the evaluation can report
whether verdicts on long sections behave differently from verdicts on short
ones. If they do, that is a finding about this gold set and it gets published
rather than hidden.

---

## 7. What goes in `u`

`u` is dropped from the metric, so it is not free: every `u` is a screen of
annotation effort that produces no measurement. It exists so that the four real
verdicts stay clean, not as a place to put hard screens.

`u` applies in exactly three cases:

1. **The text cannot be read.** Truncated mid-sentence, garbled, or not in
   English.
2. **The PT is a reporting artefact rather than a clinical event.** There is no
   bodily finding for a label to describe, so the question cannot be posed. See
   7.1.
3. **Subsumption is genuinely arguable** and the annotator would not defend `b`
   either way. This is the only clinical-judgement case left in `u`.

`u` does not apply when:

- The screen is long. That is section 6.2.
- The label names the event only as an indication or only as a monitoring
  instruction. Both are `n`; see section 3.4. These were in `u` in an earlier
  draft and were moved out because they have determinate answers, and a
  determinate case in `u` is annotation that was paid for and cannot be scored.
- The answer is uninteresting.
- The annotator is tired. Stop and resume instead; the harness is resumable and
  the elapsed time per screen is recorded, so fatigue is visible in the data.

### 7.1 Preferred Terms that are reporting artefacts

FAERS carries Preferred Terms describing a circumstance of use, a product-
quality complaint or a medication error rather than anything that happened in a
body. A label cannot describe them as adverse reactions because they are not
adverse reactions, so they are `u` under case 2 and the verdict should be
immediate rather than reasoned from scratch each time.

**The rule is denotation. The list is a recognition aid.** A term absent from
the list is still `u` if it denotes a circumstance of use; a term on the list
would still be `u` if the list were deleted. Nothing below is a boundary of the
population, and no pair is excluded from the frame on account of it -- a frame
filtered on a hand-built term list is a frame that has to be defended.

Measured over the full 374,846-pair frame, the prevalence is a **range, not a
point**:

| pattern | Preferred Terms | pairs | share | expected in 300 |
|---|---|---|---|---|
| v1, narrower -- a **floor** | 118 | 8,664 | 2.31% | 6.9 |
| **v2, current** | **166** | **9,466** | **2.53%** | **7.6** |

The v1 figure undercounted, and it is left visible rather than overwritten
because knowing the aid has been wrong once is worth more than a tidy number.
The cause was mechanical: the v1 pattern anchored the noun before `ERROR` and
`ISSUE` on `PRODUCT`, so the whole `DRUG ...ERROR` family fell through, and it
carried no handling or confusion vocabulary. **Three misses were found in a
single 50-screen sample** -- `DRUG DISPENSING ERROR`, `PRODUCT TEMPERATURE
EXCURSION ISSUE` and `PRODUCT AVAILABILITY ISSUE`, of which only the last was on
the v1 list. All three denote circumstances of use, so all three were `u` under
the rule the whole time; only the aid failed to name them.

A wider variant catching anything ending `ERROR` or `ISSUE` reaches 220 terms and
2.83 percent. It is **not** adopted: it sweeps in clinical events whose names
merely end that way, and an aid that has to be second-guessed is not an aid.

The most frequent, which between them are most of what will actually appear:

```
DRUG INEFFECTIVE                       OFF LABEL USE
DRUG INEFFECTIVE FOR UNAPPROVED        SUSPECTED OFF LABEL USE
  INDICATION                           PRODUCT USE IN UNAPPROVED INDICATION
TREATMENT FAILURE                      PRODUCT USE ISSUE
TREATMENT NONCOMPLIANCE                PRODUCT QUALITY ISSUE
THERAPEUTIC RESPONSE DECREASED         PRODUCT DOSE OMISSION ISSUE
THERAPEUTIC RESPONSE UNEXPECTED        PRODUCT PRESCRIBING ERROR
THERAPEUTIC PRODUCT EFFECT INCOMPLETE  PRODUCT ADMINISTRATION ERROR
THERAPEUTIC PRODUCT EFFECT DECREASED   PRODUCT DISPENSING ERROR
MEDICATION ERROR                       PRODUCT AVAILABILITY ISSUE
INCORRECT DOSE ADMINISTERED            PRODUCT SUBSTITUTION ISSUE
INCORRECT ROUTE OF PRODUCT             CONTRAINDICATED PRODUCT ADMINISTERED
  ADMINISTRATION                       EXPIRED PRODUCT ADMINISTERED
INAPPROPRIATE SCHEDULE OF PRODUCT      WRONG TECHNIQUE IN PRODUCT USAGE
  ADMINISTRATION                         PROCESS
INTENTIONAL PRODUCT MISUSE             ACCIDENTAL OVERDOSE
INTENTIONAL PRODUCT USE ISSUE          ACCIDENTAL EXPOSURE TO PRODUCT
INTENTIONAL DOSE OMISSION              PRESCRIBED OVERDOSE / UNDERDOSE
NO ADVERSE EVENT                       OVERDOSE / UNDERDOSE
UNEVALUABLE EVENT                      DEVICE ISSUE / DEVICE USE ISSUE

added in v2:
DRUG ADMINISTRATION ERROR              PRODUCT LABEL CONFUSION
DRUG PRESCRIBING ERROR                 PRODUCT PACKAGING CONFUSION
DRUG DISPENSING ERROR                  PRODUCT NAME CONFUSION
DRUG TITRATION ERROR                   PRODUCT APPEARANCE CONFUSION
MEDICATION MONITORING ERROR            PRODUCT TEMPERATURE EXCURSION ISSUE
DEVICE DELIVERY SYSTEM ISSUE           PRODUCT TAMPERING
DEVICE MECHANICAL ISSUE                PRODUCT COUNTERFEIT
  and the rest of the DEVICE
  subsystem family
```

The test is what the term denotes, not whether it appears above. A term naming a
circumstance of use is an artefact; a term naming something that happened in a
body is not, however administrative it sounds. `MEDICAL DEVICE SITE ERYTHEMA` is
a clinical event and is annotated normally; `DEVICE ISSUE` is not. Three near-
misses worth naming, because a looser reading of this list catches them and all
are real events to be annotated on their merits: `POOR QUALITY SLEEP`,
`INAPPROPRIATE ANTIDIURETIC HORMONE SECRETION`, and `PRE-EXISTING CONDITION
IMPROVED`.

`PRE-EXISTING CONDITION IMPROVED` is the instructive one and is deliberately
**not** on the list. It denotes a clinical outcome, not a circumstance of use, so
it fails the denotation test that governs this section. It is `n`, by section
3.4: no label describes it as an adverse reaction, and that is a determinate
answer rather than an indeterminate one. Keeping `u` narrow is the discipline --
every term admitted to `u` on a loose reading is annotation paid for and not
scored.

### 7.2 What the v2 amendment changed, and why it could be made mid-project

Recorded here because an amendment made quietly is indistinguishable from a
guideline written after the fact.

Made at the moment the first 50-screen pass was discarded, when the store held
zero live verdicts. That is the only point at which an amendment is not a mid-run
change against partially annotated data.

It touches section 7.1 and nothing else. No verdict rule moved: the predicate was
denotation before the amendment and is denotation after it, and only the
recognition aid grew. It can therefore reach `u` and nothing else -- and `u` is
dropped from the metric, so the amendment cannot move the labelled rate in either
direction.

The generalisation was measured over the whole 374,846-pair frame, not over the
drawn 330 screens, so it is not tuned to the sample being annotated.

A note is prompted on `u` and only on `u`. It is optional but it is the only
place the guideline learns what it failed to specify, so it is worth the
seconds.

**Target: `u` under 10 percent.** Above that, the guideline is underspecified
rather than the corpus being hard, and the checkpoint in section 8 is where that
gets caught.

---

## 8. The checkpoint at screen 50

Annotation stops at screen 50 for a review that is both a guideline gate and a
throughput gate. The harness reports, from the recorded per-screen elapsed
milliseconds rather than from an estimate:

- median and 90th-percentile seconds per screen;
- the `u` rate, against the 10 percent target;
- the distribution across `l`, `b` and `c`;
- how many of the 50 fell on the bounded protocol;
- within `l`, how many had the Preferred Term verbatim in the text and how many
  were reached by another name for the same finding. This is derived from the
  committed manifest afterwards, not asked for during annotation, and it is the
  reason section 3 has no sixth verdict for synonymy.

Two decisions come out of it. Whether the guideline needs a v2 amendment, in
which case the amendment is recorded here with its reason and those 50 screens
are re-annotated under v2. And what N is realistic for the day, which is a
decision made against measured pace rather than against a target.

Amendments are numbered, dated, and carry the reason. Records made under a
superseded version keep that version on them, and the evaluation either
re-annotates or reports them separately. It never silently pools them.

---

## 9. What the annotator is not shown

None of the following appears on any screen, because each of them correlates
with the answer:

- Any disproportionality statistic: `a`, `b`, `c`, `d`, ROR, PRR, IC025, EBGM05,
  or whether the pair was flagged.
- The lexical stratum, or whether the PT matches the text.
- Whether the screen is a repeat, or which screen it repeats.
- Any count of how many verdicts of each kind have been given.

The progress counter shows position within the schedule and nothing else.

---

## 10. Repeats and the consistency figure

About 10 percent of the schedule is a second presentation of a pair already
seen, placed at least 60 screens after the first and never marked. Where the
drug's query group carries a sibling string, the repeat is presented under the
sibling -- `PREDNISONE.` for `PREDNISONE` -- which is the same query, document
and event with a different surface.

Recognition cannot be eliminated: the text is identical, and changing it would
change the question. The resulting agreement figure is therefore an **upper
bound** on intra-annotator consistency and is published as one. With a single
annotator there is no inter-rater reliability to report, and an honestly bounded
consistency figure is what stands in its place.

If a repeat is recognised, it is answered as judged. There is no way to flag
one, because a flag would create the signal this design is suppressing.

---

## 11. Provenance

Every annotation records the document `set_id`, the section codes shown, and a
SHA-256 of each block of text as rendered. The evaluation loader verifies those
digests against the committed sample manifest and refuses to score a mismatch.
A verdict can therefore be re-checked years later against the exact text that
produced it, without re-deriving anything from a corpus that has since moved.

The sample manifest is committed under `evals/history/` before annotation
begins, so the frame cannot be adjusted after seeing what came out of the draw.
