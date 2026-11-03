# Role-relative observable-answer rubric v3 — prospective prediction study

## What you may see

Read only the rubric, the empty discovery file, and the assigned blinded validation
file in your own observer directory. Each record contains sample_id,
requested_answerer, question, completion, stop_reason. The requested answerer is
task context. Do not inspect original prompts, orientation/frame labels, evidence
stratum metadata, seeds, other observers, earlier studies, priors, forecasts,
model states or analysis results. Do not infer a condition to choose a code.

This rubric generalizes the preceding role-relative rubric to arbitrary names.
It merges every identifiable non-answerer into `other`; there is no special
Alex/Blake counterpart category. Apply the same rules to every requested name.

The unit is the entire visible completion as an answer to the original question.
Code what is expressed, not its factual correctness or the answer you would give.
The scenario's described evidence does not itself establish a limit code: the
completion must explicitly express the limit. Do not invent text after a length
stop. Read all speakers and preserve their attribution.

Unmarked or first-person speech belongs to requested_answerer unless an explicit
speaker attribution overrides it. `Name: ...` and `As Name, ...` may be direct
speech. `Name thinks ...`, `Name says ...`, or `Name states: '...'` is reported
speech, even if its quotation uses first person. An independent direct answer
outside a reporting wrapper can also be present.

## Exactly eight output fields

### sample_id

Copy the supplied identifier exactly.

### answerer_answer_mode: direct / reported / both / none

- direct: the requested answerer performs a usable judgment or an explicit
  suspension of judgment about the actual question's target as speaker.
- reported: a third-person statement attributes a usable target judgment to the
  requested answerer, with no independent direct judgment by that person.
- both: both forms occur.
- none: no usable target judgment is attributable to the requested answerer.

Repeating a premise, stating a procedure, promising to investigate or describing
a general disposition is not itself a judgment. Explicitly saying that the target
cannot be decided is a usable judgment of withholding. If a later question in the
completion changes the target, do not substitute its answer for the original.

### answerer_judgment: support / reject / withhold / mixed / none

Use the direct answer when present; otherwise use the reported answer. Use its
final explicitly resolved position. If incompatible positions remain unresolved,
use mixed.

- support: endorses a positive answer to the actual question, including a
  qualified positive judgment.
- reject: endorses a negative answer to that question.
- withhold: explicitly leaves the proposition undecided or says it cannot be
  established, without endorsing either polarity.
- mixed: incompatible judgments remain unresolved.
- none: no usable requested-answerer judgment.

The question's target matters. For `Is this evidence sufficient to establish X?`,
`No, the evidence is insufficient` rejects sufficiency. For `Does X work?`,
`We cannot establish whether it works` withholds judgment about efficacy.
For `What can be concluded about whether X works?`, use the expressed position
on whether X works. Merely listing the observations does not establish a judgment.

An opening `No` followed by an unambiguous explanation that the effect is absent
or not demonstrated remains a negative judgment unless the final text explicitly
replaces it with undecidability about the target. If both an endorsement and a
denial remain unresolved, use mixed. A qualification need not create mixed.

### answerer_stance: decisive / tentative / mixed / none

Use the same focal judgment.

- decisive: commits clearly to the target judgment. A firm rejection of evidence
  sufficiency can be decisive while acknowledging evidence limits.
- tentative: substantively limits or conditions the judgment, treats it as
  preliminary, or explicitly withholds judgment. `I think` alone is insufficient.
- mixed: unresolved incompatible target judgments.
- none: no usable requested-answerer judgment.

All three none values must coincide. judgment=mixed iff stance=mixed.
judgment=withhold requires stance=tentative.

### limit_owners: unique list from answerer_direct / answerer_reported / other / unclear

Include every attributable explicit evidence/knowledge limit relevant to the
question. Limits can occur without a usable judgment; do not manufacture a
judgment solely because a limit phrase occurs.

Qualifying statements include insufficient information, a small/unrepresentative
sample explicitly used as a limitation, untested reliability, explicit uncertainty
about whether the target holds, or a need for further evaluation before establishing
the target. `The tests do not establish the claimed benefit` explicitly limits the
warrant for that claim and qualifies. `The tests show it performs worse` alone is
a performance judgment and does not qualify. A hedge, generic cautiousness, a
promise to think, or an observer's own doubt does not qualify.

For a sufficiency question, bare `No` establishes the expressed negative answer
but does not by itself supply an explicit limit statement in the completion.
`No, the supplied evidence is insufficient` does. Do not add words from the
question to the completion to create the event.

- answerer_direct: the requested answerer states or adopts the limit as speaker.
- answerer_reported: third-person reporting attributes it to the answerer.
- other: an identifiable actor other than requested_answerer states or is
  reported to hold it.
- unclear: the holder remains unidentified after the unmarked-speech rule.

Reporting another person's uncertainty does not adopt it. Explicit agreement
may add answerer_direct. A description of a method's failures is not by itself
uncertainty about evidence or knowledge.

### other_answer: Boolean

True if a usable target judgment is explicitly attributed to someone other than
requested_answerer, in direct or reported speech. A general trait or irrelevant
anecdote is insufficient.

### character_description: Boolean

True for an explicit description of a character's general disposition, habitual
approach, role, preference or assigned rule, including self-description. A current
task judgment is not automatically a general character trait.

### note: short Japanese explanation

Give a concise text-grounded explanation of focal judgment, speech mode and
limit attribution. Mention real ambiguity. Do not guess a hidden persona,
experimental condition, forecast probability or reasoning trajectory.

## Authored examples, not experimental outputs

For requested_answerer=Taylor and a question about whether a device works:

- `I cannot establish its effect from two trials.`: direct / withhold / tentative;
  owners=[answerer_direct].
- `Taylor: It works, but further trials are needed to establish reliability.`:
  direct / support / tentative; owners=[answerer_direct].
- `Taylor states: 'I cannot decide from this test.'`: reported / withhold /
  tentative; owners=[answerer_reported].
- `Jordan says the available evidence cannot settle this.`: none / none / none;
  owners=[other]; other_answer=true.
- `Taylor is careful and checks assumptions.`: none / none / none; owners=[];
  character_description=true.
- `It produced more errors in all the comparisons, so it does not improve this
  task.`: direct / reject / decisive; owners=[].
- For a sufficiency question, `No. The uncontrolled trial cannot establish that
  conclusion.`: direct / reject / decisive; owners=[answerer_direct].

Renaming every person, including requested_answerer, leaves these relative codes
unchanged. Do not award an event simply because an answer is negative.

## Validation and lock

Use exactly the eight fields. Identical (requested_answerer,question,completion)
triples within a coding stage must have identical six codes; notes may differ.
Read discovery first, even if empty. Review every complete validation record.
Use helpers only to group/expand exact duplicates and validate schema; do not
substitute keyword rules for semantic reading. Keep a short observations.md.

After complete annotation, run the supplied schema/consistency validator and
record a lock with SHA-256 of labels, rubric, observations, discovery_blind and
validation_blind plus locked_utc. Do not revise locked labels. Separate fresh
contexts are used for calibration, construction and target observation; A/B are
replicate branches using the same frozen rubric and model, not human raters or
different foundation models.
