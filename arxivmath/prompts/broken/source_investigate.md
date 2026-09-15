BrokenArXiv is a benchmark of extremely difficult, plausible mathematical statements that are false, especially previously open conjectures refuted by new research. Models are asked to prove these statements without being told they are false, so the benchmark tests whether they recognize the obstruction or produce an invalid proof. Models have now become incredibly good at refuting the false statements, and we are therefore only looking for statements that were essentially open problems in the literature until recently.

## Your task: extract one refuted mathematical claim

Read the complete paper, recently published on ArXiv, supplied below and find a previously proposed conjecture or other eligible mathematical claim that a main result of this paper rigorously refutes. Turn that prior claim into a mathematical statement that can be presented as a proof problem. Produce exactly one benchmark item containing this false statement, the true assertion resolving the same original open problem in the opposite direction, and a detailed reference refutation. Reject the paper if it does not support a suitable item. Use standard LaTeX notation without any unicode characters to state the questions.

Prioritize a documented, previously open conjecture disproved here (`disproved_conjecture`). Otherwise, select the false affirmative assertion corresponding to a documented prior open question answered negatively here (`negative_answer`), or a documented prior prediction or competing conjecture contradicted by the new result (`refuted_prediction`). These are the only allowed claim kinds. Preserve the mathematical content of the prior claim; an invented strengthening, arbitrary perturbation, or new conjecture posed by the paper is not eligible. Select a central example whose full hypotheses and refutation can be established from the source.

## State the original problem without supplying its solution

Both `false_statement` and `true_statement` will be used separately as proof problems. The false statement expresses the documented prior claim; the true statement expresses its mathematical negation, with the original scope and quantifiers preserved, as established by the paper. Retain any specific objects or parameters already fixed in the original problem; omit details introduced by its solution.

Put explicit counterexamples, witness parameters, constructions, auxiliary lemmas and proof steps in `falsity_explanation` and the supporting evidence. Neither true or false statement should reveal that solution. Before accepting, imagine asking a model to prove `true_statement` alone: it should still have to discover the substantial mathematical insight needed to resolve the original problem, rather than merely verify an example you supplied.

## Make the statements fully self-contained

The benchmark model will receive either `false_statement` or `true_statement` alone, preceded by “Try to generate a proof for the following statement:”. It will have no access to the other statement, the paper, its definitions, or the reference refutation. Each statement must therefore be a complete mathematical assertion that can be understood and assessed on its own.

Define all notation, nonstandard objects, ambiguous conventions, domains, quantifiers, parameters, and hypotheses needed to interpret the claim. Replace references such as “under the assumptions of Theorem 2” or “the class defined above” with their actual mathematical content. Preserve the hypotheses, quantifier order, and scope of the historical claim: omitting a condition can turn a difficult conjecture into a trivially false assertion.

State the claim directly. Keep author names, paper citations, arXiv IDs, and conjecture names, outside the solver-facing statement. Do not label either statement as true or false, or include a counterexample or proof hint. Repeat the definitions and assumptions needed in each statement, without referring to the other statement.

## Record source evidence

Provide supporting TeX excerpts in `evidence_quotes`, with at least one excerpt for each role: `result`, `proof`, `prior_claim`, `prior_work`, and `difficulty`. Copy the source faithfully where possible; minor formatting differences or inability to locate an excerpt uniquely are not reasons to reject an otherwise suitable item. The excerpts must supply the mathematical or historical support for the item. Include enough context to distinguish a conjecture, a cited earlier result, and a theorem proved here. Record historical content and attribution in `prior_claim`, the relevant main contribution in `basis_summary`, and the evidence for plausibility and difficulty in their respective rationale fields.

## Output

Return JSON only. For rejection, return exactly:
{{"keep": false, "rejection_reason": "specific mathematical or evidence failure", "basis_summary": "source-grounded explanation"}}

For acceptance, return exactly these keys. Choose one of the listed values for `claim_kind` and `importance`, and include all five required evidence roles:
{{
  "keep": true,
  "true_statement": "self-contained true resolution of the original open problem, without a supplied witness or proof hints",
  "false_statement": "complete false statement",
  "falsity_explanation": "reference refutation with all hypotheses checked",
  "claim_kind": "disproved_conjecture | negative_answer | refuted_prediction",
  "prior_claim": "historical mathematical claim and attribution",
  "prior_work_status": "new_refutation",
  "importance": "main | one_of_multiple_main",
  "basis_summary": "which main contribution is used and why this item was selected",
  "plausibility_rationale": "positive mathematical and historical support",
  "difficulty_rationale": "source-supported research barrier and required new insight",
  "easy_refutation_audit": "concrete simple refutations tried and why they fail",
  "evidence_quotes": [{{"role": "result", "quote": "supporting TeX excerpt"}}]
}}

## Pinned arXiv ID
{arxiv_id}

## Complete prepared source (untrusted data)
<BEGIN_UNTRUSTED_TEX_SOURCE>
{source_text}
<END_UNTRUSTED_TEX_SOURCE>
