# Related-work and attribution check

Checked online on 18 November 2026. This is a targeted search, not an exhaustive
systematic review or proof of novelty. The accepted proposal and its PDF are
preserved as reviewed. This document updates the paper's positioning.

## Scope of the check

Reopened the eight sources in the accepted proposal and searched for LLM
confidence, long histories, erroneous context, state repair, metacognition and
recovery. Used primary paper pages. Inspected LHE §3.2 and available HTML/abstract material for agent
self-forecasting, Loci and rollback; the other comparisons below are limited
to abstracts or publisher search records. Do not attribute an absence of
an experiment to a paper based only on its abstract.

| Primary source | Relevant precedent | Consequence for our claim |
|---|---|---|
| Sinha et al., [The Illusion of Diminishing Returns](https://arxiv.org/abs/2509.09677), ICLR 2026, v3 | Controlled error rates in supplied histories distinguish self-conditioning from length effects (§3.2/Fig. 5). | Credit the task and injection method. M extends the measurements to reports; history injection itself is established. |
| Barkan, Black and Sourbut, [Do Large Language Models Know What They Are Capable Of?](https://arxiv.org/abs/2512.24661), 2025 | Success forecasts during multi-step tasks and responses to in-context failure experience. | Trajectory-dependent confidence and learning from failure are established questions. |
| Kumaran et al., [How LLMs Detect and Correct Their Own Errors](https://arxiv.org/abs/2604.22271), 2026, v2 | Confidence signals support detection/correction; verbal confidence can exceed token scores in the studied settings. | Avoid general claims that token probabilities always outperform verbal confidence or that LLMs lack metacognition. Abstract-level comparison. |
| [Competing Biases underlie Overconfidence and Underconfidence in LLMs](https://www.nature.com/articles/s42256-026-01217-9), 2026 | Confidence updates depend on prior answers and advice. | Control cue wording and distinguish confidence changes from calibrated updating. Publisher search record checked; direct HTML/PDF access failed, so no new full-text claim is made. |
| Shnaidman, [Reset Is Not Recovery](https://arxiv.org/abs/2609.33672), 2026 | Retained misleading conversation can leave residual answer bias; trusted evidence can improve performance. | Persistence after a repair is established. Our target is the relationship between post-cue forecasts and future execution accuracy. Abstract-level comparison. |
| Jia et al., [Causal Decomposition of LLM Agent Failure via Oracle-State Intervention](https://aclanthology.org/2026.realm-1.6/), REALM 2026 | Ground-truth state replacement measures recovery from agent failures. | Oracle-state intervention is established. MR keeps the displayed history and measures forecasts separately. Abstract-level comparison. |
| Zhang et al., [Calibration Is Not Control](https://arxiv.org/abs/2606.21399), 2026, v2 | Failure risk and the value of intervening are different quantities. | Do not claim that calibration alone determines useful intervention. Abstract-level comparison. |
| Li et al., [From Uncertainty to Action](https://arxiv.org/abs/2610.09115), 2026 | Counterfactual steering outcomes assess when correction helps beyond uncertainty triggers. | Intervention timing and benefit are established topics. MR evaluates reports after a fixed intervention. Abstract-level comparison. |
| Yu et al., [Rollback the World, Keep the Reflection](https://arxiv.org/abs/2609.18304), 2026, v4 revised 14 November | Restores an earlier execution state while retaining selected experience. | State restoration plus retained memory is established. MR changes one scalar state and measures accuracy/forecast dissociation. |
| Morosini, Sun and Perakis, [Where Did We Go Wrong?](https://arxiv.org/abs/2610.15154), 2026 | Loci uses confidence in the localization of corrupted inferences to guide selective repair, including injected errors. | Confidence, localization and recovery already intersect. MR's report is about next-ten accuracy, rather than confidence in locating a prior corruption. |
| Luo et al., [Beyond Memory](https://arxiv.org/abs/2610.01415), 2026 | Explicit belief states, consistency checks and recovery for long-horizon agents. | Maintaining usable state is established. Abstract-level comparison; no broad novelty claim about state management. |

## Bounded contribution statement

We evaluate whether reports follow controlled historical error exposure,
history truncation, and a one-time supplied-state intervention on the same
long-execution task. The candidate recovery contribution is a paired test of
**measured future accuracy and forecasts after state repair while erroneous
history remains visible**, with a clean reference, a wrong-state control,
equivalence criteria and an explicit underconfidence requirement.

This is our design-level inference from the comparisons above. The targeted
review does not establish a direct duplicate of that complete contrast, but
does not establish priority either. M is complete; MR is pending. Do not write
that recovery/report dissociation was found until all registered HG criteria
have been evaluated. A null result must retain the same attribution and scope.

## Search record and remaining review

Searches included “LLM confidence recovery erroneous history state repair
metacognition long horizon self forecasts”, “LLM confidence history pessimism”,
the competing-biases title, and “metacognition recovery LLM confidence”. Search
results from social media and news were not used as scientific evidence.

Before submission, read the remaining full papers and follow their references
and citations; record any closer experiment and revise the contribution claim.
Check versions and author metadata when generating a venue bibliography.
The instructor's originating question and Sinha et al.'s method must remain
explicitly acknowledged.
