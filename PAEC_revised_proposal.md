# Perspective-Anchored Evidential Coordination (PAEC)
## Training-Free Belief Attribution for Small LLMs via Perspective Prefix Fusion

---

## What changed from the original idea — and why

The original idea sampled *N* random reasoning paths and fused them as evidence. This is expensive because *N* is a free hyperparameter with no natural stopping point: to be reliable, you need N ≥ 20–40, meaning 20–40 forward passes per decision, per agent. For a 3B model that is feasible in isolation but prohibitive inside a coordination loop where many decisions must be made.

The key insight of the revision: **the number of evidence sources is not arbitrary — it equals the number of agents in the game (K).** In LLM-Coordination, K = 2. You run the model twice, once from each agent's perspective. The evidential fusion is now over *perspectives*, not over *random samples*. Each run produces semantically distinct evidence (what Agent 1 believes vs. what Agent 2 believes), so even K = 2 provides genuine evidential diversity. This collapses the sampling cost from O(N) = O(20+) to O(K) = O(2), a 10× reduction for typical coordination settings.

---

## The problem

LLMs used as coordination agents fail because they suffer from the **curse of knowledge**: they conflate their own (observer-level) knowledge with the knowledge of the specific agent they are playing. When Agent 1 has not seen that the door is locked, a standard LLM playing Agent 1 often reasons as if it has seen this, because the LLM's context window contains the full scenario description. This is the same failure SimpleToM identifies: LLMs infer mental states correctly ~90% of the time but fail to act from those states ~50% of the time.

For coordination specifically (LLM-Coordination Benchmark), this failure manifests as:
- **Over-confident joint plans** that assume shared knowledge the partner doesn't have.
- **Under-informative communication** — the agent doesn't ask questions it should, because it incorrectly models the partner as already knowing the relevant facts.
- **Failure to adapt** when the partner takes an unexpected action (which should update beliefs about what the partner knows).

The core missing piece is a mechanism that (a) forces explicit perspective separation between agents, and (b) quantifies when that separation is reliable enough to act on.

---

## The core idea

**Perspective-Anchored Evidential Coordination (PAEC)** has three components:

### Component 1 — Perspective prefix bank (minimal training)

Train K lightweight prefix vectors, one per agent role type. For a 2-agent coordination game, K = 2: a *self-perspective prefix* (reason from the viewpoint of the agent you are playing) and a *partner-perspective prefix* (reason from the viewpoint of the agent you are modeling). These prefixes are shared across all tasks of the same role structure — they are not task-specific. Training uses a small set of ToM perspective-taking examples (public datasets: ToMi, ExploreToM) and standard prefix tuning with cross-entropy loss on correct perspective-separated belief attributions.

**Why this is not heavy training:** Prefix tuning modifies ~0.01–0.05% of model parameters. For a 1.7B model, the prefix has ~200K parameters. Training takes minutes on a single GPU. The prefix is fixed after training and reused at zero marginal cost for all coordination games. This is qualitatively closer to a learned prompt than to fine-tuning.

### Component 2 — Perspective-anchored evidence generation

At inference, for each coordination decision point, the small LLM (1.7B–4B) is run exactly K times:

**Run 1 (self-perspective):** Prepend prefix₁ to the scenario context. The model generates a belief-attribution chain: what does Agent 1 know? What does Agent 1 believe about Agent 2? What actions are available to Agent 1 given this belief?

**Run 2 (partner-perspective):** Prepend prefix₂ to the same scenario context. The model generates: what does Agent 2 know? What does Agent 2 believe about Agent 1? What would Agent 2 expect Agent 1 to do?

Each run terminates with a mental state attribution (e.g., "Agent 2 believes the key is in Room A") and produces token-level logit sequences over the attribution tokens. These logits are converted to a **subjective logic opinion** ω = (b, d, u, a) via the LogTokU-style mapping: the Dirichlet parameters (α₁, α₂) are estimated from the log-probability mass over positive and negative attribution completions, and vacuity u = K_outcomes / (α₁ + α₂ + K_outcomes).

The two runs are independent forward passes — no beam search, no sampling beyond temperature = 0 (greedy) or temperature = 0.3 (low variance). Total compute: 2 forward passes on a 3B model ≈ equivalent to 1 forward pass on a 6B model.

### Component 3 — Dempster-Shafer fusion and vacuity-guided action routing

The two opinions (ω_self, ω_partner) are combined via Dempster's combination rule:

    ω_fused = ω_self ⊕ ω_partner

The fused opinion yields two diagnostically distinct uncertainty signals:

**Vacuity u_fused:** How much evidence is still missing? High vacuity means both perspectives are uncertain about the attributed belief — the scenario is genuinely ambiguous and more information is needed.

**Dissonance δ:** How much do the two perspectives conflict? High dissonance means self-perspective and partner-perspective give opposite attributions — a genuine perspective conflict that needs active disambiguation.

These two signals map to different actions:
- **u_fused > τ_u** (high vacuity): take an *information-seeking action* — observe, query, or wait.
- **δ > τ_δ** (high dissonance): take a *disambiguation action* — explicitly communicate or test a hypothesis about the partner's beliefs.
- **u_fused ≤ τ_u AND δ ≤ τ_δ** (low uncertainty): confidently act on the fused belief attribution.

The thresholds τ_u and τ_δ are hyperparameters calibrated on a small held-out validation set (no gradient-based training needed).

---

## Why this is novel

**Not a wrapper on self-consistency.** Self-consistency samples N random paths and majority-votes. PAEC runs exactly K semantically grounded perspective paths. The N is not a hyperparameter — it is determined by the problem structure. The fusion operator is not voting — it is Dempster combination with explicit conflict detection.

**Not a wrapper on AutoToM or ThoughtTracing.** AutoToM uses Bayesian inverse planning with standard posterior entropy — it cannot distinguish vacuity from dissonance. ThoughtTracing uses heuristic SMC weights without formal evidence theory. PAEC provides the vacuity-dissonance decomposition that Bayesian posteriors cannot.

**Not a wrapper on existing ToM prompting.** SimToM, TimeToM, and SymbolicToM improve ToM by better structuring the context. They produce point-estimate attributions without uncertainty quantification and do not connect uncertainty to action selection. PAEC's prefix mechanism is learned rather than purely prompt-engineered, and its uncertainty decomposition drives the action policy.

**The specific combination is unoccupied.** No existing work: (a) uses perspective-typed prefix tuning to generate K semantically distinct evidence sources, (b) fuses them via Dempster-Shafer/subjective logic, (c) uses the resulting vacuity-dissonance decomposition to govern coordination actions, (d) targets small LLMs (1.7B–4B) on multi-agent coordination benchmarks.

---

## Why it should work: the intuition

The curse-of-knowledge failure in small LLMs happens because the model has no structural forcing to reason from a limited information state. The perspective prefix is a learned "information blinder" — it steers the model's attention and activation patterns toward reasoning from one agent's access history rather than from the full context.

Once you have two perspective-separated belief chains, Dempster-Shafer fusion is the right aggregation tool because the two chains are *structurally distinct* evidence sources (two different perspectives on the same world state), not independent random samples from the same distribution. Dempster's rule is designed exactly for combining structurally distinct, partially overlapping evidence bodies. The conflict term in Dempster's rule (κ = 1 - Σ m₁(A)·m₂(B) for A∩B=∅) directly quantifies perspective disagreement, yielding dissonance as a first-class citizen. Vacuity falls out naturally from the Dirichlet parameterization without any additional computation.

---

## Experimental design

### Primary benchmark: LLM-Coordination

**Experiment 1 (main result):** Compare task success rate on LLM-Coordination across:
- Baseline 1: Standard prompting (no ToM, no uncertainty) with 1.7B, 3B, 4B models.
- Baseline 2: SimToM perspective-taking prompting (no prefix, no uncertainty).
- Baseline 3: Self-consistency voting (N=2 random samples, same compute as PAEC).
- Baseline 4: Self-consistency voting (N=8, 4× PAEC compute budget).
- PAEC: K=2 perspective-prefixed runs + Dempster fusion + vacuity-guided routing.

**Experiment 2 (communication quality):** Measure whether high-dissonance signals correctly predict cases where communication is necessary. Hypothesis: PAEC agents ask better questions (higher task-relevant information gain per question).

**Experiment 3 (scaling with model size):** Run PAEC on Qwen2.5-1.7B-Instruct, Llama-3.2-3B-Instruct, and Phi-3-mini-4B-Instruct. Show that the prefix-anchoring benefit holds even at 1.7B where raw ToM is near zero.

### Secondary benchmarks

**Experiment 4 (ToM accuracy):** Evaluate on ToMi, Hi-ToM, and ExploreToM to verify that perspective-anchored attributions are more accurate than SimToM and standard prompting.

**Experiment 5 (uncertainty calibration):** On ToMi (known ground truth), measure calibration of PAEC's projected probabilities (b + a·u) vs. self-consistency raw frequencies. Hypothesis: vacuity-corrected projected probabilities are better calibrated because they explicitly represent ignorance.

**Experiment 6 (prefix ablation):** Compare: (a) no prefix + Dempster fusion, (b) in-context perspective prompts + Dempster fusion, (c) full PAEC with perspective prefixes. This isolates the prefix contribution from the evidential fusion contribution.

---

## Theoretical contributions

**Theorem 1 (convergence of fused opinion):** As prefix-anchored generation quality improves, vacuity u_fused decreases monotonically and the fused belief distribution concentrates on the true attribution. Proof sketch: higher-quality evidence increases Dirichlet concentration parameters α₁ + α₂, driving u → 0. This connects prefix quality to uncertainty reduction in a formally tractable way.

**Theorem 2 (dissonance as coordination failure predictor):** High dissonance δ is a necessary condition for coordination failures caused by conflicting agent beliefs. If two agents' optimal actions given their respective belief states are incompatible, the Dempster combination of their perspective opinions must produce δ > 0. This provides a formal link between the uncertainty decomposition and coordination outcomes.

**Proposition 3 (compute-optimality of K=2):** For 2-agent coordination games, the information-theoretic value of a third sample path beyond the two perspective-typed runs is bounded by the mutual information between the third path and the scenario, which is dominated by the already-captured perspective variance. Empirically: show K=3 (adding a random path) provides negligible accuracy gain over K=2.

---

## Connection to your existing research

**Training-Free Policy Optimization:** The action routing policy (observe / disambiguate / act) is a deterministic function of (u_fused, δ) — no gradient, no learned policy. The prefix tuning is the only training step, and it is perspective-anchoring, not policy learning.

**Inference-Time Augmentation:** The entire framework operates at inference time with exactly 2 forward passes — minimal and fixed cost.

**Evidential Deep Learning:** The mathematical core is subjective logic / Dempster-Shafer theory, EDL's theoretical foundation. The logit-to-Dirichlet mapping follows LogTokU (Ma et al., 2025), which you can cite as prior work and extend to path-level fusion.

**Contrast with PDDL-Instruct:** PDDL-Instruct achieves high single-agent planning accuracy via heavy fine-tuning, with 0% cross-domain generalization. PAEC targets the strictly harder problem of multi-agent coordination with partial observability, achieves it via minimal prefix tuning plus inference-time evidential reasoning, and generalizes across coordination task types because the prefixes encode perspective roles, not task-specific knowledge.

---

## Suggested paper structure

1. **Introduction** — the curse of knowledge in LLM coordination; why uncertainty decomposition differs from accuracy improvement; the K=agents insight.
2. **Background** — subjective logic and Dempster-Shafer theory; prefix tuning; LLM-Coordination Benchmark; the SimpleToM explicit-to-applied gap.
3. **Method** — perspective prefix bank; perspective-anchored evidence generation; Dempster-Shafer fusion; vacuity-guided action routing; full algorithm pseudocode.
4. **Theory** — Theorems 1–2 and Proposition 3.
5. **Experiments** — Experiments 1–6.
6. **Analysis** — case studies of high-dissonance and high-vacuity scenarios; failure mode taxonomy.
7. **Discussion** — connection to POMDP belief updating and Friston's active inference; extension to K > 2; future directions.
