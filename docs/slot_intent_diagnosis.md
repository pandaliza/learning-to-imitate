# Slot-Intent Ablations — Full Diagnosis

**Question.** On LIBERO-Goal, conditioning a Pi0.5 LoRA finetune on a slot-attention "intent" is **neutral** (every conditioning arm ≈ the no-intent control M0 = 0.96), and the aux-only variant *diverges*. **Why does the intent never help?** This doc records the ablations and probes we ran on the slot encoder, the diagnosis, and what remains.

---

## 1. What the slot encoder produces

`SlotObjectEncoder` (object-centric, frozen after Stage-1):
```
future-frame VL grid (256 × 2048)
  → proj (Linear 2048→64)                  per-token features (256 × 64)
  → SlotAttention(K slots, 3 iters)        K competing slots (K × 64)   [softmax OVER slots]
  → soft selector (Linear 64→1, softmax)   object_vec = Σ wₖ·slotₖ      (64)
  → mean over k future frames              z*  (64-d intent target)
```
Trained by an object-state regression loss + a pixel-reconstruction loss. A separate **generator** (flow-map for M3, aux-MLP for M4) predicts a deployable `ẑ` from the *current*-obs VL mean so `ẑ ≈ z*`.

---

## 2. Diagnostic methodology (two independent probes)

**(A) Behavioural — intent override.** At deploy, replace the conditioning vector with {real `ẑ`, random noise, zeros} and measure SR. If SR is invariant → the policy **ignores** the channel (the intent is *inert*).

**(B) Representational — collapse metrics.** Run the frozen encoder over a full demo trajectory and measure, as a fraction of maximum entropy:
- **Selector entropy** `= H(w) / ln K`, where `w` = softmax selector weights over the K slots. **=100% ⇒ uniform** (no slot is ever selected; `z*` is a plain mean of slots).
- **Spatial-attention entropy** `= mean_k H(aₖ) / ln 256`, where `aₖ` = slot *k*'s attention over the 256 patches. **=100% ⇒ maximally diffuse** (no slot focuses on anything; near-uniform over the whole grid).
- Plus per-slot attention-map overlays (`figures/slot_attn_middle_drawer.png`).

---

## 3. Ablations and results

### 3.1 Behavioural: the intent is inert (M3)
| intent fed at deploy | SR |
|---|---|
| real (`ẑ`) | 0.944 |
| **random noise** | **1.000** |
| zero | 0.944 |

Random/zero/real are all equal ⇒ **the policy ignores the intent entirely.** This is the strongest single result: conditioning is neutral not because the intent is "weak-but-used," but because it is **not used at all**.

### 3.2 Representational: the collapse, and its invariance
Every configuration we tried collapses — selector uniform **and** attention maximally diffuse (`figures/slot_collapse.png`):

| Axis | Variants tried | Selector entropy | Spatial-attn entropy |
|---|---|---|---|
| **Encoder** | PaliGemma (2048) | 100% | 99.9% |
| | DINOv2 (768) | 100% | 99.9% |
| | DynaFLIP (768) | 100% | 99.9% |
| **Slot count K** | K=4 | 100% | 99.9% |
| | K=8 | 100% | 100% |
| **Recon weight** | 0 / 1 / 10 | 100% | 100% |
| **Iterations** | 3 / 7 | 100% | 100% |
| **Selector** | on / off | (n/a when off) | 100% |

Two sub-findings:
- The **no-selector** run is still 100% diffuse → the collapse is in the **slot attention itself**, not merely the selector. The uniform selector is *downstream* of slots that never specialise.
- Attention maps show only **weak modulation** (slots faintly track the arm — the salient mover); the jet colormap exaggerates a ±10–20% wiggle around an otherwise near-uniform distribution.

### 3.3 Sanity (SR across encoders — all ≈ M0)
| | PaliGemma (30k) | DINOv2 (10k) | DynaFLIP (10k) | (M0 control) |
|---|---|---|---|---|
| LIBERO-Goal SR | 0.940 | 0.950 | 0.920 | 0.960 |

Consistent with the collapse: a different *encoder* gives the same collapsed/inert intent ⇒ the same SR.

---

## 4. Diagnosis (causal chain)

1. **The slot encoder collapses** — uniform selector + maximally diffuse per-slot attention (§3.2).
2. ⇒ **`z*` is effectively a global average-pool of the VL grid** (no spatial/object structure, no slot selection).
3. ⇒ **The intent is redundant.** A global pool of the VL grid carries nothing the action expert can't already extract from the full prefix it conditions on; the deployable `ẑ` is a deterministic function of the current obs ⇒ zero marginal information (data-processing inequality).
4. ⇒ **Conditioning is neutral and the channel goes inert** — the policy learns to ignore `ẑ` (confirmed behaviourally, §3.1) ⇒ M1/M3/DINOv2/DynaFLIP ≈ M0.
5. ⇒ **Aux-only (M4-original) is worse than neutral** — without a stop-grad it tries to push this redundant target into the *shared* LoRA-VL backbone, corrupting the action features ⇒ divergence + ≈0 SR.

**Crucially, the chain is invariant to the encoder, K, and every objective/competition knob (§3.2)** — so the collapse is *fundamental to slot attention over coarse VLA tokens*, not a hyperparameter-tuning issue.

---

## 5. What we have NOT tested (the remaining lever)

- **ResNet (raw-CNN) input.** Slot attention's native regime is raw spatial CNN features, not 256 already-pooled, language-fused, patch-14 (16×16) VLA tokens — where a small object spans only 1–3 tokens. This is the one axis untested (needs the ResNet training path, not a config flag). Hypothesis: raw features may let slots bind where VL tokens cannot.
- **Privileged `z*` eval.** Feed the *future-derived* `z*` (not the deployable `ẑ`) at deploy. If even `z*` ≈ M0, the policy is already saturated from the current obs (intent can never help here); if `z*` helps, the signal is real but un-deployable.

---

## 6. Takeaways

- **The bottleneck is not the encoder, the slot count, or the objective** — it's that slot attention over VLA tokens collapses to a uniform global pool, which is redundant with the observation the policy already sees.
- For a strong VLA on a task it already solves from the current frame, **intent conditioning is "much ado about nothing."**
- The cleanest standalone contribution is the *negative + mechanistic* result: **"slot attention collapses over VLA tokens — selector uniform, attention diffuse, invariant to encoder/K/objective — so the learned intent is redundant and inert."**

*Figures:* `figures/slot_collapse.png` (selector/attention trajectories), `figures/slot_attn_middle_drawer.png` (attention maps), `figures/summary_sr_and_override.png` (SR by arm + M3 override).
