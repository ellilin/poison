# 2. Theory

*What a perceptual hash can and cannot see of a backdoor trigger, how much
detection power the surviving information carries, and what an adaptive
adversary does about it.*

---

## 2.1 Threat and defender model

**Data.** A training set `D = {(x_i, y_i)}_{i=1..n}`, `x_i ∈ {0,…,255}^{H×W×3}`,
`y_i ∈ [C]`.

**Adversary.** Poison-only, all-to-one. It chooses a target label `t`, a
poisoning rate `ε`, a subset `S ⊂ [n]` with `|S| = εn`, and a trigger map
`T : X → X`. It delivers `x_i ← T(x_i)`, `y_i ← t` for `i ∈ S` (dirty-label) or
`x_i ← T(x_i)` for `i ∈ S ⊆ {i : y_i = t}` with labels untouched (clean-label).
The adversary knows the defense, including the hash functions and the detection
statistic (Kerckhoffs).

**Defender.** Sees `D` only, in secret-shared form: no clean holdout, no trained
model, no knowledge of `T`, `t` or `ε`. It outputs a keep-mask `K ⊆ [n]` and
trains on `D|_K`. Its two figures of merit are

* **elimination rate** `η = |S \ K| / |S|` (fraction of the poison removed), and
* **sacrifice rate** `σ = |K^c \ S| / |[n] \ S|` (fraction of clean data lost).

**Why this model.** It is the *only* setting in which a defense's cost is
independent of model training, and therefore the only one in which an MPC
deployment is plausible. It is also strictly harder than the setting assumed by
Spectral Signatures / Activation Clustering / SCAn / FLARE, all of which require
a model trained on the poisoned data.

---

## 2.2 Normal form: every perceptual hash is `binarise(L x)`

**Definition 1.** A *perceptual hash* in normal form is a pair `(L, β)` where
`L ∈ R^{B×d}` is a **public** linear map and `β` is one of

| `β` | bit `j` | comparisons needed |
|---|---|---|
| `zero` | `[ (Lx)_j > 0 ]` | `B` |
| `mean` | `[ (Lx)_j > (1/B) Σ_k (Lx)_k ]` | `B` |
| `pairwise` | `[ (Lx)_j > (Lx)_{π(j)} ]` for a fixed permutation `π` | `B` |
| `median` | `[ (Lx)_j > median_k (Lx)_k ]` | `B(B−1)/2 + B` (rank) |

Every hash in the zoo of `phash_defense/hashes.py` is of this form, with `L` the
composition of the (public) colour matrix, resampling matrix, DCT/DWT matrix,
box-blur band matrix and block-pooling matrix. The single exception is the
colour hash, which counts histogram bins and is therefore *not* a threshold of a
linear map — a fact that will make it the most expensive hash in MPC relative to
its length.

**Why this matters.** Under any linear secret-sharing scheme, `x ↦ Lx` for public
`L` is computed locally by each party: zero communication, zero rounds. So

> **the entire cost of a perceptual hash in MPC is the cost of `β`.**

This single observation drives the whole MPC analysis: the `mean`, `zero` and
`pairwise` thresholds cost `B` comparisons, whereas `median` costs `Θ(B²)`
comparisons or a `Θ(B log²B)`-deep sorting network — a factor of 32 in gates
(for `B = 64`) or a factor of 400 in rounds.

### Proposition 1 (the Haar wavelet hash is the median average hash)

*Let `x` be an `2^m × 2^m` image and let `LL_J` be the level-`J` Haar
approximation band. Then `LL_J = 2^J · P_J x`, where `P_J` is the block-mean
pooling over `2^J × 2^J` blocks. Consequently, for any `J`,*

```
wHash_haar(x)  =  mHash_{2^{m−J}}(x)      (bit for bit)
```

*Proof.* The Haar scaling filter is `(1/√2)(1,1)`, so one decomposition level
replaces each aligned pair by its sum divided by `√2`, i.e. `√2 ×` its mean.
After `J` levels in both directions each coefficient equals `2^J` times the mean
of its `2^J × 2^J` block. The median threshold is invariant under multiplication
by a positive constant, and `imagehash`'s optional "remove the maximum-level LL
band" step subtracts a constant from every coefficient, which shifts values and
median equally. ∎

Verified empirically at 100.0 % bit agreement over 1000 CIFAR-10 images
(`results/hash_validation.csv`, check `whash_haar_equals_mhash`).

*Consequence:* the "range of perceptual hash functions" is smaller than the list
of names suggests. aHash, mHash and Haar-wHash differ only in their threshold,
and dHash differs only in `π`. Genuine diversity comes from the *support* of the
rows of `L` — global (DCT), local (blockhash), or band-pass (residual).

---

## 2.3 What a trigger does to the bits

Write the trigger as an additive perturbation `e = T(x) − x` (for warping and
rotation triggers `e` depends on `x`, which is exactly the case the analysis
below identifies as hopeless).

### Proposition 2 (bit budget of a patch trigger)

*Let `L = P_p` be block-mean pooling with cell size `p × p` on an `N × N` image,
and let the trigger be an `s × s` patch at a fixed location, saturated to a
constant value `v`. Then:*

1. *at most `⌈(s+p−1)/p⌉²` bits can change;*
2. *exactly `⌊s/p⌋²` of them are **deterministic** — their value is the same for
   every poisoned image, independent of `x` — namely the cells the patch covers
   completely.*

*Proof.* A cell's mean depends on `x` only through pixels inside the cell. Cells
disjoint from the patch are untouched, which gives (1). If a cell lies entirely
inside the patch its mean is exactly `v` for every `x`, so its comparison against
the (nearly image-independent) threshold gives the same bit for all poisoned
samples, which gives (2). ∎

The numbers for CIFAR-10 (`N = 32`) and the canonical BadNets patch (`s = 3`):

| hash | `p` | affected cells | deterministic bits |
|---|---|---|---|
| aHash / mHash / wHash | 4 | 1 | 0 |
| aHash-16 / blockhash / dHash-16 | 2 | 4 | 1 |
| aHash-32 | 1 | 9 | 9 |
| rHash-32 (residual, blur `k=3`) | 1 | `(s+k−1)² = 25` | ≈ 24 |

The last row deserves its own statement, because it is the reason the residual
family was introduced.

### Proposition 3 (a saturated patch is *maximally* visible to a residual hash)

*Let `r = g − B_k g` be the residual with respect to a `k × k` box blur and let
the trigger set an `s × s` patch to the saturation value `255`. Then for every
pixel `u` within `⌊k/2⌋` of the patch:*

* *if `u` is inside the patch and its `k × k` window is not entirely saturated,
  `r(u) > 0` deterministically;*
* *if `u` is outside the patch but its window intersects it, `r(u) < 0`
  deterministically whenever `g(u) < 255`.*

*Hence `(s+k−1)² − (s−k+1)²` bits of `sign(r)` are fixed across all poisoned
images (24 bits for `s = k = 3`).*

*Proof.* `g ≤ 255` everywhere, so `B_k g(u) ≤ 255 = g(u)` for a saturated `u`,
with equality only if the whole window is saturated: `r(u) = g(u) − B_k g(u) > 0`.
For `u` outside the patch, the window contains at least one saturated pixel and
`g(u) < 255`, so `B_k g(u) > g(u)` iff the saturated contribution exceeds the
local mean — which holds because `255` is the maximum of the range. ∎

Combining Propositions 2–3 with the detection bound of §2.4 gives the central
*design law* of this work:

> **detection power grows with the spatial resolution of the hash and with the
> nonlinearity of its pooling — not with the hash's perceptual quality.**

### Proposition 4 (invariance ⇒ blindness)

*If a hash `H` is `δ`-robust, i.e. `H(x+e) = H(x)` for all `x` and all
`‖e‖_∞ ≤ δ`, then for any trigger with `‖T(x)−x‖_∞ ≤ δ` the hash distributions of
poisoned and clean samples are identical, and **no** statistic computed from `H`
can achieve AUC > 1/2.*

*Proof.* Immediate: `H ∘ T = H` pointwise, so the poisoned population is a
subsample of the clean hash distribution. ∎

This is not a weakness of a particular construction — `δ`-robustness is the
*design goal* of a perceptual hash. It predicts, before any experiment, that
low-`α` blended triggers, WaNet warping and steganographic triggers are invisible
to the classical hashes; §4 confirms it.

The quantitative version: for a threshold at margin `m_j = (Lx)_j − θ_j`,

```
P[ bit j flips ]  =  P[ |m_j| < |⟨L_j, e⟩| ]  ≈  2 f_{m_j}(0) · |⟨L_j, e⟩|,
```

so the flip probability is governed by `|⟨L_j, e⟩|` — the projection of the
trigger on the hash's own rows. A perceptual hash is built so that the rows `L_j`
are *low-pass*, and typical invisible triggers are *high-pass*; the inner product
is therefore small by construction.

### Proposition 5 (null-space triggers: the adaptive attack)

*Let the hash use block-mean pooling `P_p` with `p ≥ 2`. Let
`e(u,v) = a·(−1)^{u+v}` be the Nyquist checkerboard of amplitude `a`. Then
`P_p e = 0` exactly, hence `H(x+e) = H(x)` for **every** amplitude `a` for which
`x + e` stays in range.*

*Proof.* Each `p × p` cell with `p` even contains equally many `+a` and `−a`
entries, so its mean is 0; `p` a power of two is the case of every hash in the
zoo. ∎

Consequences:

* An adaptive adversary can use an *arbitrarily strong* trigger that is **exactly
  invisible** to aHash, mHash, wHash, blockhash and dHash. Verified: 0.00 % of
  their bits flip (`results/hash_validation.csv`, check
  `checkerboard_null_space`).
* DCT-based hashes (pHash, PDQ) are not exactly blind — the DCT-II of the
  checkerboard is not a single coefficient — but they are nearly so: ≤ 0.4 % of
  bits flip at amplitude 16/255.
* The residual hashes are *not* blind: `rhash32` flips ~11–32 % of its bits and
  `rhash_energy` ~10–18 %. Two mechanisms save them — full spatial resolution
  (no pooling to hide in) and the non-linear `|·|` before pooling, whose output
  is `a` everywhere rather than 0 on average.

> **Theorem (informal).** *No perceptual hash whose binarisation is a threshold of
> a linear map with a non-trivial kernel can be a sound poison filter against an
> adaptive adversary.* The kernel is a linear subspace the adversary can compute
> and hide in, at no cost in trigger strength.

The counter-measure is to break the linearity *before* pooling (residual energy)
or to remove the pooling (full-resolution residual sign) — both of which are
implemented and measured here. §2.5 shows that even this only forces the
adversary one step further, to a *sample-specific* trigger.

---

## 2.4 How much detection power do the surviving bits carry?

**Idealised model.** Inside the attacked class, clean hash bits are independent
`Bernoulli(1/2)`; the poisoned samples agree with the clean distribution except
on a set `J ⊆ [B]` of `|J| = J` *deterministic trigger bits*, where they all take
the same value `v_j`. The poisoning rate inside the class is `ε_c`.

### Proposition 6 (AUC of the bit log-likelihood ratio)

*The two-sample LLR statistic*

```
s_i = Σ_j  h_ij log(q_j/p_j) + (1−h_ij) log((1−q_j)/(1−p_j)),
q_j = class-c bit frequency,  p_j = background (other classes) bit frequency
```

*has, in the idealised model, `AUC = 1 − 2^{−(J+1)}`, independent of `ε_c`.*

*Proof.* To first order in `ε_c`, `q_j = 1/2 + ε_c(v_j − 1/2)` for `j ∈ J` and
`q_j = p_j = 1/2` otherwise, so the per-bit weight is
`log(q_j/p_j) − log((1−q_j)/(1−p_j)) = log((1+ε_c)/(1−ε_c)) ≈ 2ε_c` on `J` and 0
elsewhere. The score is therefore an increasing affine function of
`Σ_{j∈J} 1[h_ij = v_j]`, which equals `J` for every poisoned sample and follows
`Binomial(J, 1/2)` for a clean one. Hence
`AUC = P[Bin < J] + ½P[Bin = J] = 1 − 2^{−J} + ½·2^{−J}`. ∎

| `J` | 1 | 2 | 4 | 8 | 16 | 24 |
|---|---|---|---|---|---|---|
| AUC | 0.75 | 0.875 | 0.969 | 0.998 | 0.99998 | 1−3·10⁻⁸ |

Two consequences worth stating explicitly.

* **The poisoning rate does not appear.** `ε_c` cancels because it scales all
  weights equally. It re-enters only through the *estimation error* of `q_j`,
  which is `O(1/√n_c)`; detection therefore degrades at very small `ε_c` only
  because the trigger bits stop being visible above sampling noise. Concretely
  the LLR is reliable while `ε_c √n_c ≳ 1`, i.e. `ε_c ≳ 1/√n_c ≈ 1.2 %` for a
  CIFAR-10 class — which matches the measured rate sweep.
* **Detection power is exponential in the number of deterministic bits.**
  Combined with Proposition 2, this is why a 1024-bit full-resolution residual
  hash beats a 64-bit aHash by an enormous margin on patch triggers, even though
  aHash is the *better perceptual hash*.

For non-deterministic trigger bits (bit `j` flips with probability `π_j < 1`) the
Gaussian approximation gives

```
AUC ≈ Φ( Δ / √(σ²_clean + σ²_poison) ),
Δ  = Σ_j (π_j − p_j)·w_j,   w_j = log[q_j(1−p_j) / (p_j(1−q_j))].
```

### Proposition 6b (two mechanisms, opposite signs)

Any statistic computed per label class responds to **two** effects at once, and
they push in opposite directions:

* **Trigger consistency (+).** Poisoned samples share the trigger bits, so they
  look *more typical* of their class than a clean sample does: the LLR goes up,
  the neighbour count goes up.
* **Label atypicality (−).** In a *dirty-label* attack the poisoned samples are
  images of other classes carrying the target label. Their content is atypical
  for the class, so the LLR goes *down* and their neighbour count drops.

For a small trigger the second effect dominates, and the measured AUC lands
*below* 1/2 — a real signal, with the wrong sign. Both signs are known a priori
(they follow from the mechanism, not from the data), so the deployed defense must
be **two-sided**, and the achievable power is `max(AUC, 1−AUC)`.

More importantly, the second effect exists *even with no trigger at all*. Every
number reported for a dirty-label attack must therefore be read against a
**label-flip control**: the same relabelling with the trigger removed. In the
experiments that control reaches AUC 0.77 by itself (`label_flip_only` in
`docs/04_results.md`), so only the excess above it measures trigger detection.
Clean-label attacks have no such confounder and give the clean measurement.

### Proposition 7 (near-duplicate density)

*Two poisoned samples agree on all `J` trigger bits, so their expected Hamming
distance is `(B − J)/2` against `B/2` for two clean samples: the poison cluster
is displaced by `J/2` bits. A radius-`τ` ball around a poisoned sample contains
`ε_c n_c · P[Bin(B−J,½) ≤ τ] + n_c P[Bin(B,½) ≤ τ]` points.*

This is the statistic used by `hamming_knn` / `hamming_ball`. It uses the same
information as the LLR but pays `O(n²B)` instead of `O(nB)` — the reason it is
kept in the study is that it is the natural "k-NN defense" baseline the thesis
proposal refers to.

### Proposition 8 (block-collision separation)

*If the trigger fixes an entire spatial tile of `b` bits, all `ε_c n_c` poisoned
samples share one `b`-bit code, while the expected number of clean samples
sharing any given code is `n_c 2^{−b}`. The collision-count statistic therefore
separates the two populations with ratio*

```
ε_c n_c / (n_c 2^{−b}) = ε_c 2^{b}.
```

For `ε_c = 0.05` and `b = 16`: a factor of 3·10³. This is why the block-collision
sieve is the strongest detector against patch triggers, and why it fails
completely against global or sample-specific ones (no tile is fixed).

### Proposition 9 (why the cheap threshold rule fails)

*Let a fraction `ε` of the samples be flagged by a rule of the form
`z_i = (s_i − mean)/sd > k`. By Chebyshev's inequality applied to the empirical
distribution, at most `1/k²` of the points can have `z > k`, hence the rule can
fire only if*

```
k  ≤  √((1−ε)/ε).
```

*For `ε = 5 %`, `k ≤ 4.36`; for `ε = 10 %`, `k ≤ 3.00`; for `ε = 20 %`, `k ≤ 2.00`.*

The poison inflates the very standard deviation used to normalise it — the
classical *masking* effect. The median/MAD rule has a 50 % breakdown point and
does not suffer from it. This is an unwelcome result for MPC: the robust rule
needs an order statistic over `n = 50 000` values (an oblivious sort), while the
cheap rule needs only linear statistics. §3 quantifies the gap and §4 measures
how much detection quality is actually lost.

---

## 2.5 Adaptive adversaries

Order the adversary's options by how much they cost *him*:

1. **Ignore the defense** (BadNets, blended at large α). Detected: Propositions
   2, 3, 6 give AUC ≈ 1 for the residual hashes.
2. **Reduce the trigger amplitude** (blended at α = 0.02, 1-pixel patches). This
   is the `δ`-robustness regime of Proposition 4; the classical hashes go blind
   first, the residual hashes last, and the attack's own success rate drops as
   the trigger weakens. There is a genuine trade-off curve here, and it is the
   defense's real value.
3. **Hide in the null space** (Proposition 5). Free for the adversary against
   every pooling hash, so pooling hashes must be considered *broken* in the
   adaptive setting. Countered by non-linear pooling (`rhash_energy`) or full
   resolution (`rhash32`).
4. **Make the trigger sample-specific.** Now no *individual bit* is shared across
   poisoned samples, `J = 0`, and Proposition 6 gives AUC = 1/2 for every
   per-bit statistic.

   This is the regime where the defense actually fails, and the experiments
   confirm it: WaNet (image-dependent warping), Refool (a different reflection
   image per sample) and the multi-pattern low-opacity attack of Qi et al. all
   stay at the label-flip baseline.

   But sample-specificity has to be *complete*. The block-wise variant
   `phash_adaptive_ss` — the null-space checkerboard with an independent sign per
   4×4 block per sample — is detected at AUC 1.00, because a 4×4 tile of the
   residual sign map still has only **two** possible values (the checkerboard and
   its inverse) instead of `2^16`. Proposition 8's collision argument therefore
   applies with `b = 1` effective bit per tile and a collision ratio of
   `ε·2^1`... but summed over the 225 overlapping tiles, which is more than
   enough. Randomising the sign per *pixel* would defeat the sieve — and would
   also destroy the trigger, since the perturbation then becomes i.i.d. noise
   with no learnable structure.

   The adversary pays for sample-specificity in another way too: the published
   attacks in this class (ISSBA, LIRA, IAD) need either a trained steganographic
   encoder or control of the training loop, i.e. they leave the poison-only
   threat model this defense targets.

**What the defense therefore is.** Not a solution to backdoor poisoning: a
*cheap sieve* that removes the entire class of sample-agnostic, additively
localised triggers at a cost that is negligible next to secure training, and that
forces the adversary into the sample-specific regime where poison-only attacks
are hardest. Any claim beyond that is not supported by the analysis above.

---

## 2.6 The proposed defense: PHash-Sieve

```
Input : secret-shared training set (x_i, y_i)_{i=1..n}, hash list Hs,
        detector d, threshold rule R
Output: keep-mask K

1  for each hash H in Hs:                       # free linear part + B comparisons
2      b_i  <- H(x_i)                for all i
3      s^H  <- d({b_i}, {y_i})                  # per class, standardised (median/MAD)
4  s        <- rank-average of the s^H
5  K        <- { i : R(s_i, y_i) = keep }
6  return K
```

Instantiations studied here (`rhash32` throughout: Proposition 2 says spatial
resolution is what carries the trigger bits, and §3 says it costs 1024
comparisons per image — still less than a single median threshold):

| variant | hash | detector | rule | what it catches | MPC cost (50k CIFAR-10, 3PC/LAN) |
|---|---|---|---|---|---|
| **Sieve-B** (block) | `rhash32` | `block_collision` | `mad` | fixed patches, any trigger that fixes a spatial tile | 27.9 GB, 465 rounds, 75 s |
| **Sieve-L** (linear) | `rhash32` | `bit_llr` | `cluster` | global and steganographic triggers | 7.4 GB, 23 rounds, 20 s |
| **Sieve-S** (per-sample) | `rhash32` | `hash_stability` | `mad` | fragile triggers; no cross-sample access at all | 6.2 GB, 20 rounds, 17 s |
| **PHash-Sieve** | union of Sieve-B and Sieve-L | | | | 29.2 GB, 482 rounds, 78 s |

The threshold rule matters as much as the statistic. With `ε = 5 %` of the whole
dataset relabelled into one class, the *within-class* poison rate is 34 %, which
by Proposition 9 puts a hard ceiling of `k ≤ 1.4` on any mean/σ rule and makes
even the median/MAD rule fire only when the separation is large. The `cluster`
rule (1-D 2-means with a separation guard) makes no assumption about `ε` and is
the one that converts a high AUC into an actual removal. Its price is a higher
sacrifice rate, measured in §4.

The exact cost figures are produced by `scripts/run_mpc_bench.py`
(`results/mpc_pipeline_cost.csv`) and discussed in `docs/03_mpc_analysis.md`.
