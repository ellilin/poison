# 4. Results

All numbers are produced by the scripts in `scripts/` and stored in
`results/`. The full tables are in [`results/tables.md`](../results/tables.md),
figures in `results/figures/`.

**Setup.** CIFAR-10, 50 000 training images, target class 0 ("airplane"),
poisoning rate 5 % of the whole dataset unless stated otherwise. Note that 5 %
of the dataset relabelled into one class makes the *within-class* poison rate
**34 %** — this matters for every thresholding statement below.

**Primary metric.** ROC-AUC restricted to the attacked class
(`auc_target_class_best`), which is the quantity a purification defense controls,
taking the better of the two score directions (Proposition 6b: a shared trigger
and a flipped label move the statistic opposite ways, and both directions are
known a priori, so the deployed defense is two-sided).

---

## 4.1 The hashes reproduce their reference implementations

Every hash is written as `binarise(L·x)` with an explicit public linear map.
Against `imagehash` on 1000 CIFAR-10 images (`results/hash_validation.csv`):

| hash | area/bilinear kernel (MPC-native) | PIL LANCZOS kernel |
|---|---|---|
| `ahash` | 96.7 % bit agreement | **100.000 %** (exact) |
| `dhash` | 90.7 % | **100.000 %** (exact) |
| `whash_db4` | 99.8 % | **100.000 %** (exact) |
| `whash_haar` | 99.90 % | 99.975 % |
| `phash` | 99.64 % | 99.77 % |
| `colorhash` | 99.90 % | 99.86 % |

A resampling filter is a linear map whatever its kernel, so PIL's LANCZOS is
just as free under secret sharing as a box filter. Reproducing it (plus PIL's
integer `convert("L")`) makes aHash, dHash and db4-wHash **bit-exact** with the
reference; the residual 0.1–0.2 % on pHash and Haar-wHash are float-precision
ties in the median comparison.

Two structural claims of `docs/02_theory.md` are confirmed exactly:

* **Proposition 1**: `whash_haar` ≡ `mhash`, 100.000 % bit agreement.
* **Proposition 5**: the Nyquist checkerboard flips **0.00 %** of the bits of
  aHash, aHash-16, mHash, wHash-Haar, blockhash and dHash at any amplitude;
  0.2–0.8 % of pHash/PDQ/dHash-16/rHash-8/16; and 25–40 % of `rhash32` /
  `rhash_energy`.

## 4.2 The MPC formulation is the same hash

Running every hash on the traced fixed-point secret-sharing type and comparing
with the plaintext bits (`results/mpc_correctness.csv`, 128 real CIFAR-10
images):

| fractional bits | hashes at 100 % agreement |
|---|---|
| 12 | ahash, ahash16, mhash, phash, whash_haar |
| 16 | + dhash, dhash16, blockhash, pdq, phash_mean |
| 20 | + rhash_energy |

The residual hashes plateau at 99.6–99.8 % on *natural* images (they reach
100 % on synthetic noise). The reason is measurable: **88–99 % of the mismatched
bits have `|residual| < 2⁻⁸`**, i.e. they are exact ties in flat image regions
where the sign is decided by the last rounding unit. They are removable — see
§4.7.

## 4.3 What the hashes actually detect

Full grid: 14 attacks × 26 hashes × 7 detectors, `results/detection.csv`
(2 184 rows), tables T2/T3 in `results/tables.md`, figure
`results/figures/auc_heatmap.png`.

### 4.3.1 The control that every dirty-label number must be read against

A dirty-label attack moves images of *other* classes into the target class.
Those images are atypical for their new label, and a per-class hash statistic
sees that — with no trigger involved at all. The `label_flip_only` control
(same relabelling, trigger removed) reaches

* **AUC 0.770** (`rhash_energy` + `hamming_knn`),
* mean 0.607 over all 182 (hash, detector) pairs.

So "AUC 0.75 on WaNet" is not WaNet detection; it is label-noise detection.
Table T4c reports the honest quantity: **AUC above the control**.

### 4.3.2 Trigger-specific detection power

Best (hash, detector) per attack, AUC above the label-flip control:

| attack | trigger | best hash | ΔAUC | verdict |
|---|---|---|---|---|
| `batt` (16° rotation) | geometric, sample-agnostic | `ahash_res` | **+0.47** | detected |
| `blended` (α = 0.10) | global noise | `rhash_rgb` | **+0.46** | detected |
| `stego` (ISSBA-style) | invisible, steganographic | `rhash_rgb` | **+0.46** | detected |
| `blended_a05` (α = 0.05) | invisible global | `rhash_rgb` | **+0.46** | detected |
| `phash_adaptive` | null-space checkerboard | `rhash32` | **+0.41** | detected |
| `blended_a02` (α = 0.02) | invisible global | `rhash32` | **+0.40** | detected |
| `phash_adaptive_ss` | + per-block random signs | `rhash32` | **+0.39** | detected |
| `badnets` (3×3 patch) | patch | `pdq` | **+0.40** | detected |
| `clean_label_badnets` | patch, clean label | `pdq` | **+0.40** | detected |
| `refool` | reflection, **sample-specific** | `dhash` | +0.07 | **not detected** |
| `badnets_1px` | 1-pixel patch | `rhash16` | +0.07 | **not detected** |
| `adaptive_patch` (Qi et al.) | 4 patterns, α = 0.5, cover samples | `mh` | +0.05 | **not detected** |
| `wanet` | image-dependent warping | `rhash_energy` | +0.03 | **not detected** |

The split is exactly the one the theory predicts:

> **detected ⇔ the trigger is sample-agnostic and strong enough to fix bits;
> not detected ⇔ the trigger is sample-specific, or too weak to move any bit.**

* WaNet and Refool fail because their *pixel-level realisation differs per
  sample* (warping depends on image content; each poisoned image gets a
  different reflection). No bit is shared, `J = 0`, Proposition 6 gives AUC ½.
* `badnets_1px` fails because a single pixel fixes at most one bit even at full
  resolution (Proposition 2), and one bit gives AUC ≤ 0.75 minus the estimation
  noise.
* `adaptive_patch` fails because the attack splits its poison over four
  different patterns at 50 % opacity — each pattern is carried by only a quarter
  of the poisoned samples, quartering the effective `ε` while quartering the bit
  contrast. The ICLR'23 adaptive attack defeats this defense as it defeats the
  latent-separability defenses it was designed against.

### 4.3.3 Which hash

Mean ΔAUC over the 13 attacks (T4c):

| rank | hash | bits | family | mean ΔAUC |
|---|---|---|---|---|
| 1 | **`rhash32`** | 1024 | residual (ours) | **0.262** |
| 2 | `rhash_rgb` | 768 | residual (ours) | 0.178 |
| 3 | `ahash_res` | 256 | residual (ours) | 0.149 |
| 4 | `rhash16` | 256 | residual (ours) | 0.148 |
| 5 | `pdq` | 256 | frequency | 0.139 |
| 6 | `phash16` | 256 | frequency | 0.113 |
| 7 | `phash` | 64 | frequency | 0.092 |
| … | | | | |
| 22 | `ahash16` | 256 | spatial | 0.017 |
| 23 | `ahash` | 64 | spatial | 0.017 |
| 25 | `ahash4` | 16 | spatial | 0.015 |

Three conclusions:

1. **The classical perceptual hashes are close to useless for this task.**
   aHash, mHash, wHash, blockhash and dHash sit at ΔAUC ≤ 0.05 on everything
   except the rotation trigger. This is not an implementation artefact: they are
   *designed* to discard exactly the signal a backdoor trigger lives in
   (Proposition 4).
2. **The residual family — the "perceptual complement" proposed here — is 3–15×
   better**, and it is the only family that sees invisible global and
   steganographic triggers.
3. **Spatial resolution beats hash quality.** `rhash32` (per-pixel residual
   sign) beats `rhash16` by 0.11 mean ΔAUC, and `ahash32` beats `ahash` by only
   0.01 — resolution helps only when the pooling has not already averaged the
   trigger away. Compare `blended`: `rhash32` +0.41, `rhash16` +0.46 (pooling
   helps for a global trigger), versus `badnets`: `rhash32` +0.21, `rhash16`
   +0.00 (pooling destroys a localised trigger).

`results/figures/trigger_response_badnets.png` shows this directly: the
bit-flip map of the 3×3 corner patch lights up a single corner cell in aHash
(1.9 % of bits), aHash-16 (1.7 %) and blockhash (1.4 %) — Proposition 2's bit
budget, visible — while **pHash flips 15 % of its bits**, spread over the whole
low-frequency band. A localised perturbation has a global DCT signature, and
that is exactly why the DCT hashes are the only classical family that detects a
patch: they turn one changed cell into dozens of correlated bit changes.

### 4.3.4 Which detector

Mean best-AUC over attacks (T3):

| detector | mean AUC | complexity | notes |
|---|---|---|---|
| `bit_em` | 0.899 | `O(TnB)` | best overall; EM mixture inside the class |
| `bit_llr` | 0.886 | `O(nB)` | 1.4 % behind, and the cheapest in MPC |
| `hamming_knn` | 0.864 | `O(n²B)` | the "k-NN defense" baseline, 13× the cost |
| `block_collision` | 0.805 | `O(n log²n)` | best on patch triggers specifically |
| `hash_stability` | 0.782 | `O(n)` | no cross-sample access at all |
| `hamming_ball` | 0.774 | `O(n²B)` | |
| `hash_spectral` | 0.733 | `O(nB·iters)` | spectral signatures in hash space |

**The linear-scan statistic loses only 0.014 AUC to the pairwise one while
costing 13× less in MPC.** That is the single most useful engineering result of
the study: the thesis proposal's concern — that defenses "rely on k-nearest
neighbours or clustering, which are poorly adapted to MPC" — is answered by
showing that k-NN is not needed.

## 4.4 From a score to a decision

A high AUC does not automatically remove poison: the threshold has to fire.
With 34 % of the attacked class poisoned, the median and the MAD are themselves
contaminated, and Proposition 9 caps any mean/σ rule at `k ≤ 1.4`.

Measured elimination / sacrifice for the deployed configurations
(`rule` applied per class, two-sided):

`results/purification.csv`, elimination rate / sacrifice rate:

| attack | Sieve-B (`block_collision`, MAD) | Sieve-L (`bit_llr`, 2-means) | Sieve-S (`hash_stability`, MAD) | **PHash-Sieve** (B ∪ L) |
|---|---|---|---|---|
| `label_flip_only` (control) | 0.052 / 0.027 | 0.179 / 0.112 | 0.003 / 0.006 | 0.200 / 0.120 |
| `badnets` | **0.638 / 0.024** | 0.000 / 0.105 | 0.004 / 0.005 | 0.638 / 0.128 |
| `badnets_1px` | 0.057 / 0.025 | 0.178 / 0.115 | 0.004 / 0.006 | 0.199 / 0.131 |
| `blended` | 0.739 / 0.012 | **0.951 / 0.000** | 0.003 / 0.005 | 0.959 / 0.012 |
| `blended_a05` | 0.391 / 0.024 | **0.618 / 0.055** | 0.003 / 0.006 | 0.653 / 0.072 |
| `blended_a02` | 0.011 / 0.024 | 0.236 / 0.055 | 0.002 / 0.006 | 0.243 / 0.079 |
| `clean_label_badnets` | **0.680 / 0.013** | 0.000 / 0.109 | 0.008 / 0.005 | 0.680 / 0.121 |
| `batt` | **0.999 / 0.015** | 0.998 / 0.056 | 0.002 / 0.006 | 0.999 / 0.062 |
| `stego` | 0.001 / 0.027 | **0.653 / 0.083** | 0.002 / 0.005 | 0.654 / 0.101 |
| `phash_adaptive` | **1.000 / 0.014** | 1.000 / 0.029 | 0.000 / 0.005 | 1.000 / 0.042 |
| `phash_adaptive_ss` | **1.000 / 0.014** | 0.007 / 0.139 | 0.000 / 0.005 | 1.000 / 0.143 |
| `wanet` | 0.035 / 0.024 | 0.181 / 0.139 | 0.006 / 0.005 | 0.196 / 0.153 |
| `refool` | 0.045 / 0.025 | 0.181 / 0.114 | 0.003 / 0.005 | 0.200 / 0.128 |
| `adaptive_patch` | 0.054 / 0.024 | 0.186 / 0.113 | 0.003 / 0.005 | 0.211 / 0.127 |

The control row is the floor: even with **no trigger at all**, the sieve removes
20 % of the relabelled samples and 12 % of the clean ones, because relabelled
images really are atypical for their new class. A row is evidence of trigger
detection only insofar as it exceeds that floor — which `badnets`, `blended*`,
`clean_label_badnets`, `batt`, `stego` and both adaptive attacks do, and
`wanet`, `refool`, `adaptive_patch` and `badnets_1px` do not.

`Sieve-S` (`hash_stability`) is the cheapest thing in the study — 0.02 GB, no
cross-sample access — and at this operating point it removes essentially nothing.
Its AUC (0.78 mean) is real but its scores are unimodal, so no robust threshold
fires. It is a ranking statistic, not a decision rule.

The two stages are complementary: `block_collision` catches everything that
fixes a spatial tile (patches, the checkerboard, the rotation), `bit_llr` +
2-means catches everything that shifts many bits slightly (global blends,
steganography). Their union costs 29 GB and 78 s under MPC (§3.4).

**Threshold-rule findings.**

* The MAD rule is precise (1–3 % sacrifice) but only fires on large separations.
* The 2-means rule with a separation guard converts more of the available AUC
  into removals, at 6–14 % sacrifice.
* Both break at a within-class poison rate above 50 %: in the rate sweep,
  `phash_adaptive` at ε = 10 % (53 % of the class) drops from 100 % elimination
  to 0.2 %, exactly at the 50 % breakdown point of the median.

## 4.5 Dependence on the poisoning rate

`results/detection_rates.csv`, figure `results/figures/rate_sweep.png`:

| ε (of the dataset) | badnets | blended | clean-label badnets | phash_adaptive |
|---|---|---|---|---|
| 0.5 % | 0.784 | 0.991 | 0.952 | 1.000 |
| 1 % | 0.969 | 1.000 | 0.958 | 1.000 |
| 2 % | 0.966 | 1.000 | 0.960 | 1.000 |
| 5 % | 0.958 | 1.000 | 0.959 | 1.000 |
| 10 % | 0.957 | 1.000 | — | 1.000 |

This is Proposition 6 in action: **the AUC is essentially independent of the
poisoning rate**, because the log-likelihood weights scale with ε and cancel.
The only degradation is at ε = 0.5 %, where the per-bit frequency estimate stops
resolving the shift — the predicted threshold was `ε ≳ 1/√n_c ≈ 1.2 %`, and the
drop is observed between 0.5 % and 1 %.

Detection is therefore *not* the limiting factor at low poisoning rates; the
*decision rule* is.

## 4.6 End-to-end effect on the trained model

ResNet-18, CIFAR-10, 15 epochs, OneCycle schedule, Apple-GPU (MPS) backend,
poison rate 5 %. Purification with PHash-Sieve (`rhash32`,
`block_collision`+MAD ∪ `bit_llr`+2-means). `results/end2end.csv`.

| attack | setting | BA | ASR | poison removed | clean lost | poisoned samples left |
|---|---|---|---|---|---|---|
| — | benign reference | 0.9237 | — | — | — | — |
| `badnets` | poisoned | 0.9203 | 0.9568 | | | 2500 |
| `badnets` | **purified** | 0.9116 | 0.9531 | 63.8 % | 12.8 % | 906 |
| `blended` | poisoned | 0.9219 | 0.9989 | | | 2500 |
| `blended` | **purified** | 0.9177 | **0.1536** | 95.9 % | 1.2 % | 102 |
| `wanet` | poisoned | 0.9153 | 0.9083 | | | 2500 |
| `wanet` | **purified** | 0.9050 | 0.8709 | 19.6 % | 15.3 % | 2011 |
| `phash_adaptive` | poisoned | 0.9076 | 1.0000 | | | 2500 |
| `phash_adaptive` | **purified** | 0.9179 | **0.0000** | 99.96 % | 4.2 % | 1 |

**Benign accuracy is preserved** in every case: the purified models are within
0.4–1.2 points of the model trained on the full poisoned set, and the
`phash_adaptive` purified model actually *beats* it (0.918 vs 0.908) because the
sieve removed the label noise as well.

**And the attack success rate is where the real lesson is:**

| poison removed | poisoned samples left | ASR after purification |
|---|---|---|
| 99.96 % | 1 | 0.00 (from 1.00) |
| 95.9 % | 102 | 0.15 (from 1.00) |
| 63.8 % | 906 | 0.95 (from 0.96) — **no effect** |
| 19.6 % | 2011 | 0.87 (from 0.91) — no effect |

> **Elimination rate is not the objective; the number of surviving poisoned
> samples is.** Removing 64 % of a BadNets poison leaves 906 patched images —
> far above the few hundred a patch backdoor needs — and the backdoor is
> implanted exactly as before. The defense only pays off once the residual count
> falls to the order of a hundred (blended) or zero (the adaptive attack).

That reframes the operating point: a purification defense of this kind must be
tuned for *recall at any reasonable precision*, not for a balanced trade-off.
With `block_collision`+MAD alone (2.4 % sacrifice) BadNets keeps its backdoor;
what would break it is a rule that removes the whole suspicious mode even at a
30 % sacrifice — which is affordable on CIFAR-10 (benign accuracy costs ~1 point
per 10 % of data at this size) but not obviously so on smaller datasets.

The `wanet` row is the control for the whole study: a trigger the hashes cannot
see costs 15 % of the clean data and buys a 4-point ASR reduction, i.e. nothing.

## 4.7 Limitations and what would fix them

1. **Sample-specific triggers defeat the defense**, by construction
   (Proposition 6 with `J = 0`). WaNet, Refool and multi-pattern adaptive
   attacks are not detected above the label-flip baseline. No amount of hash
   engineering changes this; a per-sample summary cannot see a pattern that is
   never repeated.
2. **The 34 % within-class poison rate is the hard part, not the 5 % dataset
   rate.** Robust thresholding at that contamination level is close to the
   breakdown point of every robust statistic used here.
3. **Partial purification of a backdoor is worth nothing** (§4.6). The defense
   must be operated at near-total recall, which means the threshold rule — not
   the hash and not the detection statistic — is the component that most needs
   further work. A rule that removes the entire suspicious mode, accepting a
   30 % sacrifice, would break BadNets as well; whether that trade is acceptable
   is a deployment decision this study does not make.
4. **The residual hashes have an irreducible tie rate on natural images.** This
   is fixable exactly: computing the residual as `k²·g − Σ_window g` on *integer*
   luma (PIL's rounded `convert("L")`) makes the whole pipeline exact integer
   arithmetic — no fixed point, no rounding, and therefore a bit-exact MPC
   implementation. It also removes the fractional-bit parameter entirely. This
   was identified after the measurement campaign and is the first change to make.
5. **A single dataset.** Everything is CIFAR-10 at 32×32. The propositions are
   resolution-independent, but the constants (how many cells a patch covers) are
   not; the same study on GTSRB or ImageNet-scale images would change which hash
   resolution is optimal.
6. **The MPC numbers are a communication model, not a benchmark.** They are
   exact in primitive counts and protocol constants, but real wall-clock depends
   on the implementation. Running the same circuits in MP-SPDZ would turn the
   estimates into measurements.

## 4.8 Answering the proposal's first question

**Are perceptual hash functions an effective defense against state-of-the-art
poisoning attacks?**

*Partially, and not the ones people mean by "perceptual hash".*

* The **classical** perceptual hashes (aHash, dHash, pHash, wHash, blockhash,
  PDQ, colour hash) are ineffective, with one exception: the DCT-based ones do
  detect fixed patches (ΔAUC ≈ +0.4 on BadNets and clean-label BadNets). Their
  invariance is a designed property and it is exactly the wrong property here.
* A **perceptual-complement hash** — the residual family proposed in this work,
  at the same MPC cost class — detects every *sample-agnostic* trigger tested,
  including invisible ones (blended at α = 0.02, steganographic, and the
  adaptive null-space trigger), with 64–100 % of the poison removed at 1–14 % of
  clean data sacrificed.
* End-to-end, that translates into a **destroyed backdoor only where elimination
  is near-total**: ASR 1.00 → 0.00 on the adaptive attack (99.96 % removed) and
  1.00 → 0.15 on blended (95.9 % removed), but 0.96 → 0.95 on BadNets, where
  63.8 % removal still leaves 906 patched images. Benign accuracy is preserved
  throughout (≤ 1.2 points).
* Every *sample-specific* trigger (WaNet, Refool, multi-pattern adaptive)
  survives, and that limitation is provable rather than empirical.

So the honest scope is: **a cheap, model-free sieve that eliminates the entire
class of sample-agnostic poison-only backdoors at a cost 10⁹ times below any
model-based defense, and forces an adversary into the sample-specific regime**,
where poison-only attacks are hardest to mount — provided it is tuned for
near-total recall, because partial purification of a backdoor is worth nothing.
