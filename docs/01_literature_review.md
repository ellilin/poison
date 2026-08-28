# 1. Literature review

*Scope: what a poisoning adversary can do, what current defenses do about it, why
those defenses do not survive a translation into multi-party computation, and
what perceptual hash functions are.*

---

## 1.1 Data poisoning and backdoor attacks

A poisoning adversary manipulates the *training data* rather than the model.
Two broad goals are distinguished:

* **Availability (indiscriminate) poisoning** — degrade accuracy on all inputs.
* **Targeted / backdoor poisoning** — leave benign accuracy untouched and force a
  chosen output whenever an adversary-chosen *trigger* is present. This is the
  regime the thesis proposal targets, and the one BackdoorBox implements.

Backdoor attacks are usually characterised along four axes:

| axis | values | why it matters for a hash-based defense |
|---|---|---|
| **capability** | poison-only vs training-controlled | a data filter only makes sense against poison-only attacks; a training-controlled adversary can ignore the filter |
| **label** | dirty-label (label flipped to the target) vs clean-label | clean-label attacks bypass any label-consistency check |
| **visibility** | visible patch vs invisible perturbation | a perceptual hash is *designed* to ignore invisible perturbations |
| **trigger locality** | sample-agnostic (identical trigger) vs sample-specific | a hash-collision argument only works for sample-agnostic triggers |

### Reference attacks

* **BadNets** (Gu, Dolan-Gavitt, Garg, *IEEE Access* 2019; arXiv 2017) — the first
  backdoor attack: stamp a small patch (a 3×3 white square) onto a fraction of the
  training images and relabel them. Sample-agnostic, visible, dirty-label.
* **Blended** (Chen et al., arXiv 2017) — blend a whole-image pattern (a fixed
  noise image, or "Hello Kitty") with weight α. As α → 0 the attack becomes
  invisible while remaining learnable; this is the canonical *global invisible*
  attack.
* **Label-consistent backdoor** (Turner, Tsipras, Madry, 2019) — perturb *only*
  target-class images adversarially so the network must rely on the patch; the
  labels stay correct, defeating every label-noise heuristic.
* **Clean-label video attack** (Zhao et al., *CVPR* 2020) — clean-label with a
  universal-adversarial-perturbation trigger (`TUAP` in BackdoorBox).
* **Refool** (Liu et al., *ECCV* 2020) — the trigger is a physically plausible
  *reflection* composited into the image; natural-looking and sample-specific.
* **WaNet** (Nguyen & Tran, *ICLR* 2021) — the trigger is a smooth elastic
  *warping field*. No pixel pattern is added at all; the perturbation is
  invisible and its pixel-level realisation depends on the image content.
* **ISSBA** (Li et al., *ICCV* 2021) — a trained StegaStamp encoder embeds the
  same secret string into every poisoned image, producing a *sample-specific
  invisible* residual. The first poison-only sample-specific attack.
* **IAD** (Nguyen & Tran, *NeurIPS* 2020) and **LIRA** (Doan et al., *ICCV* 2021)
  — training-controlled attacks that *learn* the trigger generator jointly with
  the model.
* **Blind backdoors** (Bagdasaryan & Shmatikov, *USENIX Security* 2021) — the
  adversary controls the loss computation, not the data.
* **Sleeper Agent** (Souri et al., *NeurIPS* 2022) — gradient-matching clean-label
  poisoning that survives training from scratch.
* **BATT** (Xu et al., *ICASSP* 2023) — the trigger is a *transformation*
  (a fixed rotation), so no additive pattern exists to be filtered.
* **Adaptive attacks against latent separability** (Qi et al., *ICLR* 2023) —
  several low-opacity patches plus "cover" samples that carry the trigger with a
  correct label, explicitly constructed so the poisoned samples do **not** form a
  separable cluster in feature space. This is the attack that broke the
  generation of defenses described next, and it is the reason any new defense
  must be evaluated against an adaptive adversary.

The standard metrics are **BA** (benign accuracy on clean test data) and **ASR**
(attack success rate: fraction of non-target-class test images that are
classified as the target once the trigger is applied).

---

## 1.2 Defenses, and why they do not move to MPC

Following the taxonomy used by BackdoorBox:

**(a) Sample pre-processing** — destroy the trigger before inference.
Neural Trojans / autoencoder defense (Liu et al., *ICCD* 2017), ShrinkPad
(Li et al., *ICLR-W* 2021), REFINE (*ICLR* 2025). *Cost in MPC*: needs a
neural network at inference time (autoencoder, reprogramming network), i.e.
millions of secure multiplications and comparisons per sample.

**(b) Model repairing** — Fine-Pruning (Liu et al., *RAID* 2018), MCR
(Zhao et al., *ICLR* 2020), NAD (Li et al., *ICLR* 2021). *Cost in MPC*: requires
a trained model plus clean data plus additional training.

**(c) Poison suppression during training** — ABL (Li et al., *NeurIPS* 2021),
DBD (Huang et al., *ICLR* 2022), CutMix-based schemes. *Cost in MPC*: modifies
the training loop, so the entire training must run under encryption.

**(d) Input-level detection at inference** — STRIP (Gao et al., *ACSAC* 2019),
SCALE-UP (Guo et al., *ICLR* 2023), IBD-PSC (Hou et al., *ICML* 2024).
*Cost in MPC*: many forward passes per query.

**(e) Dataset purification** — the family our defense belongs to:

| method | statistic | what it needs |
|---|---|---|
| Spectral Signatures (Tran, Li, Madry, *NeurIPS* 2018) | top singular direction of the penultimate-layer features of the attacked class | a **trained model** + SVD |
| Activation Clustering (Chen et al., *AAAI-W* 2019) | 2-means on the same features | a trained model + ICA/PCA + k-means |
| SPECTRE (Hayase et al., *ICML* 2021) | robust covariance estimation + QUE score | a trained model + robust covariance |
| SCAn (Tang et al., *USENIX Security* 2021) | EM decomposition of class-conditional feature distributions | a trained model + EM |
| FLARE (Bai et al., *TIFS* 2025) | latent representations across all layers + cluster analysis | a trained model + clustering |

Every one of them shares the same structural dependency: **the statistic lives in
the feature space of a network that must first be trained on the poisoned data**.
Under secret sharing that means training a CNN inside the MPC engine — the single
most expensive thing one can ask an MPC engine to do (see `docs/03_mpc_analysis.md`,
where a 30-epoch ResNet-18 training is costed at ≈ 6·10⁴ TB of online
communication). On top of that, k-means, SVD and robust covariance estimation all
need data-dependent iteration counts, divisions, square roots and — for k-NN —
oblivious sorting, all of which are the expensive primitives in MPC.

This is exactly the gap identified by the thesis proposal: *"These methods often
rely on k-nearest neighbors or clustering techniques, which are poorly adapted to
MPC settings."*

**Robust aggregation in federated learning** is the closest existing
MPC-compatible defense line (secure aggregation with norm bounding, RoFL, ELSA,
BREA). It defends against malicious *updates*, not against poisoned *samples*,
and it deliberately avoids exactly the operations (medians, k-NN, clustering)
that the sample-level defenses depend on — for the same cost reasons.

---

## 1.3 Perceptual hash functions

A perceptual hash maps an image to a short bit string such that *perceptually
similar images map to similar strings* (small Hamming distance), while unrelated
images map to strings at roughly half the length in Hamming distance. They are
the standard tool for near-duplicate detection, copyright enforcement and
CSAM scanning.

### The classical constructions

| hash | pipeline | bits |
|---|---|---|
| **aHash** (average hash) | luma → 8×8 thumbnail → bit = pixel > mean | 64 |
| **mHash** (median hash) | same, threshold = median | 64 |
| **dHash** (difference hash, Krawetz) | luma → 9×8 thumbnail → bit = sign of the horizontal gradient | 64 |
| **pHash** (DCT hash, Zauner 2010) | luma → 32×32 → DCT-II → 8×8 low-frequency block → bit = coefficient > median | 64 |
| **wHash** (wavelet hash) | luma → 32×32 → Haar/db4 DWT → LL band → bit = coefficient > median | 64 (169 for db4 on a 32×32 input) |
| **blockhash** (blockhash.io) | 16×16 block means, thresholded by the median of their horizontal band | 256 |
| **Marr–Hildreth hash** (Zauner 2010) | Laplacian-of-Gaussian response pooled on a grid | 72 |
| **colour hash** | fractions of black / gray / hue-binned pixels, quantised | 42 |
| **PDQ** (Facebook ThreatExchange, 2019) | Jarosz box blur → 64×64 luma → DCT → 16×16 mid-frequency band → median threshold | 256 |
| **PhotoDNA** (Microsoft) | proprietary; gradient histograms over a grid | 144 bytes |
| **NeuralHash** (Apple, 2021) | learned CNN embedding → LSH projection | 96 |

A high-level introduction to the family is the survey referenced by the thesis
proposal: <https://tsjournal.org/index.php/jots/article/view/24/14>.

### Security properties (and non-properties)

Perceptual hashes are **not** cryptographic hashes. They are explicitly designed
to leak similarity, and the literature has repeatedly shown that they are easy to
attack:

* Struppek et al., *"Learning to Break Deep Perceptual Hashing: The Use Case
  NeuralHash"*, **FAccT 2022** — gradient-based collisions and evasion for
  NeuralHash.
* Jain et al., *"Adversarial Detection Avoidance Attacks"*, **USENIX Security
  2022** — black-box evasion of perceptual-hash-based client-side scanning.
* Prokos et al., *"Squint Hard Enough: Attacking Perceptual Hashing with
  Adversarial Machine Learning"*, **USENIX Security 2023** — preimage and
  detection-avoidance attacks against PDQ and PhotoDNA.
* Abelson et al., *"Bugs in Our Pockets: The Risks of Client-Side Scanning"*,
  *Journal of Cybersecurity* 2024 — systemic critique of deploying perceptual
  hashing for content scanning.

The relevant lesson for this thesis is not that perceptual hashes are broken as
scanners, but that **their invariance is a two-sided property**: the same
robustness that makes them useful for near-duplicate detection makes them blind
to small perturbations — and an invisible backdoor trigger *is* a small
perturbation. Section 2 of `docs/02_theory.md` turns this observation into a
quantitative statement, and the experiments confirm it.

### Perceptual hashing has not been used against poisoning

To the best of the survey performed here, perceptual hashes have been used in ML
security only as *the thing being attacked* (client-side scanning), never as a
*defense mechanism for training data*. The nearest neighbours in the literature
are:

* near-duplicate removal in dataset curation (LAION-style deduplication), which
  uses perceptual or learned hashes but targets redundancy, not adversarial
  structure;
* "Data poisoning of web-scale datasets" (Carlini et al., *IEEE S&P* 2024), which
  observes that hash-based dataset integrity checks would catch *split-view*
  poisoning — the closest published statement that hashing is relevant to
  poisoning, but for exact-content integrity, not perceptual similarity.

---

## 1.4 Secure multi-party computation for ML

**Sharing schemes.** Additive/replicated secret sharing over a ring `Z_2^k`
(Araki et al., *CCS* 2016; ABY3, Mohassel & Rindal, *CCS* 2018) or over a field
(SPDZ, Damgård et al., *CRYPTO* 2012; SPDZ2k, Cramer et al., *CRYPTO* 2018).
Frameworks: ABY (Demmler et al., *NDSS* 2015), SecureML (*S&P* 2017), SecureNN
(*PETS* 2019), Falcon (*PETS* 2021), CrypTen (*NeurIPS* 2021), MP-SPDZ
(Keller, *CCS* 2020).

**The cost structure that drives every design decision.**

* Addition, subtraction and multiplication **by a public constant** are *local*:
  zero communication, zero rounds. Therefore any public linear map — a
  convolution with public weights, a DCT, a wavelet transform, an average pooling
  — is free.
* Multiplication of two secrets costs one round and O(k) bits (a Beaver triple).
* **Comparisons are the expensive primitive.** A `<`/`>`/`MSB` test requires
  bit-decomposition, implemented with a parallel-prefix adder: ≈ `k·log k` AND
  gates and `log k` rounds for a `k`-bit ring. Mixed arithmetic/binary
  conversions (`edaBits`, Escudero et al., *CRYPTO* 2020) are the standard way to
  make them affordable.
* **Order statistics are worse.** A median needs an oblivious sort — Batcher's
  odd-even mergesort, `O(n log²n)` compare-exchanges, each one a comparison *and*
  a multiplication — or a rank computation, `O(n²)` comparisons.
* Fixed-point arithmetic needs truncation after every multiplication; in
  3-party replicated sharing this can be done locally with a probabilistic
  1-ulp error (Mohassel & Rindal), which is why 3PC is the natural setting here.

**Poisoning and MPC together.** Privacy-preserving training (medical, financial
consortia) is precisely the setting where the training data cannot be inspected —
and therefore precisely the setting where poison filtering must run under
encryption. The published intersection is thin: secure aggregation with robust
statistics in federated learning (Bell et al., BREA, RoFL, ELSA) and
"cryptographically-enforced input validation" (Prio-style range proofs). None of
these operate at the level of individual training *images*.

---

## 1.5 Summary of the gap this work addresses

1. The strongest dataset-purification defenses all require a model trained on the
   poisoned data, which is the single most expensive object to compute under MPC.
2. Perceptual hashes are extremely cheap in exactly the MPC cost model above:
   their entire arithmetic is a public linear map, and only the final
   binarisation costs communication.
3. Nobody has measured whether the information that survives a 64–256 bit
   perceptual hash is sufficient to separate poisoned from clean samples.

That measurement — across 13 attacks, 26 hashes and 7 detection statistics, plus
a full MPC cost model — is the content of the rest of this report.
