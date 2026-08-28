# 3. MPC analysis

*Second question of the thesis proposal: are the chosen perceptual hash functions
effective and suitable for use in MPC?*

Everything below is produced by `scripts/run_mpc_bench.py`, which (a) executes
every hash on an instrumented secret-sharing type and checks the resulting bits
against the plaintext implementation, and (b) prices the resulting primitive
counts under two concrete protocols. Raw output:
`results/mpc_correctness.csv`, `results/mpc_hash_cost.csv`,
`results/mpc_pipeline_cost.csv`.

---

## 3.1 Cost model

**Setting.** Semi-honest adversary, secret sharing over the ring `Z_2^64`,
fixed-point with `f` fractional bits. Two protocols are priced:

| primitive | 3PC replicated (ABY3-style) | 2PC SPDZ-style |
|---|---|---|
| add / sub / mul by public constant | **free** (local) | **free** (local) |
| ring multiplication | 3·64 bits, 1 round | 4·64 bits, 1 round |
| binary AND gate | 3 bits, 1 round | 4 bits, 1 round |
| bit → arithmetic | 3·64 bits, 1 round | 2·64 bits, 1 round |
| truncation by `f` | **free** (local probabilistic truncation) | 2·64 bits, 1 round |
| comparison / MSB, width `w` | `w⌈log₂w⌉ − w + 1` AND gates, `⌈log₂ w⌉` rounds | idem |
| equality test, width `w` | `w − 1` AND gates, `⌈log₂ w⌉` rounds | idem |

Comparisons are costed with a Kogge–Stone parallel-prefix adder; a 64-bit
comparison is therefore 321 AND gates and 6 rounds. Offline (triple generation)
is reported separately and is zero for the 3PC replicated protocol, which uses
only correlated randomness.

**Networks.** LAN = 0.2 ms round-trip, 1 Gbit/s. WAN = 40 ms round-trip,
100 Mbit/s. Wall clock `≈ rounds · RTT + (bits / parties) / bandwidth`, with
perfect batching inside each depth level.

**Caveat.** This is a *communication* model. Real systems differ in local compute,
vectorisation and preprocessing amortisation; the absolute seconds should be read
as order-of-magnitude, while the *ratios* between methods are protocol-determined
and robust.

---

## 3.2 Correctness of the MPC formulation

Every hash was re-implemented on the traced fixed-point type and compared
bit-for-bit with the plaintext (float64) implementation:

| fractional bits `f` | hashes reaching 100 % bit agreement |
|---|---|
| 8 | none (0.93 – 0.999 agreement) |
| 12 | ahash, ahash16, mhash, dhash, dhash16, phash, whash_haar, blockhash |
| 16 | + phash_mean, pdq, rhash8 |
| 20 | + rhash16, rhash_energy |
| 24 | all except `rhash32` (99.999 %) |

`rhash32` never reaches exactly 100 %: its bit is `sign(x − blur(x))` at
full resolution, and in flat image regions the residual is *exactly zero*, so the
bit is decided by the last rounding unit. This is a tie, not an error — the
plaintext bit is equally arbitrary there.

**Recommendation: `f = 16` fractional bits.** With 8-bit pixel inputs and the
separable formulation (no accumulation over more than 64 terms), 16 fractional
bits keep every intermediate below `2^{45}`, far from the `2^{63}` ring bound,
and reproduce all but the tie bits.

---

## 3.3 Per-image cost of each hash

3PC replicated, LAN, `f = 16`, per image (`results/mpc_hash_cost.csv`):

| hash | bits | threshold | comparisons | mults | b2a | rounds | AND gates | online | time |
|---|---|---|---|---|---|---|---|---|---|
| `ahash` | 64 | mean | 64 | 0 | 0 | 6 | 20 544 | 7.7 kB | 1.22 ms |
| `dhash` | 64 | pairwise | 64 | 0 | 0 | 6 | 20 544 | 7.7 kB | 1.22 ms |
| `phash_mean` | 64 | mean | 64 | 0 | 0 | 6 | 20 544 | 7.7 kB | 1.22 ms |
| `rhash8` | 64 | zero | 64 | 0 | 0 | 6 | 20 544 | 7.7 kB | 1.22 ms |
| `ahash16` / `dhash16` / `rhash16` | 256 | mean / pairwise / zero | 256 | 0 | 0 | 6 | 82 176 | 30.8 kB | 1.28 ms |
| `rhash32` | 1024 | zero | 1024 | 0 | 0 | 6 | 328 704 | 123 kB | 1.53 ms |
| `rhash_energy` | 256 | mean (+ `abs`) | 1280 | 1024 | 1024 | 14 | 410 880 | 203 kB | 3.34 ms |
| `mhash` = `whash_haar` | 64 | median | 2080 | 0 | 2016 | 10 | 648 096 | 291 kB | 2.78 ms |
| `phash` | 64 | median | 2080 | 0 | 2016 | 10 | 648 096 | 291 kB | 2.78 ms |
| `blockhash` | 256 | median (4 bands) | 8320 | 0 | 8064 | 10 | 2 592 384 | 1.17 MB | 5.11 ms |
| `pdq` | 256 | median | 32 896 | 0 | 32 640 | 11 | 10 484 608 | 4.72 MB | 14.8 ms |
| `mhash` (median by Batcher sort) | 64 | median | 607 | 543 | 543 | **4350** | 194 847 | 99 kB | 870 ms |

### The four findings that matter

**(1) The transforms are free; only the threshold costs anything.**
`phash` and `phash_mean` compute exactly the same 32×32 DCT-II. Replacing the
median threshold by a mean threshold makes the hash **38× cheaper**
(291 kB → 7.7 kB per image) at the same 64-bit length. The DCT itself — the part
everyone assumes is expensive — contributes *zero* communication. The same holds
for the wavelet transform, the box blur and every resampling step.

**(2) Median thresholding is the dominant cost of classical perceptual hashing.**
It is quadratic in the hash length: `B(B−1)/2` comparisons. PDQ, the most
"modern" hash in the zoo, is the most expensive by a factor of 600 relative to
aHash, purely because it takes a median over 256 values.

**(3) Rank-based median beats a sorting network by 435× in rounds.**
`[v_j > median]` does *not* require the median: it is equivalent to
`rank_j ≥ B/2`, and all `B(B−1)/2` pairwise comparisons are independent, so the
depth is two comparison layers instead of `O(log²B)` sequential compare-exchange
layers. Batcher sorting uses 3.3× fewer gates (99 kB vs 291 kB) but needs 4350
rounds instead of 10 — 870 ms vs 2.8 ms on a LAN, and 174 s vs 0.4 s per image on
a WAN. **For MPC perceptual hashing, always threshold by rank, never by sorting.**

**(4) Non-linear pooling has a real but affordable price.**
`rhash_energy` needs `|·|` at pixel resolution: 1024 comparisons + 1024
multiplications, i.e. 6.6× the cost of the linear `rhash16`. That is the price of
immunity to the null-space adaptive attack of Proposition 5 — cheap compared to
what it buys.

The one hash that does not fit the framework is `colorhash`: it counts histogram
bins rather than thresholding a linear map, so it needs `n_pixels` secure
comparisons *plus* oblivious counting, and it produces only 42 bits. It is
dominated on both axes.

---

## 3.4 Cost of the whole pipeline on a 50 000-sample CIFAR-10 training set

3PC replicated (`results/mpc_pipeline_cost.csv`):

| component | rounds | online comm. | LAN | WAN |
|---|---|---|---|---|
| **hashing** `phash_mean` (50k) | 6 | 0.39 GB | 1.0 s | 10.5 s |
| **hashing** `ahash16` / `rhash16` (50k) | 6 | 1.54 GB | 4.1 s | 41 s |
| **hashing** `rhash32` (50k, 1024 bits) | 6 | 6.16 GB | 16.4 s | 165 s |
| **hashing** `phash` (50k) | 10 | 14.6 GB | 39 s | 389 s |
| **hashing** `pdq` (50k) | 11 | 236 GB | 629 s | 6287 s |
| detector `hash_stability` (B=1024) | 14 | 0.02 GB | 0.06 s | 1.1 s |
| detector `bit_llr` (B=1024, opened aggregates) | 17 | 1.24 GB | 3.3 s | 34 s |
| detector `bit_llr` (B=256, weights kept secret) | 26 | 0.32 GB | 0.85 s | 9.5 s |
| detector `block_collision` (225 tiles × 16 bit, B=1024) | 459 | 21.8 GB | 58 s | 598 s |
| detector `hash_spectral` (power iteration, B=256) | 107 | 6.5 GB | 17.2 s | 176 s |
| detector `hamming_ball` (B=256) | 13 | 13.3 GB | 35.5 s | 355 s |
| detector `hamming_knn` (B=1024, k = 10) | 65 | 81.5 GB | 217 s | 2175 s |
| **baseline** pixel k-NN (d = 3072) | 71 | 9 515 GB | 7.0 h | 70 h |
| **baseline** activation clustering (k-means, given features) | 300 | 24.7 GB | 66 s | 671 s |
| **baseline** ResNet-18 inference over 50k images | 60 | 6.7·10⁸ GB | 20.7 d | 207 d |
| **baseline** ResNet-18 training, 30 epochs | 2.1·10⁶ | 6.0·10¹⁰ GB | 5.1 y | 51 y |
| **baseline** Spectral Signatures (train + infer + SVD) | 2.1·10⁶ | 6.1·10¹⁰ GB | 5.2 y | 52 y |

### The deployed configurations

Totals for purifying the whole 50 000-image training set (3PC replicated):

| configuration | hash + detector | online | rounds | LAN | WAN |
|---|---|---|---|---|---|
| **Sieve-S** | `rhash32` + `hash_stability` | 6.18 GB | 20 | 16.5 s | 166 s |
| **Sieve-L** | `rhash32` + `bit_llr` | 7.40 GB | 23 | 19.7 s | 198 s |
| **Sieve-B** | `rhash32` + `block_collision` | 27.9 GB | 465 | 74.5 s | 763 s |
| **PHash-Sieve** | union of Sieve-B and Sieve-L | 29.2 GB | 482 | 77.8 s | 797 s |

### Reading the table

* The full defense costs **29 GB and 78 seconds on a LAN** for a 50 000-image
  dataset. Spectral Signatures — the statistic it replaces — needs 6.1·10¹⁰ GB,
  i.e. **2·10⁹ times more**, because it has to train a ResNet-18 under
  encryption first.
* Even the *naive* MPC-compatible baseline the thesis proposal mentions — k-NN
  in raw pixel space — costs 9.5 TB: **326 × more** than the whole PHash-Sieve
  and **1 290 × more** than Sieve-L, because its distance computation is
  quadratic in `n` *and* linear in the 3072-dimensional input. Running the *same*
  k-NN on 1024-bit hashes instead of 3072-dimensional pixels is 117× cheaper —
  which is the proposal's hypothesis, confirmed.
* `hash_stability` is essentially free (0.02 GB) and has **no cross-sample data
  dependence at all**: no oblivious sorting, no all-pairs distances, perfect
  sharding. It is the weakest detector statistically, but it is the one that
  scales to datasets where `O(n²)` is out of the question.

### Where the cost actually sits

For Sieve-L, 83 % of the communication is the hashing stage, and inside that,
100 % of it is the `B` comparisons of the binarisation. Two consequences:

* Shorter hashes are proportionally cheaper — the cost is exactly linear in `B`
  for `mean`/`zero`/`pairwise` hashes. `rhash8` (64 bits) costs a sixteenth of
  `rhash32` (1024 bits). §4 measures what that costs in detection power, and the
  answer is: almost everything, so the 1024-bit hash is the right choice.
* The detector is essentially free next to the hashing, *provided* it is
  `bit_llr` or `hash_stability`. `block_collision` triples the total (and
  multiplies the round count by 20, which is what hurts on a WAN);
  `hamming_knn` multiplies it by 13; a model-based statistic by 2·10⁹.

---

## 3.5 MPC-specific design decisions

**(a) Threshold by rank, not by sorting** (§3.3, finding 3).

**(b) Prefer mean/zero thresholds.** `phash_mean` is a drop-in replacement for
`phash` that is 38× cheaper; §4 shows it is also *no worse* at detection.

**(c) Open the aggregate bit frequencies.** The `bit_llr` weights
`log(q_j/p_j)` are functions of `2B` per-class aggregates, each an average over
thousands of records. Opening them turns the scoring step into a public linear
form (free) and costs 9 extra rounds if kept secret (a Newton division plus a
polynomial logarithm per bit). The measured difference is small (0.84 s vs
0.85 s), so the private-weight variant should simply be used unless the extra
rounds matter on a WAN. If aggregates *are* opened, they should be released with
differential-privacy noise; the sensitivity of a bit frequency is `1/n_c`.

**(d) The robust threshold rule is the expensive part of the *decision*, not of
the *score*.** The median/MAD rule needs two order statistics over 50 000 values
(an oblivious sort: `O(n log²n)` compare-exchanges ≈ 3.4·10⁶ comparisons per
class). The mean/σ rule is linear and essentially free, but Proposition 9 shows
it cannot fire when `ε > 1/(1+k²)`. Practical compromise, and the one we
recommend: use the **top-fraction rule** — remove the `p`-fraction highest scores
per class, which needs a single `p`-quantile, obtainable from an oblivious sort
*or*, much more cheaply, from a public quantile estimated on a small random
subsample (the score distribution of the clean majority is what is being
estimated, and it is not sensitive to individual records).

**(e) Batch everything.** All 50 000 images hash in the same 6 rounds; the round
count of the whole sieve is dominated by the detector, not by the dataset size.
This is why the WAN penalty is only ~10× rather than proportional to `n`.

---

## 3.6 Answer to the proposal's second question

**Are perceptual hash functions suitable for use in MPC?**

*Yes, decisively, on the cost axis.* Their arithmetic is a public linear map,
which is free under linear secret sharing; the whole online cost is `B`
comparisons per image for a well-chosen threshold. Purifying a 50 000-image
CIFAR-10 training set costs **29 GB and 78 s on a LAN** with the full
PHash-Sieve, or **7.4 GB and 20 s** with its linear-only variant — respectively
2·10⁹ and 8·10⁹ times less than the model-based defenses that currently define
the state of the art, and 326–1 290 times less than the pixel-space k-NN
baseline.

*With three caveats, all of which are design constraints rather than obstacles:*

1. **Choose the threshold, not the transform.** The classical hashes'
   median thresholds cost 30–600× more than necessary; mean- and sign-thresholded
   variants are equally good detectors (§4) and are the ones to deploy.
2. **Avoid pairwise statistics.** `hamming_knn` and `block_collision` are the
   two strongest detectors on patch triggers but they are the two that need
   `O(n²)` comparisons or oblivious sorting. `bit_llr` gets most of the
   detection power for 1–2 % of the cost.
3. **The robust decision rule is the residual difficulty.** Median/MAD is
   statistically necessary (Proposition 9) and is the only part of the pipeline
   that requires an order statistic over the whole dataset.

The efficiency answer is therefore unambiguous. Whether the hashes are
*effective* — the first question of the proposal — is a different matter, and is
answered in `docs/04_results.md`.
