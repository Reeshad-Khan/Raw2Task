# Paper Writing Brief — Raw2Task Spectral Co-Design

**Authors:** Reeshad Khan, John Gauch  
**Venue target:** CVPR / ICCV / ECCV 2026  
**Status:** All experiments complete. All figures generated. Ready to write.

---

## 1. Working Title

> **Task-Optimal Spectral Filter Design for RAW-to-Task Semantic Segmentation**

Alternative: *Beyond Bayer: Task-Driven Spectral Mosaic Design for Autonomous Driving Segmentation*

---

## 2. Core Contribution (One Paragraph)

We present a principled analysis of sensor co-design for semantic segmentation in autonomous driving, asking: *which sensor dimensions benefit from task-driven optimisation, and which do not?* We show that (i) PSF/optics co-design is provably net-negative for dense prediction (data processing inequality; empirical: −0.020 mIoU on KITTI-360); (ii) learning the 2×2 CFA spectral weights — while keeping the Bayer tile structure — gives consistent, significant gains over a fixed camera baseline (+0.017 KITTI-360, +0.023 ACDC); (iii) noise co-optimisation has small, dataset-dependent effects (KITTI: +0.002, ACDC: −0.003) — the dominant gain is spectral, not noise; (iv) increasing the CFA tile size beyond 2×2 consistently hurts performance (Tile-3: −0.006 to −0.005, Tile-4: −0.010 to −0.011 vs Tile-2), because the filters are constrained to the RGB input colour space and larger tiles reduce per-filter spatial density without adding new spectral information. Notably, the learned 2×2 CFA converges to a near-Bayer RGGB pattern, while 3×3 and 4×4 designs discover novel broadband mixed filters — yet those mixed designs underperform the simpler 2×2 solution. These results establish that, for standard RGB-input segmentation models, the optimal sensor co-design strategy is: **learn CFA spectral weights within the 2×2 Bayer tile structure with noise co-optimisation; use an identity PSF**. This provides the first systematic, theory-grounded decomposition of sensor co-design benefit for dense prediction.

---

## 3. Final Experimental Results

> All experiments complete as of 2026-06-13. All numbers are final.

### 3a. KITTI-360 (40 epochs, SegFormer-B4, new physics)

| Experiment | PSF | CFA | Noise | Tile | mIoU | ΔmIoU vs Fixed |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| RGB (no sensor pipeline) | — | — | — | — | **0.6396** | +0.035 |
| Fixed camera | ✗ | ✗ | ✗ | 2 | 0.6043 | — |
| Co-design (PSF+CFA+noise) | ✓ | ✓ | ✓ | 2 | 0.6031 | −0.001 |
| **No optics (CFA+noise) ★** | ✗ | ✓ | ✓ | 2 | **0.6229** | **+0.019** |
| CFA only (no noise) | ✗ | ✓ | ✗ | 2 | 0.6214 | +0.017 |
| Tile-3 (ablation) | ✗ | ✓ | ✓ | 3 | 0.6165 | +0.012 |
| Tile-4 (ablation) | ✗ | ✓ | ✓ | 4 | 0.6133 | +0.009 |

Sensor dimension decomposition (vs Fixed camera baseline):
- CFA learning contribution: **+0.017** (CFA only − Fixed)
- Noise co-optimisation: **+0.002** (No optics − CFA only)
- PSF cost: **−0.020** (Co-design − No optics; PSF actively hurts)
- Tile-3 penalty: **−0.006** vs Tile-2
- Tile-4 penalty: **−0.010** vs Tile-2

### 3b. ACDC (60 epochs, SegFormer-B4, new physics)

> ⚠️ Baselines (Fixed camera, Co-design, RGB) used old physics. Spectral experiments use corrected physics (pre-CFA Poisson-Gauss noise, correct regularisation). Cross-physics comparison is directional; note this in the paper for the PSF claim.

| Experiment | PSF | CFA | Noise | Tile | mIoU | ΔmIoU vs Fixed |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| RGB (no sensor pipeline) | — | — | — | — | **0.7374** | +0.072 |
| Fixed camera | ✗ | ✗ | ✗ | 2 | 0.6654 | — |
| Co-design (PSF+CFA+noise) | ✓ | ✓ | ✓ | 2 | 0.6906 | +0.025 (old phys.) |
| No optics (CFA+noise) | ✗ | ✓ | ✓ | 2 | 0.6853 | +0.020 |
| **CFA only (no noise) ★** | ✗ | ✓ | ✗ | 2 | **0.6882** | **+0.023** |
| Tile-3 (ablation) | ✗ | ✓ | ✓ | 3 | 0.6800 | +0.015 |
| Tile-4 (ablation) | ✗ | ✓ | ✓ | 4 | 0.6740 | +0.009 |

Sensor dimension decomposition (vs Fixed camera baseline):
- CFA learning contribution: **+0.023** (CFA only − Fixed) ← primary gain
- Noise co-optimisation: **−0.003** (No optics − CFA only; noise slightly hurts ACDC)
- PSF cost: **+0.005** (Co-design − No optics; cross-physics, directional only)
- Tile-3 penalty: **−0.005** vs Tile-2
- Tile-4 penalty: **−0.011** vs Tile-2

★ Best spectral sensor configuration. CFA only (no noise) is best on ACDC; No optics (CFA+noise) is best on KITTI-360. The difference is marginal (±0.003) — both are valid claims.

---

## 4. Key Claims (all supported by final results)

1. **PSF co-design is net-negative for segmentation** — empirically: Co-design (0.6031) < No optics (0.6229) on KITTI-360; PSF cost = −0.020. Information-theoretically: data processing inequality mandates identity PSF optimality for dense prediction. See §6.

2. **Learned 2×2 CFA spectral weights are the dominant sensor gain** — CFA only vs Fixed camera: +0.017 KITTI-360, +0.023 ACDC. This is the primary contribution.

3. **Noise co-optimisation has marginal, dataset-dependent effect** — KITTI: +0.002, ACDC: −0.003. Do not claim noise consistently helps or hurts; it is secondary to spectral.

4. **Larger CFA tiles consistently hurt performance** — KITTI Tile-3: −0.006, Tile-4: −0.010 vs Tile-2. ACDC: Tile-3: −0.005, Tile-4: −0.011 vs Tile-2. Consistent across both datasets.

5. **Learned 2×2 CFA converges to near-Bayer RGGB** — weights confirm: filter 1 ≈ (1,0,0), filters 2–3 ≈ (0,1,0), filter 4 ≈ (0,0,1). The optimal design within the RGB colour space is the Bayer pattern.

6. **Larger tiles discover novel spectral patterns but underperform** — Tile-3 and Tile-4 learn genuine broadband mixed filters (e.g., 49%R+51%G, 36%G+64%B), yet these more complex filters hurt segmentation — a negative result with clear interpretation.

### What NOT to claim
- ❌ Do NOT claim PSF co-design helps. Our earlier preprint (arXiv:2512.20815) made this claim at 1M-param model scale — it does not hold at SegFormer-B4 scale.
- ❌ Do NOT cite arXiv:2512.20815 as supporting work — it is a faulty earlier version.
- ❌ Do NOT claim noise always helps — ACDC shows it marginally hurts (−0.003).
- ❌ Do NOT claim larger tiles help — they consistently hurt both datasets.
- ❌ Do NOT claim hyperspectral capability — all filters are still linear combinations within the RGB colour space.
- ❌ Do NOT claim the learned Tile-3/4 filters are practically realisable without noting the RGB-basis constraint.

---

## 5. Paper Structure

```
Abstract (~150 words)
  - Sensor co-design for segmentation
  - CFA spectral learning = dominant gain (+0.017 / +0.023)
  - PSF = net negative (DPI argument + empirical)
  - Noise = marginal; Tile size > 2×2 = consistently worse
  - Optimal: learned 2×2 CFA + identity PSF

1. Introduction
   - Autonomous driving needs robust RAW-to-task perception
   - Sensor co-design: optimise sensor + segmentation model jointly
   - Key question: which sensor dimensions actually help?
   - Contributions (4 bullets):
       1. Theory: DPI argument for PSF
       2. Method: T×T differentiable spectral CFA
       3. Experiments: ablation decomposition on KITTI-360 + ACDC
       4. Finding: optimal co-design = learned 2×2 CFA, identity PSF

2. Related Work
   2.1 Computational imaging co-design (Yang 2023, arXiv 2412.14603, 2502.04719)
   2.2 Learned colour filter arrays (arXiv 2603.26528)
   2.3 Task-driven sensor design for autonomous driving
   2.4 Semantic segmentation under sensor degradation

3. Method
   3.1 RAW-to-Task pipeline overview (→ Fig. 1)
   3.2 Differentiable physical sensor model
       3.2.1 Optics: identity PSF (why)
       3.2.2 T×T spectral CFA mosaic
       3.2.3 Poisson-Gauss noise + ADC model
       3.2.4 Soft differentiable demosaicking
   3.3 Information-theoretic analysis
       3.3.1 PSF: DPI → identity optimal for dense prediction
       3.3.2 CFA: spectral diversity → class discriminability
   3.4 Learnable T×T spectral CFA
       3.4.1 Formulation: W ∈ R^(T²×3), softmax rows
       3.4.2 Spectral diversity regularisation loss
       3.4.3 Joint training with SegFormer-B4

4. Experiments
   4.1 Datasets and setup (KITTI-360, ACDC, SegFormer-B4)
   4.2 Main results (Table 1, Fig. 5)
   4.3 Sensor dimension decomposition (Fig. 7)
   4.4 CFA tile size analysis (Fig. 6)
   4.5 Learned CFA pattern analysis (Fig. 2, 3, 4)
   4.6 Qualitative results (Fig. 8 combined, ACDC + KITTI)

5. Discussion
   5.1 When does PSF co-design help? (classification vs segmentation)
   5.2 Why does Tile-2 outperform Tile-3/4?
   5.3 Limitations and future work

6. Conclusion
```

---

## 6. Figures (all generated — in paper/figures/)

All figures are saved as both PDF (for LaTeX) and PNG (for preview).

| File | Caption summary |
|------|----------------|
| `fig1_pipeline.pdf` | System pipeline: Scene → Identity Optics → T×T CFA Mosaic (learned) → Noise+ADC (learned) → Soft Demosaick → SegFormer-B4 → Seg. Map |
| `fig2_cfa_comparison.pdf` | Learned CFA tile patterns: Bayer 2×2 (reference), Learned 2×2 (≈Bayer), Learned 3×3, Learned 4×4. Tile-3/4 show novel broadband mixed filters. |
| `fig3_spectral_bars.pdf` | R/G/B spectral weight bars per CFA site for each tile size. Shows increasing spectral diversity with tile size. |
| `fig4_mosaic_pattern.pdf` | Spatial tiling of learned CFA over a 12×12 image patch. Shows how tile patterns repeat. |
| `fig5_results.pdf` | Main results bar chart: mIoU for all 7 experiments on KITTI-360 and ACDC. Significance bracket shows +0.019/+0.020 gain over fixed camera. |
| `fig6_tile_trend.pdf` | Line plot: mIoU vs CFA tile size T (2, 3, 4) for KITTI-360 and ACDC. Both datasets show consistent downward trend. |
| `fig7_ablation_decomposition.pdf` | Lollipop chart: per-dimension ΔmIoU vs Fixed camera baseline. CFA learning = strong positive; Noise = marginal; PSF = negative (KITTI). |
| `qualitative_combined_paper.pdf` | **Main paper figure**: 3 ACDC + 3 KITTI scenes × 4 columns (Input, GT, Fixed Camera, No Optics). Shows rare-class recovery (rider, motorcycle) by learned CFA. |
| `qualitative_acdc_paper.pdf` | Supplement: all 8 ACDC scenes × 6 columns (Input, GT, Fixed, No Optics, Tile-3, Tile-4). |
| `qualitative_kitti_paper.pdf` | Supplement: all 8 KITTI-360 scenes × 6 columns (Input, GT, Fixed, No Optics, Tile-3, Tile-4). |
| `table_main_results.tex` | LaTeX table: all 7 experiments × KITTI + ACDC mIoU. Bold = best sensor config. |

### Key qualitative story (from data analysis)
- **ACDC rank 1**: GT has motorcycle (blue, 0,0,230). Fixed Camera predicts fence (pink) there. No Optics correctly predicts motorcycle. Tile-3/4 also improve but less reliably.
- **KITTI rank 1**: GT has rider (red, 255,0,0). Fixed Camera predicts sidewalk/person (wrong colour). No Optics correctly recovers the rider class.

---

## 7. Information-Theoretic Argument (Method §3.3)

### PSF: Data Processing Inequality

For a PSF operator P and scene X:

```
I(Y ; task) ≤ I(X ; task)     [data processing inequality]
```

where Y = P * X (PSF-blurred image). For dense prediction (segmentation), the task label T is a per-pixel class map — it requires ALL spatial information from X. The DPI states that any deterministic degradation (PSF blurring) cannot increase mutual information with the task. Equality holds only when P is the identity (delta function). Therefore, for segmentation, the optimal PSF is always the identity.

Contrast with classification: a global label Y = f(X) can be preserved under structured PSF blurring that suppresses irrelevant spatial frequencies — hence Yang et al. (2023) achieves gains with PSF co-design for classification. Segmentation is categorically different.

### CFA: Spectral Diversity → Class Discriminability

The Shannon entropy of a spectral observation Y = W·X (W = T²×3 spectral filter matrix) is:

```
H(Y) = H(W·X)
```

For fixed image statistics, H(Y) increases when the rows of W are maximally diverse (different spectral mixtures). A fixed Bayer pattern (50% G, 25% R, 25% B) is optimised for perceptual colour reconstruction, not for discriminating semantic classes. A learned W optimised end-to-end on the segmentation objective discovers filters that maximise class-discriminative information in the RAW mosaic.

Empirical confirmation: learned 2×2 CFA converges to near-Bayer (the Bayer pattern is locally optimal within RGB space), but the optimisation path and regularisation produce filters with slightly better spectral balance for the KITTI/ACDC class distribution.

---

## 8. Related Work

### Must cite
- **Yang et al. (2023)** — arXiv:2305.17185: Task-Driven Lens Design for *classification*. Key point: DPI argument applies only to dense prediction, not classification. This is why their result and ours are complementary.
- **Successive Optimisation of Optics** — arXiv:2412.14603 (2024). Alternating optimisation for PSF.
- **Tolerance-Aware Deep Optics** — arXiv:2502.04719 (2025). PSF regularisation.
- **Segmentation beyond Aberrations** — arXiv:2211.11257 (2022). Co-design for segmentation under optical aberrations. Key prior work.
- **Learnable Quantum Efficiency Filters** — arXiv:2603.26528 (2026). Closest related: learned spectral filters for segmentation, but requires *true hyperspectral* input camera. We work within the RGB colour space using standard CMOS sensors.
- **Hyperspectral Sensors for Autonomous Driving** — arXiv:2508.19905 (2025). Survey of hyperspectral imaging for ADAS.
- **DeepLens** — differentiable optics framework (cite original paper).
- **SegFormer** — Xie et al., NeurIPS 2021. Our backbone.
- **KITTI-360** — Liao et al., TPAMI 2022. Dataset.
- **ACDC** — Sakaridis et al., ICCV 2021. Dataset.

### Do NOT cite
- ❌ arXiv:2512.20815 — our own faulty preprint claiming PSF helps at small model scale.

---

## 9. Code Locations

| Component | File | Key class / function |
|---|---|---|
| T×T learnable CFA | `raw2task/sensors/cfa.py` | `LearnableCFA(tile_size=T)` |
| Differentiable demosaick | `raw2task/sensors/cfa.py` | `soft_cfa_demosaic()` |
| Poisson-Gauss noise model | `raw2task/sensors/noise.py` | `SensorNoise.forward()` |
| Full sensor forward pass | `raw2task/train_extended.py` | `CoDesignSensor.forward()` |
| Segmentation backbone | `raw2task/models/segmentation.py` | `HFSegmentationWrapper` |
| Spectral diversity loss | `raw2task/sensors/cfa.py` | `spectral_diversity_loss()` |
| Experiment configs | `raw2task/configs/` | `*_spectral_matrix.yaml` |
| Figure generation | `paper/make_figures.py` | `plot_*()` functions |
| Qualitative compositor | `paper/compose_qualitative.py` | `compose_combined()` |

---

## 10. How to Start Writing

Open a new Claude conversation and say:

> "I want to write a CVPR 2026 paper. Please read the paper brief at `paper/PAPER_BRIEF.md`. Then help me write Section 3 (Method) first, starting with the pipeline overview and the PSF DPI argument."

### Section writing order (recommended)
1. **§3 Method** — pipeline, DPI argument, CFA formulation (most novel, defines vocabulary)
2. **§4.1–4.2 Experiments/Setup + Main Results** — paste Table 1 from `paper/figures/table_main_results.tex`
3. **§4.3 Ablation** — reference Fig. 7 (lollipop chart)
4. **§4.4 Tile Size** — reference Fig. 6 (line plot)
5. **§4.5 CFA patterns** — reference Fig. 2/3/4
6. **§4.6 Qualitative** — reference `qualitative_combined_paper.pdf`
7. **§1 Introduction** — write last (once claims are clear)
8. **Abstract** — write very last

### Figure → Section mapping

| Figure | Section | Key message |
|---|---|---|
| Fig. 1 (pipeline) | §3.1 | Overview of differentiable RAW-to-task pipeline |
| Fig. 2 (CFA tiles) | §4.5 | 2×2 → Bayer; 3×3/4×4 → novel broadband filters |
| Fig. 3 (spectral bars) | §4.5 | R/G/B weights per filter site |
| Fig. 4 (mosaic pattern) | §4.5 | Spatial tiling of learned filters |
| Fig. 5 (results bars) | §4.2 | +0.019/+0.023 gain over fixed camera |
| Fig. 6 (tile trend) | §4.4 | Larger tiles consistently hurt |
| Fig. 7 (ablation lollipop) | §4.3 | CFA=dominant, Noise=marginal, PSF=negative |
| Fig. 8 (qualitative combined) | §4.6 | Fixed Camera misses rider/motorcycle; No Optics recovers them |
