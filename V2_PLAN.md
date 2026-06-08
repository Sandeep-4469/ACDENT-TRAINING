# Training v2 — Improvement Plan

**Current results (ResNet50, 5-fold CV):**
| Fold | sum3_mm |
|------|---------|
| fold_1 | 9.411 |
| fold_2 | 12.926 |
| fold_3 | **17.067** |
| fold_4 | 10.681 |
| fold_5 | 10.318 |
| **Mean** | **12.081 ± 2.75** |

The target is to bring the mean below ~9mm and eliminate the fold_3 catastrophic failure.

---

## Problem Diagnosis

### 1. Fold_3 catastrophic failure (17mm)
Fold_3 has 42 test images (largest test set) and the worst result by a wide margin. This is likely a distribution mismatch: some patient subgroup ends up concentrated in fold_3's test split. The fix is better architecture generalisation, not data re-splitting.

### 2. High scale error
Scale_mm oscillates from 2.32mm to 6.46mm across folds. Scale is just one line (2 keypoints). This level of error on a single line is the biggest single contributor to sum3_mm. Suggests the model is not stably localising the scale reference landmark.

### 3. Val loss does not discriminate
All configs converge to val_loss ~0.00103–0.00108. This means ReduceLROnPlateau is driving the schedule based on a signal that doesn't reflect actual test accuracy. The model may be converging to a local minimum early and then plateau.

### 4. No skip connections (spatial information loss)
ResNet50 stride-32 backbone → 16×16 feature map. Three deconvs bring it to 128×128, but intermediate spatial features from the encoder are discarded. This is the single largest architectural limitation.

---

## v2 Improvements — Prioritised

### P1 — U-Net style skip connections (biggest expected gain)

**What:** Connect backbone intermediate feature maps to the deconv head (same as U-Net encoder-decoder). ResNet50 has outputs at stride 8 (layer2, 512ch), stride 16 (layer3, 1024ch), and stride 32 (layer4, 2048ch).

**How:** 
- Tap `layer2` output → 1×1 conv → 128ch → concat or add into deconv stage 1
- Tap `layer3` output → 1×1 conv → 256ch → concat or add into deconv stage 2
- This gives the head access to fine-grained spatial detail that gets lost in the deep backbone

**Why high priority:** Landmark detection is fundamentally a spatial localisation task. Discarding early feature maps is the core architectural bottleneck.

---

### P2 — Cosine Annealing with linear warmup (replaces ReduceLROnPlateau)

**What:** Replace ReduceLROnPlateau with `CosineAnnealingLR` (T_max = total_epochs). Add 10-epoch linear warmup at the start.

**Why:** ReduceLROnPlateau reduces LR only when val_loss plateaus — but we just showed val_loss barely changes between configs. So the scheduler is effectively blind. Cosine annealing decays the LR regardless of metric, providing smooth exploration then fine-tuning. Warmup prevents early instability from large initial gradients.

```
epoch 0→10:   LR linearly 1e-5 → 1e-4  (warmup)
epoch 10→300: CosineAnnealing 1e-4 → 1e-6
```

---

### P3 — Differential learning rate (backbone vs. head)

**What:** Backbone (pretrained on ImageNet) gets LR × 0.1. Deconv head (random init) gets full LR.

**Why:** The backbone is already trained on good features. Updating it too fast destroys pretrained representations. This is standard practice in fine-tuning.

```python
optimizer = AdamW([
    {"params": model.backbone.parameters(), "lr": LR * 0.1},
    {"params": model.deconv.parameters(),   "lr": LR},
    {"params": model.head.parameters(),     "lr": LR},
], weight_decay=WEIGHT_DECAY)
```

---

### P4 — Use full 6-loss combination as default

**What:** The ablation study confirmed `06_full` (all 6 losses) achieves the best sum3_mm (9.321mm). The current `train_resnet50.py` uses these losses but may not have the optimal weights.

**Tune loss weights:** Based on ablation, coord loss drives the biggest gain but can also destabilise scale. Reduce coord weight slightly and increase arc contrast weight.

Current: `COORD=0.02, LENGTH=0.01, ARC_SIDE=0.008, ARC_ALIGN=0.005, ARC_CONTRAST=0.004`  
Proposed: `COORD=0.015, LENGTH=0.012, ARC_SIDE=0.010, ARC_ALIGN=0.004, ARC_CONTRAST=0.008`

Rationale: length loss and arc_contrast gave the cleanest improvements in ablation; coord is slightly reduced to prevent scale regression.

---

### P5 — Gradient clipping

**What:** Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`.

**Why:** Without clipping, large gradient spikes (especially early in training with the new losses) can corrupt the backbone's pretrained weights. Low risk, high safety.

---

### P6 — Larger sigma and normalised heatmaps

**What:** Increase heatmap sigma from 3 to 4 pixels (on the 128×128 map, sigma=3 corresponds to ~12px in the 512px image ≈ 1.2mm, which is very tight). Also normalise each heatmap channel to peak at 1.0.

**Why:** Larger sigma gives a broader training signal — the model gets gradient from a wider region around each keypoint, not just exactly on the peak. This improves stability in early training especially for challenging keypoints.

---

### P7 — More augmentation variety + higher aug count

**What:** In `prepare_folds.py`, increase `aug_per_image` from 4 to 7 (1 original + 7 augmented = 8× train data).

**Add two new augmentation transforms to `augment.py`:**
1. **Random erasing** (cutout): randomly mask 10–20% of image with grey patch — forces the model to not rely on any single image region
2. **Random horizontal shear** (mild ±5°): dental arches have some asymmetry that shear can simulate

**Why:** 171 unique training images is very small. The current augmentation (affine + colour + blur) is already good. Adding cutout/shear adds diversity without breaking anatomical validity.

---

### P8 — Heatmap resolution upgrade (128 → 192)

**What:** Change `HEATMAP_SIZE` from 128 to 192. The backbone still outputs 16×16; add one more deconv stage to reach 192×192 (or 256×256).

**Why:** Each pixel in a 128×128 heatmap corresponds to 4×4px in the 512px input image = ~0.4mm per pixel at typical dental scale. Upgrading to 192 or 256 improves localisation precision directly. This is especially important for scale and incisor lines where precision matters most.

**Tradeoff:** Memory and compute increase. With RTX A6000 (48GB), 256×256 at batch=8 should be fine.

---

### P9 — Test-time augmentation (TTA)

**What:** At inference, run the model on 4 versions of each image (original + horizontal flip + slight rotation ±5°), average the heatmaps, then decode.

**Why:** Cheap way to get ensemble-like robustness without training extra models. No training changes needed — just add to `test.py` and the validation loop.

**Note:** Flip must account for keypoint mirroring (left↔right landmarks swap on horizontal flip).

---

### P10 — Dropout in deconv head

**What:** Add `nn.Dropout2d(p=0.1)` after each deconv+BN+ReLU block.

**Why:** The head is applied to 695 training images. Dropout prevents the head from memorising training set spatial patterns and forces more robust feature usage.

---

## Implementation Order

```
Phase 1 (highest impact, implement together):
  ✦ P1  Skip connections (U-Net head)
  ✦ P2  Cosine annealing + warmup
  ✦ P3  Differential LR

Phase 2 (data and regularisation):
  ✦ P5  Gradient clipping
  ✦ P6  Sigma=4 + normalised heatmaps
  ✦ P7  More augmentation

Phase 3 (optional, if phase 1+2 not enough):
  ✦ P8  Heatmap resolution 192/256
  ✦ P9  TTA at inference
  ✦ P10 Dropout
```

---

## What We Are NOT Changing

- **Dataset**: same 171 non-boston images, same 5-fold patient split. No new data.
- **Full 6-loss combination**: confirmed best by ablation.
- **ResNet50 backbone**: established, pretrained. Not replacing with a ViT or other exotic architecture yet.
- **Offline augmentation pipeline**: works well. Only increasing count + diversity.
- **512×512 input resolution**: already at max.

---

## Expected Outcome

| Improvement | Expected Δ sum3_mm |
|-------------|-------------------|
| Skip connections (P1) | −1.5 to −2.5mm |
| Better schedule + diff LR (P2+P3) | −0.5 to −1.5mm |
| Augmentation (P7) | −0.3 to −0.8mm |
| Gradient clip + sigma + dropout (P5+P6+P10) | −0.2 to −0.5mm |
| **Total expected** | **−2.5 to −5.3mm** |

Target: **mean sum3_mm < 9mm**, fold variance < 2mm.

---

*Current baseline: mean=12.081mm ± 2.749  (best fold: 9.411mm)*
