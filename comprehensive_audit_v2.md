# DeepRoof-2026: Comprehensive "Ideal Result" Analysis

Following the resolution of the critical class imbalance bug, a deep, comprehensive re-analysis of the entire project was conducted to identify any remaining bottlenecks preventing the system from achieving an "ideal" (state-of-the-art) result.

## 1. Mathematical Implementations
**Status: Highly Robust, minor edge-case improvement needed.**

*   **Geometry Head and Normal Vectors:** The `GeometryHead` correctly uses `F.normalize(p=2, dim=-1)` to enforce $\|\mathbf{n}\| = 1$, strictly adhering to vector physics.
*   **Cosine Similarity Loss:** The derivation $1 - \cos(\theta)$ correctly mapped to $1 - (\mathbf{n}_{pred} \cdot \mathbf{n}_{gt})$ is the optimal loss function for 3D orientation regression, as it avoids singularities common with L1/L2 loss on angles.
*   **Data Augmentations:** The `GeometricAugmentation` pipeline correctly implements physical rotation matrices upon stochastic spatial transforms (e.g., flipping $n_x$ upon Horizontal Flip, applying $2D$ rotation matrices during RandomRotate).
*   **💡 Area for Improvement:**
    *   **Slope Calculation:** In `deeproof/utils/geometry.py`, the slope is built on `arccos(n_z)`. If noise causes a slight negative $n_z$, `arccos` will output $>90^\circ$, causing data discontinuity. **Change to `arccos(abs(n_z))`** to guarantee physical boundaries.

## 2. Model Architecture
**Status: SOTA, but missing specialized layer configs.**

*   **Mask2Former + Swin-Large:** This is currently one of the strongest publicly known architectures for dense prediction. The decoder cleanly separates segmentation logic from structural logic.
*   **Geometry Head Capacity:** A 3-layer MLP with 256 hidden dimensions is perfectly adequate to regress 3 values ($n_x, n_y, n_z$) from a rich 256-D query embedding. Increasing depth further would likely cause overfitting rather than better regression.
*   **DropPath:** `swin_large.py` sets `drop_path_rate=0.3`. This is mathematically in-line with recommendations for Swin-L to prevent co-adaptation without crippling the feature extraction.

## 3. ML Training Process (The Largest Bottleneck)
**Status: Contains a severe optimization flaw preventing "Ideal" results.**

*   **Learning Rate Configuration:** The training config (`deeproof_production_swin_L.py`) uses a uniform `AdamW` optimizer for the *entire network*.
*   **💡 CRITICAL Improvement (LLRD):** You are fine-tuning a massive pre-trained `Swin-Large` backbone (initialized with SatMAE/ImageNet weights) simultaneously with randomly initialized Mask2Former heads.
    *   Applying the same learning rate ($0.0001$ or $0.00005$) to both destroys the pre-trained semantic representations in the backbone early in training.
    *   **Fix:** We must implement **Layer-wise Learning Rate Decay (LLRD)** or at minimum, a `backbone_multiplier`. In the `optim_wrapper`, you must set `paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)})`. This allows the heads to learn at `5e-5` while the backbone gently shifts at `5e-6`.
*   **Learning Rate Scheduler:** Consider switching from `PolyLR` to `CosineAnnealing` with warm restarts for the final fine-tuning sweep on the A100 cluster.

## 4. Quality & Post-Processing
**Status: Functional, but constrained by heuristic limits.**

*   **Inference Pipeline:** The `sliding_window_inference` logic elegantly batches crops and maintains absolute coordinates. 
*   **Non-Maximum Suppression (NMS):** Currently, the pipeline utilizes Bounding Box NMS (`torchvision.ops.nms`). 
    *   **💡 Area for Improvement:** For long, diagonal, or L-shaped buildings, the bounding boxes heavily overlap even if the masks do not. Box NMS will suppress valid adjacent roof segments. Switching to **Mask-IoU NMS** is strictly required for dense urban areas to achieve ideal recall.
*   **Vectorization (`enforce_orthogonality`):** The post-processing tries to aggressively snap angles to 90 degrees. This yields "pretty" CAD files for rectangular suburban houses, but significantly degrades geometry on modern architecture, curved layouts, or hexagonal structures. 
    *   **Fix:** Introduce a secondary angle-tolerance curve that only snaps angles within 10-15 degrees of 90, heavily penalizing geometric alteration outside that bound.

## Summary Action Plan for "Ideal" Results:
1.  **Immediate:** Add `paramwise_cfg` to reduce the backbone learning rate by 10x (`lr_mult=0.1`). This is the single biggest ML change to get optimal accuracy without structural changes.
2.  **Short-term:** Swap bounding-box NMS for Mask-IoU NMS in the inference pipeline to prevent valid roofs being deleted in dense clusters.
3.  **Short-term:** Update the `get_slope` mathematical logic to use `abs(n_z)`.
4.  **Long-term:** Train using a Cosine Annealing learning rate schedule instead of polynomial decay for the final 20% of training steps.
