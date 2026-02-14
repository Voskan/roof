# Product Requirements Document (PRD)

**Project Name:** DeepRoof-2026 (High-Fidelity Roof Reconstruction)
**Version:** 1.0
**Status:** Approved
**Date:** February 14, 2026
**Target Release:** Q3 2026

---

## 1. Executive Summary
**DeepRoof-2026** is an enterprise-grade AI system designed to automatically reconstruct Level of Detail 2 (LOD-2) 3D roof models from 2D high-resolution satellite imagery (Google Maps Max Zoom).

Unlike traditional segmentation models that output "blobs" of roof pixels, DeepRoof-2026 uses a **Multi-Task Mask2Former architecture** to segment individual roof facets (planes), calculate their precise 3D geometry (slope, azimuth), and output regularized, CAD-ready vector polygons. The system is specifically engineered to handle the "single large building" use case with high precision, overcoming artifacts like shadows and obstructions.

---

## 2. Problem Statement
Current automated solutions for roof analysis have critical limitations:
1.  **Lack of Structure:** They treat a complex roof as a single mask, failing to distinguish between different slopes and orientations.
2.  **No 3D Geometry:** They cannot estimate roof pitch (slope) from 2D images, which is critical for solar potential analysis.
3.  **Raster Artifacts:** Output masks have "stair-step" pixel edges, requiring heavy manual cleanup for CAD use.
4.  **Zoom Issues:** Standard models fail on high-zoom imagery where a single building occupies the entire view, losing context.

---

## 3. Target Audience
* **Solar Energy Providers:** Automated layout planning for PV panels (requires precise tilt/azimuth).
* **Insurance Companies:** Risk assessment based on roof material and steepness.
* **Urban Planners & GIS Analysts:** Large-scale 3D city modeling.
* **Roofing Contractors:** Automated estimation of surface area and material costs.

---

## 4. Product Scope

### 4.1 In Scope
* Ingestion of high-resolution RGB satellite imagery (GeoTIFF, Google Maps Tiles).
* Instance segmentation of individual roof facets (not just whole buildings).
* Prediction of surface normal vectors (3D geometry) per facet.
* Calculation of Roof Pitch (degrees) and Azimuth (orientation).
* Post-processing: Regularization (straightening lines) and Orthogonalization (90-degree corners).
* Output in GeoJSON and DXF formats.

### 4.2 Out of Scope (for v1.0)
* LiDAR point cloud ingestion (system must work on RGB *only*).
* Street-view image integration.
* Real-time processing on mobile devices (Server-side GPU required).

---

## 5. Functional Requirements (FR)

### FR-01: Data Ingestion & Preprocessing
* **FR-01.1:** System must accept input images with a ground sampling distance (GSD) of 15cm–30cm/pixel.
* **FR-01.2:** System must handle images larger than GPU memory via a **Sliding Window** algorithm with configurable overlap (default 50%).
* **FR-01.3:** System must apply **Test Time Augmentation (TTA)** (rotations 0, 90, 180, 270) to ensure rotation invariance.

### FR-02: Core AI Inference (The "Brain")
* **FR-02.1:** The model architecture must be **Mask2Former** with a **Swin-Transformer V2-Large** (or Giant) backbone.
* **FR-02.2:** The model must utilize a **Multi-Head Decoder**:
    * *Head A:* Class Prediction (Flat, Facet, Obstacle).
    * *Head B:* Binary Mask Generation (Instance Segmentation).
    * *Head C:* Geometry Regression (predicting pixel-wise Normal Vector $\vec{n}$).
* **FR-02.3:** The model must achieve a Mean Intersection over Union (mIoU) of > **0.85** on the validation set.

### FR-03: Geometry Extraction
* **FR-03.1:** System must calculate the **Slope (Pitch)** for each facet using the formula $\theta = \arccos(n_z)$.
* **FR-03.2:** System must calculate the **Azimuth (Aspect)** using $\phi = \arctan2(n_y, n_x)$.
* **FR-03.3:** Slope accuracy must be within **±5 degrees** for clearly visible roofs.

### FR-04: Vectorization & Regularization
* **FR-04.1:** System must convert raster masks to vector polygons.
* **FR-04.2:** System must apply **Ramer-Douglas-Peucker (RDP)** simplification to remove pixel noise.
* **FR-04.3:** System must apply **Building Regularization**:
    * Snap lines to dominant angles (parallel/perpendicular).
    * Force near-90-degree corners to be exactly 90 degrees.
* **FR-04.4:** System must merge adjacent facets that share a boundary and have similar normal vectors (< 5° difference).

### FR-05: Output Formats
* **FR-05.1:** Generate **GeoJSON** files containing `Polygon` features with properties: `instance_id`, `slope_degrees`, `azimuth_degrees`, `area_sqm`, `confidence_score`.

---

## 6. Non-Functional Requirements (NFR)

### NFR-01: Performance
* **Inference Speed:** Must process a 1km x 1km area at max zoom in under 60 seconds on a single NVIDIA A100 GPU.
* **Latency:** Single building request API latency < 3 seconds.

### NFR-02: Quality & Reliability
* **Edge Quality:** Use **PointRend** module to ensure boundaries are sharp, not blurry.
* **False Positives:** Rate of detecting non-roof objects (patios, driveways) as roofs must be < 2%.

### NFR-03: Scalability
* System must be containerized (Docker) and orchestratable via Kubernetes.
* Support for distributed training across multiple GPUs (DDP).

---

## 7. Data Strategy

### 7.1 Datasets
* **Primary Training Data:** **OmniCity Dataset** (Fine-grained 3D). Contains roof plane instances and height maps.
* **Secondary Training Data:** **Building3D**. For learning complex geometries.
* **Validation Data:** A manually annotated subset of 500 diverse buildings from Google Maps (to benchmark against the specific target domain).

### 7.2 Augmentation Strategy (Domain Adaptation)
To match Google Maps quality, the training pipeline must implement:
* **JPEG Artifact Injection:** Quality 50-75.
* **Blur:** Gaussian blur $\sigma = 0.5 - 2.0$.
* **Shadow Injection:** Synthetic shadows overlaid on training images to teach the model to ignore/utilize them.

---

## 8. Technical Architecture

### 8.1 High-Level Diagram
1.  **Input:** Satellite Image Tile ($1024 \times 1024$).
2.  **Backbone:** Swin-L (Feature Extraction).
3.  **Pixel Decoder:** Multi-scale features (FPN).
4.  **Transformer Decoder:** Query-based attention.
5.  **Heads:**
    * $\rightarrow$ Mask Head $\rightarrow$ **PointRend** $\rightarrow$ High-Res Mask.
    * $\rightarrow$ Class Head $\rightarrow$ Label.
    * $\rightarrow$ Normal Head $\rightarrow$ Normal Map ($N_x, N_y, N_z$).
6.  **Post-Processing:** TTA $\rightarrow$ Merge $\rightarrow$ Vectorize $\rightarrow$ Regularize.
7.  **Output:** JSON/DXF.

### 8.2 Loss Functions
The model training will minimize the following weighted loss:
$$L_{total} = \lambda_{mask}L_{Dice} + \lambda_{cls}L_{CrossEntropy} + \lambda_{geo}(1 - \cos(\theta_{err}))$$

---

## 9. Roadmap & Milestones

### Phase 1: Foundation (Month 1)
* Setup AWS/GCP GPU environment.
* Ingest and clean OmniCity dataset.
* Implement standard Mask2Former training loop.
* **Milestone:** Model detects roofs (binary) with IoU > 0.80.

### Phase 2: Geometry & "Deep" Understanding (Month 2)
* Implement Custom Geometry Head.
* Generate Normal Maps from OmniCity height data.
* Train with Multi-Task Loss.
* **Milestone:** Model colors roof planes based on slope angle correctly.

### Phase 3: High-Fidelity Refinement (Month 3)
* Integrate Swin-Large Backbone.
* Implement PointRend for sharp edges.
* Implement Sliding Window Inference & TTA.
* **Milestone:** Clean segmentation of large warehouses and complex residential roofs.

### Phase 4: Vectorization & Production (Month 4)
* Develop Regularization algorithms (orthogonalization).
* Build API wrapper (FastAPI).
* Export logic for GeoJSON/DXF.
* **Milestone:** Final "DeepRoof-2026" Release Candidate.

---

## 10. Success Metrics (KPIs)

| Metric | Target | Measurement Method |
| :--- | :--- | :--- |
| **Mask mIoU** | > 85% | Intersection over Union on Test Set |
| **Instance Separation** | > 90% | AP (Average Precision) @ IoU 0.50 |
| **Slope Error (MAE)** | < 5° | Mean Absolute Error vs Ground Truth LiDAR |
| **Azimuth Error (MAE)**| < 10° | Mean Absolute Error vs Ground Truth |
| **Corner Straightness** | > 95% | % of corners within 2° of 90 degrees |

---

## 11. Risks & Mitigation

* **Risk:** Model overfits to clean OmniCity data and fails on noisy Google Maps.
    * *Mitigation:* Heavy use of "Google Maps Style" augmentations (compression, noise, shadows).
* **Risk:** Swin-Large is too slow for production.
    * *Mitigation:* Use TensorRT optimization and FP16 (half-precision) inference.
* **Risk:** Flat roofs detected as sloped due to texture noise.
    * *Mitigation:* Post-processing rule: if calculated slope < 5°, force to 0°.