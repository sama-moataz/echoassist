# EchoAssist: Automated Echocardiogram Video Analysis Pipeline for Left Ventricular Ejection Fraction Estimation, 3D Spatiotemporal Saliency Visualization, and Point-of-Care Clinical Interface

- **Track:** Health (AI + Health)
- **Competition:** Veritas AI International High School Hackathon
- **Author/Team:** Sama Moataz Fayez Ahmed Ali (and team)
- **Kaggle Notebook & Artifacts:** [https://www.kaggle.com/code/samamoatazt/echoassist](https://www.kaggle.com/code/samamoatazt/echoassist)

---

## The problem

Heart failure affects more than 64 million individuals worldwide, with Left Ventricular Ejection Fraction (LVEF) serving as the single most critical diagnostic biomarker for guiding therapy, pharmacological titration, and surgical interventions (such as implantable cardioverter-defibrillator placement). In clinical cardiology, sonographers and cardiologists evaluate hundreds of apical 4-chamber (A4C) echocardiographic video clips daily. Calculating LVEF traditionally relies on manual tracing of the left ventricular endocardial border across two cardinal phases of the cardiac cycle—End-Diastole (ED) and End-Systole (ES)—using Simpson's biplane method of disks:

$$
\text{LVEF} = \frac{\text{EDV} - \text{ESV}}{\text{EDV}} \times 100\%
$$

This manual workflow introduces three fundamental bottlenecks:
1. **High Inter- and Intra-Observer Variability:** Clinical literature documents inter-observer variability between 7% and 12%, frequently leading to diagnostic misstratification near critical clinical cutoff boundaries (e.g., 40% for HFrEF vs. HFmrEF, 50% for HFpEF).
2. **Temporal Decoupling:** Simpson's biplane method examines only two isolated static frames (ED and ES), discarding vital kinetic wall-motion velocities, myocardial strain rates, and valvular timing dynamics occurring throughout the continuous cardiac cycle.
3. **Severe Clinical Burden:** Manual contouring requires substantial time per patient, causing diagnostic delays in emergency departments and intensive care units where rapid triage is life-saving.

---

## What we built

**EchoAssist** is an end-to-end spatiotemporal deep learning pipeline that automates continuous LVEF regression, provides clinically transparent visual verification, and deploys an interactive point-of-care web application directly from raw Apical 4-Chamber (A4C) echocardiogram videos.

The system incorporates three core pillars:
1. **Spatiotemporal 3D Convolutional Neural Network (R(2+1)D-18):** Ingests raw video volumes $(C \times T \times H \times W)$ sampled at $T = 32$ frames ($112 \times 112$ resolution) to capture continuous systolic contraction and diastolic relaxation kinetics simultaneously.
2. **3D Grad-CAM Explainable AI Engine (`EchoGradCAM3D`):** Targets the final spatiotemporal residual block (`model.backbone.layer4[-1]`). By extracting 3D feature activations and gradients, calculating spatiotemporal channel weights via global average pooling, and upsampling via trilinear interpolation to original video dimensions $(32 \times 112 \times 112)$, EchoAssist produces publication-ready saliency heatmaps overlaid on echo frames. This allows clinicians to confirm that predictions are anchored to anatomical structures (the left ventricular apex, interventricular septum, and mitral annulus) rather than background text or noise.
3. **Interactive Clinical Web Application (Gradio):** A bedside-ready web interface where clinicians can drag-and-drop any A4C echocardiogram video (`.avi` / `.mp4`), click "Analyze Echocardiogram", and receive:
   - Automated continuous LVEF prediction (e.g., "54.2%").
   - Formal ACC/AHA clinical phenotype classification (HFrEF, HFmrEF, or HFpEF).
   - Side-by-side systolic 3D Grad-CAM saliency overlay visualizing regional myocardial activation.

---

## What we added

Most machine learning hackathon solutions in healthcare evaluate static tabular electronic health records (EHR) or 2D image classification baselines. We substantially advanced beyond standard baselines by implementing:
- **True Spatiotemporal Video Architecture:** Employing factored $(2+1)\text{D}$ convolutions that decompose 3D spatiotemporal operations into separate 2D spatial convolutions and 1D temporal convolutions, preserving spatiotemporal parameter efficiency while analyzing 32 frames across the entire cardiac cycle.
- **Explainability by Design:** Rather than treating deep regression as an opaque black box, our customized 3D Grad-CAM module isolates phase-specific myocardial contractions (end-diastole vs. end-systole), establishing transparent clinical verification.
- **Interactive Point-of-Care Deployment:** We integrated a functional Gradio interface directly in the Kaggle environment, translating a deep learning research model into a testable clinical decision-support tool.
- **Stress-Testing on External & Out-of-Distribution Data ("Make It Your Own"):** Rather than presenting EchoAssist as an idealized system, we systematically investigated how the pipeline behaves when exposed to non-standard ultrasound sources, evaluating both static image inputs and cross-vendor acoustic domain shifts.

---

## How it works

### 1. Data Pipeline and Cohort Hygiene
EchoAssist was developed using the **EchoNet-Dynamic** benchmark dataset from Stanford Health Care:
- **Cohort Composition:** 10,030 apical 4-chamber ultrasound video recordings in native `.avi` format.
- **Partitioning:** Strictly partitioned with zero patient overlap:
  - **Training Set:** 7,465 videos
  - **Validation Set:** 1,288 videos
  - **Held-Out Test Set:** 1,277 videos
- **Spatiotemporal Preprocessing:** Each video is uniformly sampled to $T = 32$ frames, resized to $112 \times 112$, converted to RGB tensors, and normalized using ImageNet population statistics $(\mu, \sigma)$.

### 2. Architecture and Optimization Engine
- **Backbone:** ResNet (2+1)D-18 pretrained on spatiotemporal action recognition.
- **Custom Regression Head:** Combines Adaptive Average Pooling 3D, Layer Normalization, Dropout ($p = 0.3$), and a linear projection head.
- **Loss Formulation:** Smooth L1 Loss ($\beta = 1.0$), transitioning from quadratic error at small differentials to linear error at large deviations, preventing instability from extreme clinical outlier cases.
- **Modern PyTorch AMP:** Fully optimized with modern native Automatic Mixed Precision (`torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')`) for high-throughput GPU training.
- **Deterministic Reproducibility:** Deterministic worker seeding (`seed_worker`, `torch.manual_seed(42)`) and OpenCV threading locks (`cv2.setNumThreads(0)`) eliminate multiprocessing conflicts.
- **Artifacts:** Complete model checkpoints (376.68 MB), training curves, regression plots, and confusion matrices are persistently version-controlled and hosted on Kaggle.

---

## Results

EchoAssist was rigorously evaluated on the official held-out test split of **1,277 patient video records** never seen during training or validation.

### Quantitative Regression Performance
| Metric | Test Set Value (1,277 samples) | Clinical Significance |
| :--- | :---: | :--- |
| **Mean Absolute Error (MAE)** | **4.47%** | Substantially outperforms human inter-observer variability ($\approx 7-12\%$). |
| **Root Mean Squared Error (RMSE)** | **6.29%** | Demonstrates low residual variance and absence of extreme outliers. |
| **Coefficient of Determination ($R^2$)** | **0.744** | Explains 74.4% of total variance in cardiac ejection fraction. |
| **Pearson Correlation ($r$)** | **0.863** | Demonstrates strong, statistically significant linear agreement ($p < 0.001$). |
| **Validation Benchmark** | **4.35% MAE ($R^2 = 0.773$)** | Verified optimal convergence across validation epochs. |

### Bland-Altman Agreement Analysis
- **Mean Bias:** $+0.09\%$ (virtually zero systematic over- or under-estimation across the entire dynamic range of $10\%$ to $85\%$ EF).
- **95% Limits of Agreement:** $-12.04\%$ to $+12.21\%$ ($\text{Mean} \pm 1.96 \text{ SD}$), confirming consistent accuracy across normokinetic, hypokinetic, and hyperdynamic ventricles.

### Clinical Heart Failure Stratification
Continuous predictions were mapped into standard ACC/AHA diagnostic categories:
- **HFrEF (Heart Failure with Reduced EF):** $\text{LVEF} < 40\%$
- **HFmrEF (Heart Failure with Mildly Reduced EF):** $40\% \le \text{LVEF} < 50\%$
- **HFpEF (Heart Failure with Preserved EF):** $\text{LVEF} \ge 50\%$

The model demonstrated high discriminative fidelity, reliably separating preserved systolic function ($\ge 50\%$) from pathological impairment ($<50\%$).

---

### Where It Fails: Failure Modes & Domain Shift Analysis

Evaluating EchoAssist on external, out-of-distribution inputs reveals critical challenges inherent to medical artificial intelligence:

#### 1. Ingestion of Static Images (Temporal Deprivation)
* **Technical Incompatibility:** The R(2+1)D-18 backbone strictly requires a 4D spatiotemporal tensor $(C \times T \times H \times W)$ where $T = 32$. Ingesting a 2D static frame (e.g., a single JPEG screenshot) violates tensor dimensionality.
* **Physiological Constraint:** Ejection Fraction is by definition a dynamic, time-dependent volumetric differential ($\Delta V = \text{EDV} - \text{ESV}$). A single static frame freezes myocardial movement, depriving the model of temporal wall velocities and displacement vectors. If repeated statically across 32 time steps, the network detects zero temporal variance and outputs an arbitrary, ungrounded prediction.

#### 2. Cross-Hospital and Cross-Vendor Domain Shift
* **Acoustic & Hardware Divergence:** Different ultrasound manufacturers (such as GE Healthcare, Philips, and Siemens) employ proprietary beamforming algorithms, gain post-processing, contrast compression curves, transducer frequencies, and scanning depth limits.
* **Impact on Representation:** While cardiac anatomy (the A4C view) is conserved, differences in speckle pattern texture and intensity distributions between Stanford’s EchoNet-Dynamic cohort and outside machines cause distribution shift, resulting in reduced predictive confidence or localized bias.

#### 3. Anatomical & Acoustic Artifacts
* **Severe Dilated Cardiomyopathy (Extreme Minority Cases):** In patients with end-stage HFrEF ($\text{LVEF} < 25\%$), massive left ventricular dilation and near-static akinetic walls cause minor regression toward the cohort mean ($30-35\%$).
* **Acoustic Shadowing & Narrow Windows:** Patients with severe obesity, chronic obstructive pulmonary disease (COPD), or surgical dressings exhibit acoustic shadowing across the cardiac apex, requiring greater reliance on basal/annular kinetics.
* **Burned-In Telemetry:** Persistent Doppler overlays, ECG leads, or vendor calibration grids occasionally trigger spurious peripheral Grad-CAM activations.

> **Mandatory Disclaimer:** *This project is a prototype and is not medical advice.*

---

## What we'd do next

1. **Acoustic Domain Adaptation & Preprocessing:** Implement adaptive histogram equalization, contrast-invariant augmentations, and cycle-consistent generative adversarial networks (CycleGAN) to harmonize ultrasound scans across diverse machine manufacturers (GE, Philips, Siemens).
2. **Biplane Multi-View Fusion (A4C + A2C):** Simpson's biplane gold standard clinically requires both Apical 4-Chamber (A4C) and Apical 2-Chamber (A2C) projections. Future work will integrate an asymmetric cross-attention transformer to jointly process both orthogonal views for complete volumetric agreement.
3. **Multi-Task Geometric Regularization:** Co-training the 3D backbone with auxiliary segmentation heads for the left ventricular endocardium, myocardium, and left atrium to enforce anatomically bounded geometric priors.
4. **Adaptive Temporal Resampling:** Integrating frame-rate-agnostic temporal interpolation to gracefully handle echocardiograms recorded at varying scan rates ($30\text{ fps}$ to $80+\text{ fps}$) or varying heart rates (tachycardia vs. bradycardia).

---

## Sources

1. **EchoNet-Dynamic Dataset:** Ouyang, D., He, B., Ghorbani, A., et al. *Video-based AI for beat-to-beat assessment of cardiac function.* Nature 580, 252–256 (2020). Stanford Health Care via kagglehub (`mahnurrahman/echonet-dynamic`).
2. **R(2+1)D Architecture:** Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M. *A Closer Look at Spatiotemporal Convolutions for Action Recognition.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 6450–6459 (2018).
3. **Grad-CAM Methodology:** Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* IEEE International Conference on Computer Vision (ICCV), 618–626 (2017).
4. **Computational Libraries:** PyTorch (`torch`, `torchvision`), OpenCV (`cv2`), scikit-learn, Seaborn, Matplotlib, Gradio.
5. **Kaggle Repository & Execution Artifacts:** [EchoAssist on Kaggle](https://www.kaggle.com/code/samamoatazt/echoassist)
