


This repository contains the implementation of **“FedNML: A Robust Federated Learning Framework for Noisy and Missing Labels in Medical Image Classification.”**
<img width="3318" height="2074" alt="fig2" src="https://github.com/user-attachments/assets/6f6e38c0-6262-4957-b116-9d3394f9c4f4" />
---

## 🧠 Paper Abstract

Federated learning has become a prominent research frontier in medical image classification, as it enables the integration of data from multiple institutions under strict privacy constraints. Nevertheless, practical medical environments exhibit heterogeneous label quality across clients, where label noise (such as mis-annotations and omissions) and missing labels (e.g., annotations limited to target classes) coexist, significantly undermining the stability and generalizability of the global model.

To tackle these challenges, we propose **FedNML (Federated Learning for Noisy and Missing Labels)** — a robust federated learning framework that jointly models and optimizes for both label noise and missing annotations.

FedNML adopts a **two-phase training paradigm**:
1. **Warm-up stage:** Uses FedAvg to extract reliable sample features for constructing a stable initial model.  
2. **Refinement stage:** Employs a 2D Gaussian Mixture Model (2D-GMM) to jointly model loss values and consistency scores for identifying noisy samples, which are corrected using high-confidence pseudo labels.

Additionally:
- A **hybrid pseudo-labeling strategy** (hard + soft) compensates for missing classes.
- A **class-aware loss formulation** with logit adjustment and knowledge distillation enhances performance on underrepresented categories.

Extensive experiments on **RFMiD** and **MuRed** demonstrate that FedNML consistently surpasses existing methods in mAP, F1, Recall, and Balanced Accuracy — validating its robustness and effectiveness under low-quality label conditions.

---

## 📊 Experimental Results

### Table II — Results on **MuRed**
| Method | 20% Noise, Missing 2 | | | | 40% Noise, Missing 4 | | | | 60% Noise, Missing 8 | | | |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| | mAP | F1 | Recall | Bacc | mAP | F1 | Recall | Bacc | mAP | F1 | Recall | Bacc |
| FedAvg | 0.5394 | 0.4781 | 0.4128 | 0.6981 | 0.3618 | 0.3179 | 0.2532 | 0.6197 | 0.2265 | 0.1561 | 0.1265 | 0.5444 |
| FedNoRo | 0.5638 | 0.5279 | 0.5357 | 0.7543 | 0.4438 | 0.3804 | 0.3291 | 0.6559 | 0.2530 | 0.2740 | 0.2902 | 0.6061 |
| FedMLP | 0.5762 | 0.5206 | 0.4637 | 0.7239 | 0.4366 | 0.3549 | 0.2851 | 0.6360 | 0.2791 | 0.2779 | 0.2576 | 0.6075 |
| Fedlsm | 0.5865 | 0.5250 | 0.4904 | 0.7366 | 0.4738 | 0.4017 | 0.3653 | 0.6749 | 0.2863 | 0.2217 | 0.2154 | 0.5797 |
| FedDC | 0.5935 | 0.5184 | 0.4988 | 0.7352 | 0.4225 | 0.3403 | 0.3175 | 0.6446 | 0.2701 | 0.2003 | 0.1970 | 0.5677 |
| **Ours (FedNML)** | **0.6265** | **0.5823** | **0.5847** | **0.7787** | **0.4738** | **0.4795** | **0.4762** | **0.7233** | **0.4763** | **0.4393** | **0.3986** | **0.6892** |

---

### Table V — Results on **RFMiD**
| Method | 20% Noise, Missing 4 | | | | 40% Noise, Missing 6 | | | | 60% Noise, Missing 8 | | | |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| | mAP | F1 | Recall | Bacc | mAP | F1 | Recall | Bacc | mAP | F1 | Recall | Bacc |
| FedAvg | 0.4288 | 0.3883 | 0.3342 | 0.6585 | 0.2941 | 0.2777 | 0.2337 | 0.6075 | 0.2029 | 0.1729 | 0.1538 | 0.5653 |
| FedNoRo | 0.4611 | 0.4364 | 0.4038 | 0.6880 | 0.3281 | 0.3194 | 0.3072 | 0.6390 | 0.2208 | 0.2192 | 0.2333 | 0.5926 |
| FedMLP | 0.4763 | 0.4393 | 0.3815 | 0.6817 | 0.3834 | 0.3195 | 0.2563 | 0.6208 | 0.2763 | 0.1956 | 0.1444 | 0.5679 |
| Fedlsm | 0.4909 | 0.4643 | **0.5290** | 0.6998 | 0.3739 | 0.3357 | 0.3045 | 0.6416 | **0.3060 | 0.2485 | 0.2343 | 0.6047 |
| FedDC | 0.4885 | 0.4676 | 0.4461 | 0.7138 | 0.3746 | 0.3237 | 0.2576 | 0.6222 | 0.2535 | 0.1881 | 0.1666 | 0.5597 |
| **Ours (FedNML)** | **0.5077** | **0.5048** | 0.4998 | **0.7206** | **0.4238** | **0.3827** | **0.3585** | **0.6625** | 0.2886 | **0.2658** | **0.2518** | **0.6109** |

---

### Table VIII — Results on **BRSET**
| Method | 20% Noise, Missing 2 | | | | 40% Noise, Missing 4 | | | | 60% Noise, Missing 6 | | | |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| | mAP | F1 | Recall | Bacc | mAP | F1 | Recall | Bacc | mAP | F1 | Recall | Bacc |
| FedAvg | 0.3937 | 0.3934 | 0.3554 | 0.6690 | 0.3627 | 0.2748 | 0.2123 | 0.6009 | 0.2491 | 0.1320 | 0.0815 | 0.5393 |
| FedNoRo | 0.4275 | 0.4351 | 0.4206 | 0.6986 | 0.3501 | 0.2793 | 0.2173 | 0.6023 | 0.2147 | 0.2271 | 0.2851 | 0.6035 |
| FedMLP | 0.4271 | 0.4098 | 0.3505 | 0.6680 | 0.3976 | 0.3021 | 0.2331 | 0.6123 | 0.2334 | 0.1816 | 0.1372 | 0.5640 |
| Fedlsm | 0.4880 | 0.4045 | 0.3364 | 0.6635 | 0.4063 | 0.3858 | 0.3125 | 0.6520 | 0.2569 | 0.2270 | 0.1853 | 0.5879 |
| FedDC | 0.4645 | 0.3786 | 0.3618 | 0.6720 | 0.3889 | 0.3340 | 0.2718 | 0.6293 | 0.2086 | 0.1629 | 0.1076 | 0.5499 |
| **Ours (FedNML)** | **0.5166** | **0.4404** | **0.4671** | **0.7177** | **0.4927** | **0.4120** | **0.4647** | **0.6999** | **0.4101** | **0.3667** | **0.4479** | **0.6383** |

---

## 🧬 Datasets

### 1. RFMiD
- **Retinal Fundus Multi-Disease dataset** for multi-label fundus classification.  
- 3,200 color fundus images (1,000×1,000 – 4,000×3,000 px).  
- 46 retinal disease labels (multi-label setting).  
- Each image can have multiple diseases.

### 2. MuRed
- 2,208 fundus images (520×520 – 3,400×2,800 px).  
- 20 retinal disease labels including diabetic retinopathy, cataract, glaucoma.  
- Designed for multi-label classification of coexisting conditions.

### 3. BRSET
- 16,266 fundus images from 8,524 Brazilian patients.  
- Contains demographic, anatomical, and multi-label disease annotations.  
- Image resolution: 874×951 – 2,304×2,984 px.  

> ⚠️ Very low-frequency labels (<20 samples) were filtered out.  
> All images were resized to **256×256** for training.

---

## 📈 Evaluation Metrics

To comprehensively assess performance, four key metrics are used:
- **mAP** — Mean Average Precision (overall detection capability)
- **F1 Score** — Balance between precision and recall
- **Recall** — Sensitivity to positive samples
- **Balanced Accuracy (BACC)** — Robustness under class imbalance

---

## 📂 Dataset Structure

Example (MuRed):
```bash
./data/MuReD/
├── images/         # All image files
├── train.csv       # Training annotations
└── test.csv        # Testing annotations
```

### ⚙️ Parameter Settings

| Parameter | Description | Default |
|:--|:--|:--:|
| `epochs` | Number of global training rounds | 100 |
| `num_users` | Number of clients | 4 (MuRed/RFMiD), 6 (BRSET) |
| `local_ep` | Local epochs per client | 3 |
| `local_bs` | Local batch size | 32 |
| `lr` | Learning rate | 1e-4 |
| `num_classes` | Number of output categories | depends on dataset |
| `noise_r` | Label noise ratio | customizable |
| `annotation_num` | Retained class count after missing label simulation | customizable |

## 📂 running

```bash
run file ./src/train_FedNML.py
```


