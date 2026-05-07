# Leakage-Aware Optional-Modality Fusion for Alzheimer's MRI–PET Classification

## Project Overview

This project investigates deep learning for Alzheimer's disease-related biomarker classification using multimodal neuroimaging data, specifically MRI and PET scans. The original State-of-the-Art (SOA) survey focused on **site-invariant Alzheimer's detection using disease–acquisition latent factorization**.

The project is based on the idea that MRI and PET provide complementary information:

- **MRI** provides structural brain information.
- **PET** provides functional or biomarker-related information.
- A multimodal MRI–PET model can potentially use more information than a single-modality model.

The final project direction focuses on building a multimodal MRI–PET classification pipeline that is careful about:

1. **Subject-level leakage**
   - The same subject should not appear across training, validation, and test sets.
   - The final notebook uses subject-wise splitting and reports a subject leakage check.

2. **Class imbalance and class collapse**
   - The previous baseline struggled with class collapse and acted like a majority-class predictor.
   - This caused weak sensitivity for the positive class.

3. **Acquisition-related nuisance information**
   - The SOA survey identified acquisition shift as a major issue in Alzheimer's imaging.
   - The final model explicitly separates disease-related and acquisition-related latent representations.

4. **Multimodal MRI–PET fusion**
   - The project uses MRI and PET together instead of relying on one modality only.
   - The final architecture keeps MRI/PET encoders and summaries, optional modality gating, joint summary learning, and factorized latent representations.

---

## Research Motivation

Alzheimer's disease prediction from neuroimaging is difficult because deep learning models can learn shortcuts from scanner, site, protocol, or batch differences instead of learning true disease-related patterns.

The SOA survey states that Alzheimer's imaging datasets are collected across different institutions, scanners, and imaging protocols. Because of this, deep networks may learn acquisition-specific shortcuts and then perform poorly when tested on new scanners or cohorts.

The SOA survey also highlights that evaluation should not rely only on accuracy. Metrics such as balanced accuracy, sensitivity, specificity, F1-score, and AUC are important because class imbalance can make accuracy misleading.

This project therefore focuses on:

- multimodal MRI–PET learning,
- subject-wise leakage-safe splitting,
- class-collapse reduction,
- disease–acquisition latent factorization,
- acquisition leakage analysis,
- and reporting multiple evaluation metrics.

---

## Original SOA Direction

The SOA survey proposed the following research direction:

> Can a multimodal MRI–PET representation for Alzheimer's disease be decomposed into disease-relevant and acquisition-relevant latent factors such that AD diagnosis remains accurate while residual information about site, scanner, field strength, tracer, and protocol is reduced under external evaluation?

The key idea is that the model should not only classify Alzheimer's-related labels, but should also separate:

- **Disease latent representation**
  - This should contain diagnostic or disease-related information.

- **Acquisition latent representation**
  - This should contain nuisance or acquisition-related information.

The final notebook implements this direction through a disease-acquisition factorized MRI–PET model.

---

## Dataset Used

The project uses the **ds006148** dataset.

The notebooks use the `Amyloid_status` label.

The label mapping used in the baseline notebook is:

- `positive → 1`
- `negative → 0`

The dataset exploration notebook reported the following labeled subjects:

| Amyloid Status | Count |
|---|---:|
| Negative | 140 |
| Young | 62 |
| Positive | 44 |
| Total | 246 |

For the binary classification setup, the project focuses on the positive and negative amyloid status labels.

---

## Dataset Splitting

The project uses subject-wise splitting to reduce leakage.

The dataset exploration notebook used `GroupShuffleSplit` with subject IDs as groups and reported:

| Split | Subjects |
|---|---:|
| Training Set | 196 |
| Testing Set | 50 |
| Total Subjects | 246 |

The notebook also reported:

```text
Subject Leakage Check: There are no overlapping IDs
```

The final deliverable notebook used a subject-wise split on the multimodal MRI–PET samples and reported:

| Split | Samples | Subjects |
|---|---:|---:|
| Train | 69 | 23 |
| Validation | 24 | 8 |
| Test | 63 | 21 |
| Total | 156 | 52 |

The final split class distribution was:

| Split | Negative | Positive | Total |
|---|---:|---:|---:|
| Train | 54 | 15 | 69 |
| Validation | 18 | 6 | 24 |
| Test | 51 | 12 | 63 |
| Total | 123 | 33 | 156 |

The acquisition proxy distribution by split was:

| Split | Batch 1 | Batch 2 | Batch 3 | Batch 4 | Total |
|---|---:|---:|---:|---:|---:|
| Train | 30 | 21 | 12 | 6 | 69 |
| Validation | 6 | 3 | 9 | 6 | 24 |
| Test | 15 | 18 | 15 | 15 | 63 |
| Total | 51 | 42 | 36 | 27 | 156 |

All 156 final samples had both MRI and PET available:

| MRI Present | PET Present | Samples |
|---:|---:|---:|
| 1 | 1 | 156 |

---

## Project Progression

The project progressed through the following stages:

1. **Dataset Selection and Annotation**
2. **Baseline Model**
3. **Final Disease-Acquisition Factorized MRI–PET Model**

---

## Dataset Selection and Annotation

### Goal

The goal of the dataset selection stage was to inspect the available ds006148 data, understand labels, check subject IDs, and verify whether MRI and PET files could be used for multimodal learning.

### What Was Done

The notebook:

- loaded `QNLD_cognitive_scores_logfile.csv`,
- inspected the `Amyloid_status` label,
- counted available labeled subjects,
- used subject IDs for group-based splitting,
- checked for subject leakage,
- visualized MRI and PET slices for a selected subject,
- and defined the research scope for the next stages.

### Important Note About ADNI

The dataset selection notebook states that ADNI access had been applied for, but access had not been granted at that deliverable stage. Therefore, ADNI was not included in that deliverable.

---

## Baseline Model

### Goal

The baseline model was implemented using the MultimodalAD repository as the cited baseline.

The baseline notebook states that the model was adapted from:

```text
https://github.com/Kateridge/MultimodalAD
```

The goal was to adapt the baseline model to the local ds006148 subset while keeping the original repository model implementation and trainer classes the same.

### Label Setup

The baseline used `Amyloid_status` labels from ds006148 and mapped them to a binary classification task:

```text
positive -> 1
negative -> 0
```

### Main Baseline Idea

The baseline used a multimodal MRI–PET setup where MRI and PET are processed and then fused for prediction.

```text
MRI Volume  -> MRI Encoder  \
                            -> Fusion Module -> Classifier -> Prediction
PET Volume  -> PET Encoder  /
```

### Baseline Adaptation

The baseline notebook states that:

- MultimodalAD was selected because the SOA had identified it as a modality-flexible reference for clinically incomplete multimodal AD diagnosis.
- The repository's model implementation and trainer classes were kept the same.
- The uploaded ds006148 subset was adapted into the path and CSV structure expected by the repository.
- The paper baseline accuracy was around 76%.
- The project achieved around 75% with the same architecture.

### Baseline Problem

The final deliverable notebook states that the earlier baseline run struggled with **class collapse**. It behaved like a majority-class predictor and had poor positive-class detection.

The final deliverable notebook also states that this issue was linked to:

- data preprocessing errors,
- severe class imbalance,
- and weak sensitivity for the positive class.

---

## Final Model: Disease-Acquisition Factorized MRI–PET Model

### Goal

The final model extends the already trained multimodal fusion model by adding the SOA novelty: **disease-acquisition latent factorization**.

The goal is to separate disease-related information from acquisition-related nuisance information.

### Final Architecture Overview

The final notebook describes the architecture as follows:

```text
MRI/PET encoders + summaries
        ↓
optional modality gate + joint summary
        ↓
factor base representation
        ↓                         ↓
z_disease                  z_acquisition
        ↓                         ↓
diagnosis classifier       acquisition classifier
        ↓                         ↑
final diagnosis logits      acquisition proxy supervision
        ↑
GRL adversary tries to predict acquisition from z_disease,
but gradient reversal teaches z_disease to hide acquisition.
```

### Main Components

The final model includes:

1. **MRI/PET encoders**
   - Separate encoder branches process MRI and PET inputs.

2. **Multi-scale summaries**
   - MRI and PET features are summarized before fusion.

3. **PET masking**
   - The model includes handling for modality presence.

4. **Optional modality gate**
   - PET contribution is controlled through an optional modality gate.

5. **Joint summary head**
   - MRI and PET summaries are combined into a joint multimodal representation.

6. **Fused projection**
   - The joint representation is projected into a shared feature space.

7. **Auxiliary heads**
   - Additional heads support stable learning.

8. **Disease latent**
   - `z_disease` is intended to capture diagnostic information.

9. **Acquisition latent**
   - `z_acquisition` is intended to capture acquisition or batch-related nuisance information.

10. **Gradient Reversal Layer**
    - The acquisition adversary tries to predict acquisition from `z_disease`.
    - Gradient reversal encourages `z_disease` to hide acquisition information.

11. **Orthogonality loss**
    - This is used to separate disease and acquisition latent spaces.

12. **Acquisition proxy supervision**
    - Batch IDs are used as acquisition proxy groups in the final notebook.

---

## Key Improvements in the Final Model

### 1. Disease-Acquisition Latent Factorization

The final model explicitly decomposes the learned representation into:

- `z_disease`
- `z_acquisition`

This directly follows the research gap from the SOA survey, where previous work aligned or harmonized features but did not clearly separate disease and acquisition factors.

### 2. Gradient Reversal Layer and Adversarial Training

The final model uses a Gradient Reversal Layer so that the disease latent representation becomes less informative about acquisition proxy labels.

This helps reduce acquisition leakage into the disease representation.

### 3. Orthogonality Loss

The final model adds orthogonality loss to encourage the disease and acquisition representations to be separated.

### 4. Zero-Leakage Subject-Wise Split

The final notebook uses subject-wise splitting and reports that the subject leakage check passed.

### 5. Corrected Preprocessing

The final notebook states that preprocessing and paired data construction were corrected compared with the earlier baseline.

### 6. Missing Modality Handling

The final notebook keeps missing-modality handling and optional modality gating from the previous fusion model.

### 7. Validation Threshold Selection

The best model was evaluated using the best validation threshold instead of using only a fixed default threshold.

The final notebook reported:

| Item | Value |
|---|---:|
| Best Epoch | 15 |
| Best Validation Score | 0.7435 |
| Best Threshold | 0.53 |

---

## Final Test Results

The final factorized model was evaluated on the held-out test set using threshold `0.53`.

### Final Metrics

| Metric | Value |
|---|---:|
| Prediction Positive Rate | 0.1111 |
| Accuracy | 0.8254 |
| Balanced Accuracy | 0.6373 |
| F1 Macro | 0.6591 |
| F1 Weighted | 0.8065 |
| Precision Positive | 0.5714 |
| Recall Positive | 0.3333 |
| F1 Positive | 0.4211 |
| AUC | 0.7369 |
| Probability Positive Mean | 0.4877 |
| Probability Positive Min | 0.4080 |
| Probability Positive Max | 0.5658 |
| Sensitivity Positive | 0.3333 |
| Specificity Negative | 0.9412 |

### Final Confusion Matrix

The confusion matrix uses labels:

- `0 = negative`
- `1 = positive`

```text
[[48,  3],
 [ 8,  4]]
```

This means:

- 48 negative samples were correctly predicted as negative.
- 3 negative samples were incorrectly predicted as positive.
- 8 positive samples were incorrectly predicted as negative.
- 4 positive samples were correctly predicted as positive.

### Final Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.86 | 0.94 | 0.90 | 51 |
| Positive | 0.57 | 0.33 | 0.42 | 12 |
| Accuracy |  |  | 0.83 | 63 |
| Macro Avg | 0.71 | 0.64 | 0.66 | 63 |
| Weighted Avg | 0.80 | 0.83 | 0.81 | 63 |

---

## Acquisition Proxy Group Results

The final notebook also reports test performance across acquisition proxy groups.

| Acquisition Group | n | Pred Pos Rate | Accuracy | Balanced Accuracy | F1 Macro | F1 Weighted | Precision Positive | Recall Positive | F1 Positive | AUC | Sensitivity Positive | Specificity Negative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batch_1 | 15 | 0.0667 | 0.9333 | 0.9333 | 0.4828 | 0.9655 | 0.0000 | 0.0000 | 0.0000 | NaN | 0.0000 | 0.9333 |
| batch_2 | 18 | 0.1111 | 0.7778 | 0.6667 | 0.6786 | 0.7381 | 1.0000 | 0.3333 | 0.5000 | 0.8611 | 0.3333 | 1.0000 |
| batch_3 | 15 | 0.2000 | 0.8667 | 0.7917 | 0.7917 | 0.8667 | 0.6667 | 0.6667 | 0.6667 | 0.9444 | 0.6667 | 0.9167 |
| batch_4 | 15 | 0.0667 | 0.7333 | 0.4583 | 0.4231 | 0.6769 | 0.0000 | 0.0000 | 0.0000 | 0.3889 | 0.0000 | 0.9167 |

---

## Latent Leakage Probe Results

The final notebook includes leakage probes to check what information is present in the learned latent spaces.

| Probe | Accuracy | Balanced Accuracy |
|---|---:|---:|
| Predict acquisition from `z_disease` | 0.1746 | 0.1750 |
| Predict acquisition from `z_acquisition` | 0.1746 | 0.1694 |
| Predict diagnosis from `z_disease` linear probe | 0.8095 | 0.5000 |

These probes were included because the SOA survey emphasized the need to check whether scanner or acquisition information is still present in the disease representation.

---

## Comparison Between Baseline and Final Model

The attached notebooks describe the baseline as a MultimodalAD adaptation and the final model as a disease-acquisition factorized MRI–PET model.

| Model / Stage | Key Idea | Main Reported Result |
|---|---|---|
| Baseline | Adapted MultimodalAD to ds006148 binary amyloid classification | Around 75% accuracy with the same architecture |
| Earlier baseline run | Previous run had class collapse and acted like a majority-class predictor | Around 62.5% accuracy with 0% sensitivity |
| Final factorized model | MRI/PET fusion with disease-acquisition latent factorization, GRL, orthogonality loss, and subject-wise split | 0.8254 accuracy, 0.6373 balanced accuracy, 0.7369 AUC |

---

## Why the Final Model Is Better

The final model improves the project in the following ways:

1. **It follows the SOA research gap more directly**
   - The model separates disease and acquisition representations instead of using only a single fused latent space.

2. **It addresses class collapse**
   - The previous baseline struggled with majority-class prediction and weak positive-class sensitivity.
   - The final model achieved non-zero positive-class recall and F1-score.

3. **It uses leakage-safe splitting**
   - The final notebook reports a subject-wise split and a passed subject leakage check.

4. **It checks acquisition leakage**
   - The final notebook includes linear probe results for acquisition prediction from latent spaces.

5. **It reports multiple metrics**
   - The final evaluation includes accuracy, balanced accuracy, F1, AUC, sensitivity, specificity, confusion matrix, and group-level metrics.

---

## Metrics Explained

### Accuracy

Accuracy measures the overall fraction of correct predictions.

```text
Accuracy = Correct Predictions / Total Predictions
```

Accuracy can be misleading when the dataset is imbalanced.

### Balanced Accuracy

Balanced accuracy averages performance across both classes.

```text
Balanced Accuracy = (Sensitivity + Specificity) / 2
```

This is useful when one class appears more often than the other.

### Sensitivity / Recall Positive

Sensitivity measures how many positive samples were correctly detected.

```text
Sensitivity = True Positives / (True Positives + False Negatives)
```

In this project, sensitivity is important because the earlier baseline struggled to detect positive cases.

### Specificity

Specificity measures how many negative samples were correctly detected.

```text
Specificity = True Negatives / (True Negatives + False Positives)
```

### F1-score

F1-score combines precision and recall.

```text
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

It is useful when the positive class is harder to detect.

### AUC

AUC measures how well the model ranks positive samples above negative samples across thresholds.

### MCC

MCC stands for Matthews Correlation Coefficient. It is useful for imbalanced classification because it considers all parts of the confusion matrix.

---

## Related Work Used in the SOA Survey

The SOA survey reviewed papers related to:

- multimodal self-supervised learning,
- multimodal domain adaptation,
- multimodal domain generalization,
- image harmonization,
- multimodal fusion,
- and methodological rigor in Alzheimer's disease imaging research.

Important papers listed in the SOA survey include:

1. Ali et al. — Multimodal Self-Supervised Learning for Early Alzheimer's
2. Fu et al. — Prediction of Alzheimer's Disease Based on Multi-Modal Domain Adaptation
3. Lteif et al. — Disease-driven Domain Generalization for Neuroimaging-based Assessment of Alzheimer's Disease
4. Choi et al. — Deep Learning-based Amyloid PET Harmonization
5. Li et al. — DiaMond: Dementia Diagnosis with Multi-Modal Vision Transformers Using MRI and PET
6. Zuo et al. — HACA3: A Unified Approach for Multi-site MR Image Harmonization
7. Guan et al. — A Survey of Multimodal Fusion for Alzheimer's Disease Prediction
8. Young et al. — Data Leakage in Deep Learning for Alzheimer's Disease Diagnosis
9. Fedorov et al. — Self-supervised Multimodal Learning for MRI Data

---

## Limitations

The attached files show the following limitations:

1. **ADNI was not included in the dataset selection deliverable**
   - The dataset selection notebook states that ADNI access had been applied for but had not yet been granted.

2. **The final dataset is small for 3D deep learning**
   - The final multimodal dataset contained 156 samples from 52 subjects.

3. **The positive class remains difficult**
   - The final model improved over the collapsed baseline, but positive recall was still 0.3333.

4. **Acquisition proxy groups are batch-based**
   - The final notebook uses batch IDs as acquisition proxy groups.

5. **External validation remains an important future requirement**
   - The SOA survey emphasized external testing cohorts and residual leakage checks as important for robust Alzheimer's imaging models.

---

## Main Lessons Learned

The main lessons from the attached project files are:

1. **Accuracy alone is not enough**
   - The baseline could look acceptable by accuracy while still failing to detect positive cases.

2. **Subject-wise splitting is necessary**
   - The notebooks explicitly check for subject leakage.

3. **Multimodal fusion needs robustness**
   - MRI and PET fusion is useful, but the model must handle modality and acquisition variation carefully.

4. **Disease and acquisition information should be separated**
   - The final model implements disease-acquisition factorization to address the SOA research gap.

5. **Residual leakage should be checked**
   - The final notebook includes latent leakage probes.

---

## Future Work

Future work suggested by the attached files includes:

1. Testing on external cohorts once access is available.
2. Improving positive-class sensitivity.
3. Strengthening residual acquisition leakage checks.
4. Comparing against stronger multimodal fusion baselines.
5. Expanding from batch-level acquisition proxies to richer acquisition metadata when available.
6. Continuing the disease-acquisition latent factorization direction proposed in the SOA survey.

---
