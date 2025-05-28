# MusDis-Sports Dataset

## Overview

The **MusDis-Sports** dataset is developed for ergonomic posture risk assessment using multimodal inputs. It consists of annotated images with **REBA (Rapid Entire Body Assessment)** class labels, focusing on assessing musculoskeletal disorder risks in various sports and physical activity contexts.

Each image in the dataset contains **66 skeletal coordinates**, offering detailed posture representations.

- **Total Samples**: 8,989  
- **Total Coordinates**: 592,374 (66 coordinates × 8,989 samples)

## Class-wise Sample Distribution

| REBA Class | Samples |
|------------|---------|
| Class 1    | 1083    |
| Class 2    | 1213    |
| Class 3    | 1181    |
| Class 4    | 1332    |
| Class 5    | 1144    |
| Class 6    | 1117    |
| Class 7    |  888    |
| Class 8    | 1031    |
| **Total**  | **8989**|

Higher REBA classes (e.g., Class 4 - 8) indicate riskier postures associated with a greater likelihood of musculoskeletal disorders.

---

## Dataset Access

The dataset is available here:  
📂 [MusDis-Sports Google Drive Link](https://drive.google.com/drive/folders/1-Cx5NTvR9uMMXvLTIIgcETqSxgCGmAdX?usp=sharing)

To request access:
- Use an institutional email (ending in `.edu`)
- Send an email to: **m.raiaan.cs@gmail.com**  
  Include the following:
  - Your name and homepage or Google Scholar profile
  - Your primary affiliation and job title
  - The intended use of the dataset (research only)

Responses will be provided within 2 business days.

---

## Classification Framework: ViSK-GAT

We propose a novel multimodal framework called **ViSK-GAT** for posture risk classification.
<p align="center">
  <img src="https://github.com/mak-raiaan/MusDis-Sports_Dataset/blob/main/Pictures/OverallModel.png" alt="ViSK-GAT" width="1000"/>
</p>

### Description

**ViSK-GAT** integrates:
- **Visual features** from RGB images
- **Skeletal coordinates** using a hybrid backbone
- **Fine-Grained Attention Module** for enhanced feature selection
- **Multimodal Geometric Correspondence Module** for aligning modalities

This design enables robust assessment of ergonomic risk by leveraging both pose geometry and visual cues.

---

## Citation

*Please cite this dataset and framework in any publications derived from it. A formal citation will be provided upon publication.*

---
