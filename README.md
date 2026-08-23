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


## Classification Framework: ViSK-GAT

Online version can be access at:

[![Elsevier](https://img.shields.io/badge/Elsevier-10.1016%2Fj.compeleceng.2026.111281-FF6C00?logo=elsevier&logoColor=white)](https://doi.org/10.1016/j.compeleceng.2026.111281)

Model Diagam:
<p align="center">
  <img src="Pictures/OverallModel.png" alt="ViSK-GAT">
</p>

**ViSK-GAT** integrates:
> **Visual features** from RGB images

> **Skeletal coordinates** using a hybrid backbone

> **Fine-Grained Attention Module** for enhanced feature selection

> **Multimodal Geometric Correspondence Module** for aligning modalities

## Directories

```text
.
├── models/
│   ├── __init__.py
│   ├── image_encoder.py
│   ├── residual_block.py
│   ├── fgam.py
│   ├── mgcm.py
│   ├── transformer_block.py
│   └── multimodal_model.py
├── src/
│   ├── __init__.py
│   ├── data_generation.py
│   ├── data_loader.py
│   ├── reba_scoring.py
│   ├── train.py
│   └── evaluate.py
├── notebook/
│   └── PostureCode.ipynb
├── Pictures/
├── requirements.txt
└── README.md
```

###  Model Data Flow

`CoordinateImageDataset` loads an RGB image and its 66 skeletal coordinate values; `CustomImageEncoder` extracts a sequence of visual tokens from the image; `FineGrainedAttentionModule` refines the visual tokens using self-attention `MultimodalGeometricCorrespondenceModule` aligns visual tokens with the 33 `(x, y)` joint tokens; `LightweightTransformerBlock` refines the fused multimodal representation. And the classifier returns logits for the eight REBA risk classes.

The main model is available as `models.MultimodalModel`. The training helpers are defined in `src.train`, and evaluation metrics are available through `src.evaluate.evaluate_predictions`.


## Citation
Paper accepted on 1 Jun 2026 in 
*Computers and Electrical Engineering*

```bibtex
@article{rahman2026fine,
    title={A fine-grained attention and geometric correspondence model for musculoskeletal risk classification in athletes using multimodal visual and skeletal features},
    author={Rahman, Md Abdur and Karim, Wasimul and Raiaan, Mohaimenul Azam Khan and Shermin, Tamanna and Islam, Md Rafiqul and Hussain, Mukhtar and Azam, Sami},
    journal={Computers and Electrical Engineering},
    volume={138},
    pages={111281},
    year={2026},
    publisher={Elsevier}
}