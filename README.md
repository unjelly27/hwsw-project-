# Face Recognition Project

## Overview

This project improves face recognition accuracy by combining:

- strong face alignment using `MTCNN`
- a face-recognition-specific pretrained model, `FaceNet / InceptionResnetV1`
- metric-based evaluation using gallery, template, and kNN matching

The final pipeline was designed after exploring weaker baselines such as `ResNet50` and `ViT-B/16`. The best-performing system achieved more than `95%` accuracy and reached a best template Rank-1 score of `98.11%`.

## Final Pipeline

1. Start from face images.
2. Align and crop faces using `MTCNN`.
3. Build the matched identity dataset.
4. Feed aligned images into pretrained `FaceNet / InceptionResnetV1`.
5. Extract compact identity embeddings.
6. Train/evaluate using metric-based matching.
7. Report results with:
   - Gallery Rank-1
   - Template Rank-1
   - kNN Rank-1

## Main Files

- `train_fullface.py`
  Final training and evaluation pipeline.

- `align_faces_facenet.py`
  Preprocessing script that aligns faces using `MTCNN`.

- `face_identity_dataset.py`
  Dataset builder for the matched identity set.

- `plot_fullface_results.py`
  Generates publication-style plots and result figures.

- `FINAL_REPORT_SECTIONS.md`
  Report-ready summary, methodology, results, and contribution framing.

## Important Artifacts

- `aligned_faces_facenet_160/`
  Final aligned face dataset used in the best pipeline.

- `fullface_best.pth`
  Best checkpoint from the final run.

- `fullface_history.json`
  Training history and validation metrics.

- `results_figures/`
  Saved result plots and visual summaries.

## Final Results

Best final metrics:

- Gallery Rank-1: `0.9779`
- Template Rank-1: `0.9811`
- kNN Rank-1: `0.9732`

This means the final model crossed the target comfortably.

## How To Run

### 1. Align Faces

```bash
.venv310/bin/python align_faces_facenet.py --device cpu
```

### 2. Train / Evaluate Final Model

```bash
.venv310/bin/python train_fullface.py
```

### 3. Generate Result Figures

```bash
.venv310/bin/python plot_fullface_results.py --device auto --eval-tta
```

## Generated Figures

The plotting script saves separate outputs inside `results_figures/`:

- `cmc_curve.png`
- `rank1_metrics.png`
- `training_curves.png`
- `retrieval_examples.png`
- `combined_results.png`
- `metric_summary.json`

## Research Contribution

The project solves one practical issue related to the base paper:

improving recognition accuracy through better face alignment and stronger task-specific deep embeddings.

Instead of reproducing the entire complex architecture from the base paper, this work shows that a simpler but well-designed pipeline can achieve very high recognition accuracy.
