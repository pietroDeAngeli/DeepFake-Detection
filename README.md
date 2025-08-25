# DeepFake Detection

This repository contains a University project for the course **Signal, Image and Video Processing**.  
The main constraint of the project was to design a pipeline where **AI is only a narrow task** and not the whole implementation.  
Here, AI is used as a **classifier** on top of traditional video processing features.

---

## 📄 About the Project
The implementation is based on the paper:

> *Efficient Temporally-Aware DeepFake Detection using H.264 Motion Vectors*  

The core idea is to detect DeepFakes using **H.264 motion vectors (MVs)** instead of **optical flow**.  
While optical flow is more precise, it is also computationally expensive. Motion vectors, being a byproduct of H.264 encoding, are extremely efficient to compute, making this approach suitable for **embedded systems with limited computational resources**.

---

## 📂 Dataset
We use a subset of **200 videos** from the [FaceForensics++ dataset](https://www.kaggle.com/datasets/hungle3401/faceforensics?resource=download).  

The dataset is split into `real/` and `fake/` directories. Each video is processed to extract both:
- **Cropped face frames** (via YuNet detector, resized to 224x224).
- **Motion vectors (MVs) and Information Masks (IMs)** from H.264 streams.

---

## 🔄 Pipeline

1. **Dataset Setup**  
   Videos are organized into `real/` and `fake/` and preprocessed for MV extraction.

2. **Face Detection**  
   Using **YuNet**, the largest face per frame is cropped and resized to 224×224.

3. **Motion Vector Extraction**  
   With **PyAV**, block-level motion vectors (16×16 macroblocks) and Information Masks are extracted from H.264 encoded videos.

4. **Feature Computation**  
   MV + IM features are normalized and stored along with the cropped face images.

5. **Classification**  
   A **two-stream MobileNetV3-based architecture** is trained:
   - One branch for RGB face frames.  
   - One branch for motion vectors (MV + IM).  
   - Outputs are fused with a weighted sum (learnable `α`) to predict Real vs Fake.

6. **Training Strategy**  
   - **Phase 1:** Backbone frozen, only classifier head is trained.  
   - **Phase 2:** Entire network is unfrozen and fine-tuned with a smaller learning rate.  
   - Augmentations include **color changes** and **horizontal flips** (as in the paper).

7. **Evaluation**  
   The model is evaluated with **accuracy, loss, confusion matrix, precision, recall, F1 score**.

---

## 🧪 Results
- Motion vectors provide a cheaper but effective alternative to optical flow.  
- Data augmentation improves robustness to color/lighting variations.  
- The two-stream fusion (RGB + MV) achieves better generalization than single-stream models.  

Artifacts produced during training/evaluation include:
- `best_model.pth` (weights of the best checkpoint).  
- `metrics.json` (accuracy, loss, precision, recall, F1).  
- `confusion_matrix.png` and `.csv`.

---

## ⚙️ Requirements
- Python 3.10+  
- PyTorch  
- OpenCV (`cv2`)  
- PyAV  
- Albumentations  
- Matplotlib, Seaborn, Pandas  

Install with:
```bash
pip install .
```

From the main directory.