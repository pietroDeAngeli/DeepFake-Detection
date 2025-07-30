# DeepFake Detection Pipeline (H.264 Motion Vectors)

## Overview
This system detects DeepFakes in videos by leveraging **Motion Vectors (MV)** and **Information Masks (IM)** embedded in H.264-encoded videos, avoiding the computational cost of **Optical Flow (OF)**.

---

## 1. Preprocessing

### a. Frame Extraction and Face Detection
- Videos are opened using PyAV.
- A fixed number of random frames (P-frames) are sampled.
- Faces are detected using `FaceDetectorYN`; the largest face is selected.
- The face is cropped and resized to **224×224**.

### b. MV and IM Extraction
- Motion vectors and information masks are extracted from decoded frames.
- Data is resized to match the face crop using interpolation.

---

## 2. Feature Computation
- Feature maps **(mvx, mvy, im)** are computed per frame.
- Motion vectors are **standardized** only in inter-coded areas (`im == 0`).
- Features are packed into torch tensors with shape `(H, W, 3)`.

---

## 3. Model Input
- Input:
  - 3D: (past-x, past-y, IM_past)
- A **two-stream network** may be used (RGB + MV/IM).

---

## 4. Classification
- Backbone: **MobileNetV3** (lightweight and accurate).
- Modifications:
  - First convolutional layer supports multiple input channels.
  - Output layer produces a binary probability (real/fake).
  - Two separate streams (RGB and MV), concatenated before the final decision.

---

## 6. Final Decision
- **100 random frames** are sampled from each video.
- The final fake probability is the **average of the frame-wise predictions**.

## 7. UI
I will implement also a simple UI to show the classification results.