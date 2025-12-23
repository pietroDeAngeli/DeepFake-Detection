# DeepFake Detection

This repository contains a University project for the course **Signal, Image and Video Processing**.  

I decided to implement a [paper](https://arxiv.org/abs/2311.10788) where I could code a full ML pipeline for video processing using `opencv`, `ffmpeg` and `Pytorch`.

![alt text](image.png)

---

## About the Project
The implementation is based on the paper:

> *Efficient Temporally-Aware DeepFake Detection using H.264 Motion Vectors* 

The core idea is to detect DeepFakes using **H.264 motion vectors (MVs)** instead of **optical flow**.  
While optical flow is more precise, it is also computationally expensive. Motion vectors, being a byproduct of H.264 encoding, are extremely efficient to compute, making this approach suitable for **embedded systems with limited computational resources**.

---

## Dataset
I used [FaceForensics++ dataset](https://www.kaggle.com/datasets/hungle3401/faceforensics?resource=download).  

The dataset is split into `real/` and `fake/` directories. Each video is processed to extract both:
- **Cropped face frames** (via YuNet detector, resized to 224x224).
- **Motion vectors (MVs) and Information Masks (IMs)** from H.264 streams.
- **Splits** 70% training, 15% validation, 15% test.

## Goal
The Goal is to reproduce the results of the paper but using a lighter model for inference. 

---

## Pipeline

1. **Dataset Setup**  
   Videos are organized into `real/` and `fake/` and preprocessed by extracting fraces from 100 random samples in the video and extracting the motion vectors from the extracted crops. 

2. **Face Detection**  
   Using **YuNet**, the largest face per frame is cropped and resized to 224×224. The paper suggests to use MTCNN but I decided to use YuNet since it's lighter and optimized for CPU. 

3. **Motion Vector Extraction**  
   With **PyAV**, block-level motion vectors (16×16 macroblocks) and Information Masks are extracted from H.264 encoded videos.

4. **Feature Computation**  
   MV + IM features are normalized and stored along with the cropped face images.

5. **Classification**  
   A **two-stream MobileNetV3-based architecture** is trained:
   - One branch for RGB face frames.  
   - One branch for motion vectors (MV + IM).  
   - Outputs are averaged to find the final prediction.

6. **Training Strategy**  
   Since I had really few videos for the training I divided the training into 2 phases:
   - **Phase 1 (first 5 epoches):** Backbone frozen, only classifier head is trained.  
   - **Phase 2:** Entire network is unfrozen and fine-tuned with a smaller learning rate.  
   - Augmentations include **color changes** and **horizontal flips** (as in the paper).

7. **Evaluation**  
   The model is evaluated with **accuracy, loss, confusion matrix**.

---

## Results
The resulting models achieves the same accuracy of the paper. It means I've been able to implement the paper even without technical details on the implementation but using a lighter inference because I replaced MTCNN with YuNet for the face extraction which is the bottleneck of the paper. 

For more details on the project check the [documentation file](https://github.com/pietroDeAngeli/DeepFake-Detection/blob/main/DeepFake_detection___documentation.pdf) or feel free to send me an email.  

---

## Requirements
- Python=3.10
- PyTorch  
- OpenCV (`cv2`)  
- PyAV  
- Albumentations  
- Matplotlib, Seaborn, Pandas

## Installation
```
conda create --name df python=3.10
conda activate df
pip install torch torchvision
pip install -e .
```

## References
```
@article{gronquist2023efficient,
  title={Efficient Temporally-Aware DeepFake Detection using H. 264 Motion Vectors},
  author={Gr{\"o}nquist, Peter and Ren, Yufan and He, Qingyi and Verardo, Alessio and S{\"u}sstrunk, Sabine},
  journal={arXiv preprint arXiv:2311.10788},
  year={2023}
}
```
