# DeepFake-Detection (In progress)

On this repo there is a University project about the course of Signal, Image and Video where a constraint was to build a project where AI is just a narrow task and not the whole implementation of the project. I will just use AI as a comparison with traditional techniques of video processing. 

## About the project
The project will implement the paper "Efficient Temporally-Aware DeepFake Detection using H.264 Motion Vectors" that proposes to detect DeepFakes using *motion vectors* instead of *Optical flow* techniques, that are more precise but the computational complexity is much higher. Thanks to the low complexity this approach can be used in embedded systems with low computational power. 

## Dataset
The used dataset is a subset of 200 videos from the **FaceForensics++** found on Kaggle at link: [https://www.kaggle.com/datasets/hungle3401/faceforensics?resource=download](https://www.kaggle.com/datasets/hungle3401/faceforensics?resource=download).

## Pipeline

1. **Dataset Setup**: Videos are split into `real/` and `fake/` directories. They are optionally transcoded into a format suitable for H.264 MV extraction.
2. **Face Detection**: We use YuNet to detect and crop the largest face from each frame, resized to 224x224 pixels.
3. **Motion Vector Extraction**: Using PyAV, we extract 16x16 block-level MVs and Information Masks from H.264 encoded videos.
4. **Feature Extraction**: For each frame, compute statistical features from the MV map: mean, variance.

I will develop soon the Classifier as described in the paper. 

## Notes
- The MVs are less precise than Optical Flow but extremely efficient to compute.
