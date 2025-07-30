# Paper notes

## Abstract
The paper aim to implement DeepFake detection using **Motion vectors** and not *Optical flow* techniques that are much more expensive.

## Intro
DeepFakes are a problem in nowadays life because of  defamation and political misuse to video-call scams and pornography. 

The two main problems of the current solutions are:
 - We only keep track of the frames and not on the information given by the video.
 - Insufficient generalizabilty. 

### Problem
Many solutions now rely on evaluation the per-frame fake problility without really consider the local motion of the video like the normal pace movement or constant eye color. A solution to this problem is **Optical flow**, however is really expensive especially with the continous growing of the online streaming platform and this could cause an efficiency bottleneck in the detection. 

### Solution
The paper propose the use of **Motion vectors** H.264 as a method to approximate the more complex solution and better generalize the classification problem, then we'll compare the MV model solution to the OF one. 

We'll use an MLP to detect the DeepFake and comparing it with the OF baseline showing an improving of 14%.

## Related works

### DeepFake detection
We can categorize DeepFake detection algorithms in 2 categories:
 - *Image-level*: they just pay attention to the changes in the frames and they don't use the temporal information in videos such as cross frame inconsistency
 - *Temporally-aware*
  
### Temporally-aware DFD
Solutions that usually involves LTSMs, Transformes and process multiple frames at each time. It has been shown that usually two types of networks: one evaluates the RGB features and the other one the OF achieve better overall accuracy (but with high complexity cost). In the paper they used H.264 codec as approximation of OF. 


#### Motion vector as motion approximation
H.264 are usually used to compress video, by dividing the frames into chunks and then calculate only the motion of the blocks intstead of the whole frame. 

## Preliminaries and methodology
Now we'll talk about OF and MV and then the pipeline of the proposed solution. 

### Optical flow
OF represent the projection of every 3D point's trace to the image place (movement of the pixels) this approach is often used in object recognition in videos but it's still an open problem due to the nature of the video where an object could be *occlused*, *non rigid movement* or *blurred image*. 

The paper use the RAFT model which use a stochastic prior to detect the OF and it has top score on both dataset. 

### H.264 motion vectors
MVs are part of H.264 video compression scheme that exploits temporal and spatial redundancy for a better compression rate dividing the image in **macroblocks**. 

There exist three types of frames in the H.264 format:
 - *Intra-coded* (they don't contain temporal information)
 - *Predicted* (P-frames): they only refers to future frames 
 - *Bidirectional predicted* (B-frames): they refer to both future and previous frames. 

They are both limited to 16 frames and not all frames are available because sometimes they are encoded into *I-frames macroblock*. Infact there are some macroblocks that are encoded "in plain" and I think they are use as reference block and they does not depend on the other blocks. These blocks are called **Information Mask (IM)** and they are labelled by a binary mask. 

#### How the MV are evaluated with H.26X

1. **Motion Estimation**: We estimate the MV of each macroblock, the estimate has been done using easy tecniques such as the median neighbors... the exact method depends on the chosen version of H.26X.
2. **Motion vector Difference (MVD)**: Using our first guess we estimate the real MV *v* for that block. The MVD is the difference between the real one and the estimated one.  We only send the MVD to the decoder to save space. 
3. **Decode**: The decoder receive the estimated MV (using the same method above) and add the MVD received. 

This process requiries really low computational power. 

## Data preprocessing

### Face detector 
We use a face detector to find the face using a pretrained MTCNN, we could find multiple faces but the paper only use the bigger one. Then after finding the four corner points the dimension that is the smallest is padded with pixels to obtain a **square** bounding box. The face is resized to 224x224 resolution and the resulting frames are normalized using standard resolution

### Motion vectors
Then we extract the MV and IM from the faces and we stack them toghether to use them as an input. We can have 2 imput formats:
 - 4D input: (past-x, past-y, future-x, future-y) -> says if the movement is to the past or future and in which direction. 
 - 6D input: If IM has been used we add 2 dimesions.


Since some blocks can be without MV (or MV = 0) then the bocks are rescaled to the target resolution. 

### Data Augmentation
Add some data Augmentation tecniques to improve the generalization using the library *albumentations*. The applied transformations include:
 - Blur
 - RGB shift
 - Hue-saturation adjustment
 - Gaussian noise
 - Fancy PCA
 - Random brightness contrast
 - Grey-scale transformation

**NB**: Every tranformation has a probability to be applied, they are not always applied. 

### Classifier
They choosed MobileNetV3 as classifier's backbone, it's a neural network that require low computations but accurate. 

For the paper they applied 3 modifications:
 - They substiture the final output with a neuron that gives a probability [0,1] that is the probability of the video to be fake.
 - Number of input channel because they're working with RGB image, MVs, IMs + IMs. 
 - 2 streams, one for RGB and one for MVs. Each stream pass throught MobileNetV3, then they combine the results concatenando the last features and they produce a single output. 

At the end for each video they average the predictions on 100 random frames to get the final decision. 

#### Loss Function
They use the **Binary cross entropy** (fake or real): $L = y \log \hat{y} + (1 - y) \log (1 - \hat{y})$.

## Experiments

### Experimental setups

#### Datasets
FaceForensics++ on the HQ version C23. 
The dataset contains 1000 YT videos manipulated with different DeepFakes methods:
 - FaceShifter (F5)
 - FaceSwap (FSwap)
 - DeepFakes (DF)
 - Face2Face (F2F)
 - NeuralTexture (NT)
Every video has a real and a fake version. 

#### Baselines
They compare their model (based on MV) with:
 - base RGB
 - RAFT (advanced optical flow)
  
RAFT has been choosen because it has good performances (it's used with the same params of the original paper).

To reduce the memory allocated they save the OF using grey-scale images JPEG lossless.

#### Implementation details
PyTorch and PyTorch Lightning.

Hardware: 2 GPU NVIDIA GTX Titan X, 12 GB RAM, CPU Intel Xeon E5-2680 v3.

Adam Optimizer

Balanced data: for each type of DeepFake they have the same number of real video.

8 Epochs training using the model with the lowest error on validation. 

### DeepFake detection accuracy
Comparing the two streams models (use temporal data) and the RGB-only but the results just shows that the RGB easyily overfits the data. (They should have used a more complex dataset). 

### Cross forgery generalization ability
Here we want to understand if a model trained on a single type of DeepFake works well also on unseen types of DF. The MV approach has better generalization than RGB-only. 

### Classification with only temporal data
In this section we're going throught the classification without using the temporal data (just one stream). 

We'll use 4 configurations: 
 - OF: Optical flow with RAFT
 - MV_P: past MV
 - MV+IM: MV + IM
 - MV+IM+H: like the previous one but with deeper concatenation

We have better results using MV than OF, so they are deep enought to classify deepfakes. 

### Computational cost: OF vs MV
The MV are much 1000 times faster than RAFT. 

## Future works
We still have a bottleneck that is made by the faceRec that is quite slow, moreover the MV has a 16 times lower resolution and this could be a problem if the "fake area" is really small.

--> Estimate the optical flow only where necessary, or using MV to make it lighter