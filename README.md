# DeepFake-Detection

On this repo there is a University project about the course of Signal, Image and Video where a constraint was to bulid a project where AI is just a narrow task and not the whole implementation of the project. I will just use AI as a comparison with traditional techniques of video processing. 

## About the project
The project will implement the paper XXXXXX that proposes to detect DeepFakes using *motion vectors* instead of *Optical flow* tecniques, that are more precise but the computatioal complexity is much higher. Thanks to the low complexity this approach can be used in embedded system with low computational power. 

**Potrei anche pensare di provare entrambe e confrontarle invece di fare solo una, ma sentiamo anche il paerere di chat**


## Dataset
The used dataset is a subset of 200 videos from the **FaceForensics++** found on Kaggle at link: [https://www.kaggle.com/datasets/hungle3401/faceforensics?resource=download](https://www.kaggle.com/datasets/hungle3401/faceforensics?resource=download).


✅ Opzione 1 – Versione Classica (senza Deep Learning)
Titolo: "DeepFake Detection via Motion Vectors and Classical Machine Learning"

🔧 Cosa implementi:
Estrazione delle Motion Vectors (MV) da video codificati H.264.

Pre-processing facciale (usando MTCNN o anche qualcosa di più semplice tipo Dlib).

Estrazione feature da MV:

media, varianza, magnitudo, direzione

percentuale di blocchi vuoti (MV = 0)

(Opzionale) Aggiunta dell’Information Mask (IM) se hai tempo.

Allenamento di un classificatore tradizionale (Random Forest, SVM, LightGBM).

Confronto con:

Optical Flow (con OpenCV o RAFT)

Baseline RGB

