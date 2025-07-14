🧠 Pipeline Step-by-Step
🔹 1. Dataset setup
Scarica da FaceForensics++ (HQ / C23 preferibile)

Struttura:

go
Copia
Modifica
FF++/
├── real/
└── fake/
├── real_h264/
└── fake_h264/

🔹 2. Face detection

Per ogni frame, estrai la faccia (MTCNN) più grande

Ritaglia e resize a 224×224 px

🔹 3. Estrazione dei Motion Vectors

Blocchi 16x16 (default)

Converti i video MP4 a un formato che supporti h264 per estrarre i MV

Usa PyAV per accedere direttamente ai motion vectors

(puoi usare anche l’IM se riesci a identificarla)

🔹 4. Feature extraction
Per ogni frame (o per video):

Media dx, dy

Varianza dx, dy

Media magnitudo (sqrt(dx² + dy²))

Percentuale di vettori nulli (mv = 0)

Entropia delle direzioni (bucket angoli in 8 bin)

Densità di blocchi con movimento

Salva come:

python
Copia
Modifica
{
  "features": [...],  # lista float
  "label": 0 or 1     # fake o real
}
🔹 5. Classifier
Costruisci un dataframe con:

python
Copia
Modifica
import pandas as pd
df = pd.DataFrame(data=feature_list, columns=feature_names)
df["label"] = labels
Allenamento con:

python
Copia
Modifica
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
🔹 6. Valutazione
Matrice di confusione

Accuracy, Precision, Recall, F1

Tempo di esecuzione totale

(Opzionale) PCA per visualizzare separabilità

🔹 7. Confronto con Optical Flow (opzionale)
Usa cv2.calcOpticalFlowFarneback(...) per calcolare il flow frame-to-frame, poi estrai le stesse feature dei MV.
Confronta tempi di esecuzione e performance.

📁 Struttura progetto consigliata
sql
Copia
Modifica
mv-deepfake-detector/
├── data/
│   ├── real/
│   └── fake/
├── mv_extraction/
│   └── extract_mv.py
├── face_crop/
│   └── detect_faces.py
├── features/
│   └── compute_features.py
├── classifier/
│   └── train_model.py
├── utils/
│   └── opticalflow_baseline.py (opzionale)
└── README.md
Vuoi che ti generi anche un piccolo esempio di compute_features.py o extract_mv.py?