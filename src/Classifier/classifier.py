import pandas as pd
from sklearn.ensemble import RandomForestClassifier

dataset_path = "dataset.csv"

def train_classifier():

    data = pd.read_csv(dataset_path)

    X = data.drop(columns=['label'])
    y = data['label']

    # Initialize the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train the classifier
    clf.fit(X, y)
    # Save the trained model
    import joblib
    joblib.dump(clf, 'deepfake_classifier.pkl')

def load_classifier():
    import joblib
    # Load the trained model
    clf = joblib.load('deepfake_classifier.pkl')
    return clf

def predict(video_features):
    clf = load_classifier()
    # Predict the label for the given video features
    prediction = clf.predict([video_features])
    return prediction[0]

