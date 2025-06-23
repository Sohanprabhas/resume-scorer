import json
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def clean_text(text):
    """Preprocess resume text: lowercase, remove punctuation, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def load_and_preprocess_data(file_paths):
    """Load and preprocess resume texts from JSON files."""
    all_resume_texts = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping.")
            continue
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            for item in data:
                if 'resume_text' in item:
                    cleaned_text = clean_text(item['resume_text'])
                    if cleaned_text:
                        all_resume_texts.append(cleaned_text)
                else:
                    print(f"Warning: Missing 'resume_text' in item: {item}")
        except json.JSONDecodeError:
            print(f"Error: File {file_path} contains invalid JSON.")
            raise
    return all_resume_texts

def train_and_save_vectorizer():
    """Train TF-IDF vectorizer and save to app/model/tfidf_vectorizer.pkl."""
    training_files = ['data/training_amazon_sde.json']
    resume_texts = load_and_preprocess_data(training_files)
    if not resume_texts:
        raise ValueError("No valid resume texts found for training.")

    print(f"Training vectorizer with {len(resume_texts)} resume samples")

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    vectorizer.fit(resume_texts)

    output_path = 'app/model/tfidf_vectorizer.pkl'
    joblib.dump(vectorizer, output_path)
    print(f"TF-IDF vectorizer saved to {output_path}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer

def train_and_save_model(goal, training_file):
    """Train Logistic Regression model for a specific goal."""
    if not os.path.exists(training_file):
        print(f"Error: Training file {training_file} not found for goal {goal}")
        return
    with open(training_file, 'r') as f:
        data = json.load(f)
    resume_texts = [clean_text(item['resume_text']) for item in data]
    labels = [item['label'] for item in data]
    if len(resume_texts) < 50:
        print(f"Warning: Only {len(resume_texts)} samples for {goal}, expected at least 50")
    label_counts = {0: labels.count(0), 1: labels.count(1)}
    print(f"Label distribution for {goal}: {label_counts}")

    vectorizer = joblib.load('app/model/tfidf_vectorizer.pkl')
    features = vectorizer.transform(resume_texts)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{goal} model accuracy: {accuracy:.2f}")
    if accuracy > 0.8:
        model_path = f"app/model/{goal.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    else:
        print(f"Model accuracy below 80% for {goal}. Tuning or more data needed.")
    return accuracy

if __name__ == "__main__":
    try:
        train_and_save_vectorizer()
        accuracy = train_and_save_model("Amazon SDE", "data/training_amazon_sde.json")
        if accuracy <= 0.8:
            print("Accuracy <= 80%. Consider adding more samples or tuning parameters.")
    except Exception as e:
        print(f"Error: {e}")