#  Resume Scorer Microservice

A FastAPI-based microservice that evaluates resumes against predefined job roles using machine learning and NLP. It scores resumes, provides skill-match feedback, and offers a personalized learning path.

---

##  Features

-  Resume scoring using a trained ML model (TF-IDF + Logistic Regression)
-  Supports multiple job goals like SDE, ML Intern, DevOps, etc.
-  Skill matching with exact/partial match options
-  Health check & versioning endpoints
-  Configurable scoring threshold and analytics
-  Docker-ready deployment

---

##  Project Structure
resume_score/
├── app/
│ ├── main.py # FastAPI app logic
│ ├── scorer.py # ResumeScorer class
├── config/
│ └── config.json # Configuration for goals and scoring
├── data/
│ ├── goals.json # List of supported goals
│ └── training_*.json # Training data files
├── model/
│ ├── tfidf_vectorizer.pkl # Shared TF-IDF vectorizer
│ └── *.pkl # Trained models for each goal
├── train/
│ └── generate_shared_vectorizers.py
├── test/
│ ├── test_score.py
│ └── test_tfidf_shared_vectorizer.py
├── requirements.txt
├── Dockerfile
└── README.md



---

## ⚙️ Configuration (`config/config.json`)

Defines model behavior, scoring thresholds, performance limits, API settings, and logging:

```json
{
  "version": "1.0.0",
  "minimum_score_to_pass": 0.65,
  "model_goals_supported": [
    "Amazon SDE", "ML Internship", "Data Scientist", ...
  ],
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4
  }
}

# Model Training

Run to create a shared TF-IDF vectorizer:

"python train/generate_shared_vectorizers.py"

Train individual models using your own script or Jupyter notebook using scikit-learn.


#Running the API

->Locally
"uvicorn app.main:app --reload"

->With Docker
"docker build -t resume-scorer .
docker run -p 8000:8000 resume-scorer"


#Testing
->Install required packages:
"pip install -r requirements.txt"

->Run tests:
"python test/test_score.py
python test/test_tfidf_shared_vectorizer.py"


->API Endpoints
"GET /health – Health check

GET /version – Get API and config version

POST /score – Submit resume JSON for scoring"

Example payload:
{
  "goal": "Amazon SDE",
  "resume_text": "Experience in Python, AWS, REST APIs..."
}


# Requirements
"Python 3.8+

FastAPI

scikit-learn

joblib

pydantic

uvicorn

httpx (for testing)"

->Install:
"pip install -r requirements.txt"



Author
Katari Sohan Prabhas
Resume Scorer Microservice | Internship Project

