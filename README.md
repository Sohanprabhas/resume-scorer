
#  Resume Scoring Microservice

The Resume Scoring Microservice is a smart, offline-ready system that evaluates the relevance of a candidate's resume for specific job roles such as **Amazon SDE** or **ML Internship**. It processes raw resume text, matches it against role-specific skill requirements, assigns a score, and suggests personalized learning paths for missing skills. Built with **FastAPI**, **TF-IDF**, **Logistic Regression**, and containerized using **Docker**, this project streamlines resume evaluation for recruiters and candidates alike.



## Why This Project?

Traditional resume screening is manual, inconsistent, and inefficient. Companies receive thousands of resumes daily, and it's not feasible to evaluate each with precision. Candidates also lack actionable feedback. This microservice solves that by:

- Automating resume evaluation.
- Matching resumes to job-specific skill requirements.
- Returning scores and improvement suggestions.



##  Key Features

-  **Offline Resume Scoring** using TF-IDF + Logistic Regression (per job goal).
-  **Outputs** a score, matched/missing skills, and a custom learning path.
-  **Skill Matching Logic** based on `goals.json` skill sets.
-  **Configurable** thresholds and model goals via `config.json`.
-  **Unit Tested** with edge case and fallback support.
-  **Dockerized** for platform-independent deployment.
-  **REST API** with full Swagger (OpenAPI) documentation.



##  How It Works (Architecture Overview)

1. **Resume Text** is submitted to the `/score` API.
2. The text is preprocessed and vectorized using **TF-IDF**.
3. A **Logistic Regression model** (trained per goal) predicts the relevance score.
4. Skills in the resume are matched against goal-specific requirements.
5. The system returns a **score**, **matched & missing skills**, and a **learning path**.



##  Technologies Used

- **Programming Language**: Python 3.x
- **Framework**: FastAPI
- **Machine Learning**: scikit-learn (TF-IDF + Logistic Regression)
- **Containerization**: Docker
- **Testing**: Pytest
- **Development**: VS Code, GitHub



##  Project Structure

```
resume-scorer/
├── app/
│   ├── main.py                 # FastAPI app entry
│   ├── scorer.py               # Model and scoring logic
│   └── model/                  # Pretrained models and vectorizers
├── data/
│   ├── goals.json              # Required skills per job role
│   └── training_amazon_sde.json
├── tests/
│   └── test_score.py           # Unit tests
├── config.json                 # App configuration
├── schema.json                 # API schema definitions
├── requirements.txt            # Dependencies
├── Dockerfile                  # Docker build instructions
├── README.md                   # Project documentation
```



## API Endpoints

| Method | Endpoint     | Description                        |
|--------|--------------|------------------------------------|
| POST   | `/score`     | Submit resume for scoring          |
| GET    | `/health`    | Service health check               |
| GET    | `/version`   | Returns API/model/config version   |

### Sample Input (POST `/score`)
```json
{
  "student_id": "stu_001",
  "goal": "Amazon SDE",
  "resume_text": "Final year CSE student skilled in Java, Python, SQL, and REST APIs."
}
```

###  Sample Output
```json
{
  "score": 0.83,
  "matched_skills": ["Java", "SQL"],
  "missing_skills": ["System Design", "Data Structures"],
  "suggested_learning_path": [
    "Review system design concepts",
    "Practice LeetCode data structures problems"
  ]
}
```

---

##  Testing

Run the following to test:
```bash
pytest tests/
```

Tests validate:
- Proper score and label for known inputs
- Graceful failure for unknown goals or empty input
- API format and schema correctness



##  Docker Instructions

###  Build Docker Image
```bash
docker build -t resume-scorer.
```

###  Run Docker Container
```bash
docker run - 8000:8000 resume-scorer
```

Visit the API at: [http://localhost:8000/docs](http://localhost:8000/docs)



##  Configuration – `config.json`

json
{
  "version": "1.0.0",
  "minimum_score_to_pass": 0.6,
  "log_score_details": true,
  "model_goals_supported": ["Amazon SDE", "ML Internship"],
  "default_goal_model": "Amazon SDE"
}


If the config is missing or invalid, the service will terminate with an error.



##  Future Scope

- Support multilingual resumes (e.g., Hindi, Telugu)
- Add PDF/DOCX parsing support
- Build a recruiter dashboard interface
- Use BERT/GPT for contextual scoring
- Integrate email feedback for users



##  Licensing & Attribution

- Built by **Katari Sohan Prabhas** for academic submission under the **Turtil Internship Program**
- Educational use only. All logic and model code is original.
- Inspired by modern ML-based hiring platforms and GPT-based brainstorming



##  Contact

-  Email: sohannaidu040@gmail.com
