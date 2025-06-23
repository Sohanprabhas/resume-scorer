import pytest
import time
from fastapi.testclient import TestClient
from app.main import app

# Use lifespan-aware context
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client

def test_score_response_time():
    resume_text = "Python, Java, SQL, Docker, System Design, DSA, Microservices"
    payload = {
        "student_id": "timing001",
        "goal": "Amazon SDE",
        "resume_text": resume_text
    }

    start = time.time()
    response = client.post("/score", json=payload)
    end = time.time()

    duration = end - start
    print(f" Response time: {duration:.4f} seconds")

    assert response.status_code == 200
    assert duration < 1.5, "Response time exceeded 1.5 seconds"

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code in [200, 503]
    body = response.json() if response.status_code == 200 else response.json()["detail"]
    assert "status" in body
    assert "checks" in body
    assert "config" in body["checks"]

def test_version_endpoint(client):
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "model_goals_supported" in data
    assert isinstance(data["model_goals_supported"], list)

def test_score_resume_valid(client):
    response = client.post("/score", json={
        "student_id": "123",
        "goal": "Amazon SDE",  # Make sure this goal exists in your config.json
        "resume_text": "Experienced in Python, machine learning, and cloud technologies."
    })
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert isinstance(data["score"], float)
    assert "matched_skills" in data
    assert "missing_skills" in data
    assert "suggested_learning_path" in data

def test_score_resume_unsupported_goal(client):
    response = client.post("/score", json={
        "student_id": "456",
        "goal": "Unknown Goal",
        "resume_text": "Familiar with C++, Java, and system design."
    })
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert isinstance(data["score"], float)

def test_score_resume_invalid_input(client):
    response = client.post("/score", json={
        "student_id": "789",
        "goal": "Amazon SDE"
        # Missing 'resume_text'
    })
    assert response.status_code == 422  # Unprocessable Entity
