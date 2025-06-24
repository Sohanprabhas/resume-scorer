import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# 1. Empty resume text
def test_empty_resume(client):
    response = client.post("/score", json={
        "student_id": "edge001",
        "goal": "Amazon SDE",
        "resume_text": ""
    })
    assert response.status_code == 200
    assert response.json()["score"] == 0.

# 2. Unknown goal
def test_unknown_goal(client):
    response = client.post("/score", json={
        "student_id": "edge002",
        "goal": "Alien Job",
        "resume_text": "Python, Java, AWS"
    })
    assert response.status_code == 200
    assert "score" in response.json()
    assert isinstance(response.json()["score"], float)

# 3. Very long resume text
def test_very_long_resume(client):
    long_text = "Python, Java, AWS, Docker. " * 500
    response = client.post("/score", json={
        "student_id": "edge003",
        "goal": "Amazon SDE",
        "resume_text": long_text
    })
    assert response.status_code == 200
    assert isinstance(response.json()["score"], float)

# 4. Missing fields (student_id)
def test_missing_student_id(client):
    response = client.post("/score", json={
        "goal": "Amazon SDE",
        "resume_text": "Python"
    })
    assert response.status_code == 422  # Validation error

# 5. Missing fields (resume_text)
def test_missing_resume_text(client):
    response = client.post("/score", json={
        "student_id": "edge004",
        "goal": "Amazon SDE"
    })
    assert response.status_code == 422

# 6. Gibberish input
def test_gibberish_input(client):
    response = client.post("/score", json={
        "student_id": "edge005",
        "goal": "Amazon SDE",
        "resume_text": "asdjlk234###@@!"
    })
    assert response.status_code == 200
    assert isinstance(response.json()["score"], float)


# 7. Synonyms only (if your scorer handles synonyms)
def test_synonyms_only(client):
    response = client.post("/score", json={
        "student_id": "edge006",
        "goal": "Amazon SDE",
        "resume_text": "programming, scripting, development"
    })
    assert response.status_code == 200
    assert "matched_skills" in response.json()

# 8. Duplicate student_id (should work like any other)
def test_duplicate_student_id(client):
    payload = {
        "student_id": "duplicate",
        "goal": "Amazon SDE",
        "resume_text": "Python, Java, SQL"
    }
    response1 = client.post("/score", json=payload)
    response2 = client.post("/score", json=payload)
    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json() == response2.json()

# 9. Null values explicitly
def test_null_values(client):
    response = client.post("/score", json={
        "student_id": None,
        "goal": "Amazon SDE",
        "resume_text": "Python"
    })
    assert response.status_code == 422

# 10. No JSON payload
def test_no_json(client):
    response = client.post("/score")
    assert response.status_code in [400, 422]

# 11. Resume with all matching skills (perfect match)
def test_perfect_match(client):
    perfect_resume = "Python, Distributed Systems, Microservices, Testing, Algorithms, Docker, Object-Oriented Programming, System Design, CI/CD"
    response = client.post("/score", json={
    "student_id": "edge001",
    "goal": "Amazon SDE",  # Make sure this matches exactly with your goals_data keys
    "resume_text": perfect_resume
})

    assert response.status_code == 200
    assert response.json()["score"] > 0.9
