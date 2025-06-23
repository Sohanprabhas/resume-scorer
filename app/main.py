import os
import json
import logging
import joblib
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from datetime import datetime,timezone
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from app.scorer import ResumeScorer
from contextlib import asynccontextmanager
from app.utils import load_config  
from app.logger import logger  


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
GOALS_PATH = os.path.join(BASE_DIR, "data", "goals.json")
MODEL_DIR = os.path.join(BASE_DIR, "app", "model")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
REGISTRY_PATH = os.path.join(MODEL_DIR, "model_registry.json")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("resume-scorer")


# Define request & response models
class ScoreRequest(BaseModel):
    student_id: str = Field(..., description="Unique student identifier")
    goal: str = Field(..., description="Target position or domain (e.g., Amazon SDE)")
    resume_text: str = Field(..., description="Full plain-text resume content")

class ScoreResponse(BaseModel):
    score: float = Field(..., description="Match score between 0.0 and 1.0")
    matched_skills: list[str] = Field(..., description="Skills found in resume that match goal")
    missing_skills: list[str] = Field(..., description="Skills required for goal but not found in resume")
    suggested_learning_path: list[str] = Field(..., description="Recommended steps to improve match")

# Load configuration at startup
def load_config() -> Dict[str, Any]:
    """Load and validate config.json file."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Validate required fields
        required_fields = [
            "version", 
            "minimum_score_to_pass", 
            "log_score_details", 
            "model_goals_supported", 
            "default_goal_model"
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
                
        return config
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.critical(f"Failed to load config: {str(e)}")
        raise RuntimeError(f"Configuration error: {str(e)}")

# Load goals data
def load_goals() -> Dict[str, list]:
    """Load goals.json containing required skills per goal."""
    goals_path = os.path.join(os.path.dirname(__file__), "..", "data", "goals.json")
    
    try:
        with open(goals_path, "r") as f:
            goals = json.load(f)
        return goals
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.critical(f"Failed to load goals data: {str(e)}")
        raise RuntimeError(f"Goals data error: {str(e)}")
async def _check_app_state() -> Dict[str, Any]:
    """Check if FastAPI app state is properly initialized."""
    try:
        if not hasattr(app.state, 'scorer'):
            return {
                "status": "error",
                "message": "ResumeScorer not initialized in app state",
                "details": {"missing_component": "scorer"}
            }
            
        if not hasattr(app.state, 'config'):
            return {
                "status": "error", 
                "message": "Config not loaded in app state",
                "details": {"missing_component": "config"}
            }
            
        return {
            "status": "ok",
            "message": "App state properly initialized",
            "details": {
                "has_scorer": True,
                "has_config": True,
                "scorer_type": type(app.state.scorer).__name__
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": "App state check failed",
            "details": {"error": str(e)}
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        config = load_config()

        # Normalize model goal names in config
        config["model_goals_supported"] = [g.strip().lower().title() for g in config["model_goals_supported"]]

        # Load and normalize goals data
        goals_path = os.path.join(os.path.dirname(__file__), "..", "data", "goals.json")
        with open(goals_path, 'r') as f:
            raw_goals = json.load(f)
            goals = {k.strip().lower().title(): v for k, v in raw_goals.items()}

        model_dir = os.path.join(os.path.dirname(__file__), "model")

        # Attach scorer and config to app state
        app.state.scorer = ResumeScorer(
            model_dir=model_dir,
            goals_path=goals_path,
            config=config,
            goals=goals
        )

        app.state.config = config

        logger.info(
            f"Resume Scorer initialized with {len(app.state.scorer.goals_data)} goals "
            f"and {len(config['model_goals_supported'])} supported models"
        )

    except Exception as e:
        logger.critical(f"Failed to initialize application: {str(e)}")
        os._exit(1)

    yield


# Create FastAPI app using the lifespan context
app = FastAPI(
    title="Resume Scoring Microservice",
    description="Evaluates resumes against job goals and provides skill-based insights",
    version="1.0.0",
    lifespan=lifespan  # ← this replaces on_event("startup")
)
# Error handler for internal exceptions
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Health check endpoint

@app.get("/health")
async def health_check():
    health_status = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "resume-scorer",
        "checks": {}
    }
    overall_healthy = True

    try:
        checks = {
            "config": await _check_config(),
            "vectorizer": await _check_vectorizer(),
            "models": await _check_models(),
            "goals": await _check_goals(),
            "registry": await _check_model_registry(),
            "performance": await _check_performance()
        }

        for k, v in checks.items():
            health_status["checks"][k] = v
            if v["status"] not in ("ok", "warning"):
                overall_healthy = False

        if not overall_healthy:
            health_status["status"] = "degraded"

    except Exception as e:
        health_status["status"] = "error"
        health_status["error"] = str(e)

    if health_status["status"] == "error":
        raise HTTPException(status_code=503, detail=health_status)
    elif health_status["status"] == "degraded":
        raise HTTPException(status_code=200, detail=health_status)

    return health_status

async def _check_config() -> Dict[str, Any]:
    try:
        if not os.path.exists(CONFIG_PATH):
            return {
                "status": "error",
                "message": "config.json not found",
                "details": {"path": CONFIG_PATH}
            }

        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)

        required_fields = [
            "version", "minimum_score_to_pass",
            "model_goals_supported", "default_goal_model"
        ]
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            return {
                "status": "error",
                "message": "Invalid config format",
                "details": {"missing_fields": missing_fields}
            }

        return {
            "status": "ok",
            "message": "Config loaded successfully",
            "details": {
                "version": config.get("version"),
                "supported_goals": len(config.get("model_goals_supported", [])),
                "min_score_threshold": config.get("minimum_score_to_pass")
            }
        }

    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "message": "Invalid JSON in config.json",
            "details": {"error": str(e)}
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Config check failed",
            "details": {"error": str(e)}
        }


async def _check_vectorizer() -> Dict[str, Any]:
    try:
        if not os.path.exists(VECTORIZER_PATH):
            return {
                "status": "error",
                "message": "TF-IDF vectorizer not found",
                "details": {"path": VECTORIZER_PATH}
            }

        vectorizer = joblib.load(VECTORIZER_PATH)
        vocab_size = len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else 0

        return {
            "status": "ok",
            "message": "Vectorizer loaded successfully",
            "details": {
                "vocabulary_size": vocab_size,
                "file_size_mb": round(os.path.getsize(VECTORIZER_PATH) / (1024*1024), 2)
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "message": "Vectorizer load failed",
            "details": {"error": str(e)}
        }

async def _check_models() -> Dict[str, Any]:
    try:
        if not os.path.exists(MODEL_DIR):
            return {
                "status": "error",
                "message": "Model directory not found",
                "details": {"path": MODEL_DIR}
            }

        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_model.pkl')]

        if not model_files:
            return {
                "status": "error",
                "message": "No trained models found",
                "details": {"directory": MODEL_DIR}
            }

        loaded_models = {}
        failed_models = []

        for model_file in model_files:
            try:
                model_path = os.path.join(MODEL_DIR, model_file)
                model = joblib.load(model_path)

                goal_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
                loaded_models[goal_name] = {
                    "file": model_file,
                    "size_mb": round(os.path.getsize(model_path) / (1024*1024), 2)
                }

            except Exception as e:
                failed_models.append({"file": model_file, "error": str(e)})

        status = "ok" if not failed_models else ("degraded" if loaded_models else "error")

        return {
            "status": status,
            "message": f"Models check completed: {len(loaded_models)} loaded, {len(failed_models)} failed",
            "details": {
                "loaded_models": loaded_models,
                "failed_models": failed_models,
                "total_models": len(model_files)
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "message": "Models check failed",
            "details": {"error": str(e)}
        }

async def _check_goals() -> Dict[str, Any]:
    try:
        if not os.path.exists(GOALS_PATH):
            return {
                "status": "error",
                "message": "goals.json not found",
                "details": {"path": GOALS_PATH}
            }

        with open(GOALS_PATH, 'r') as f:
            goals = json.load(f)

        if not isinstance(goals, dict):
            return {
                "status": "error",
                "message": "Invalid goals.json format",
                "details": {"expected": "dict", "got": type(goals).__name__}
            }

        goal_stats = {}
        for goal, skills in goals.items():
            if isinstance(skills, list):
                goal_stats[goal] = len(skills)
            else:
                goal_stats[goal] = "invalid_format"

        return {
            "status": "ok",
            "message": "Goals loaded successfully",
            "details": {
                "total_goals": len(goals),
                "goals_skills_count": goal_stats
            }
        }

    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "message": "Invalid JSON in goals.json",
            "details": {"error": str(e)}
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Goals check failed",
            "details": {"error": str(e)}
        }

async def _check_model_registry() -> Dict[str, Any]:
    try:
        if not os.path.exists(REGISTRY_PATH):
            return {
                "status": "warning",
                "message": "Model registry not found (optional)",
                "details": {"path": REGISTRY_PATH}
            }

        with open(REGISTRY_PATH, 'r') as f:
            registry = json.load(f)

        missing_files = []
        if "models" in registry:
            for goal, model_file in registry["models"].items():
                model_path = os.path.join(MODEL_DIR, model_file)
                if not os.path.exists(model_path):
                    missing_files.append(model_file)

        status = "ok" if not missing_files else "warning"

        return {
            "status": status,
            "message": "Registry check completed",
            "details": {
                "registered_models": len(registry.get("models", {})),
                "missing_files": missing_files,
                "has_metrics": "metrics" in registry
            }
        }

    except Exception as e:
        return {
            "status": "warning",
            "message": "Registry check failed",
            "details": {"error": str(e)}
        }

async def _check_performance() -> Dict[str, Any]:
    """Quick performance test with dummy data."""
    try:
        # This would normally call your scorer function
        test_text = "Python programming data structures algorithms"
        start_time = datetime.now(timezone.utc).isoformat()
        
        # Simulate scoring (replace with actual scorer call)
        # score_result = await score_resume("test", "Amazon SDE", test_text)
        
        end_time = datetime.now(timezone.utc).isoformat()
        response_time = (end_time - start_time).total_seconds()
        
        # Check if response time meets SLA (< 1.5s as per requirements)
        status = "ok" if response_time < 1.5 else "warning"
        
        return {
            "status": status,
            "message": f"Performance test completed in {response_time:.3f}s",
            "details": {
                "response_time_seconds": round(response_time, 3),
                "sla_threshold": 1.5,
                "meets_sla": response_time < 1.5
            }
        }
        
    except Exception as e:
        return {
            "status": "warning",
            "message": "Performance test failed",
            "details": {"error": str(e)}
        }

# Version endpoint
@app.get("/version")
async def version():
    """Return version and model metadata."""
    config = app.state.config
    scorer = app.state.scorer

    # Log metadata
    logger.info(" /version endpoint called")
    logger.info(f" Version: {config['version']}")
    logger.info(f" Supported goals: {len(config['model_goals_supported'])} goals")
    logger.info(f" Default goal: {config['default_goal_model']}")
    logger.info(f" Loaded models: {list(scorer.models.keys())}")
    logger.info(f" Logging enabled: {config['log_score_details']}")
    logger.info(f" Analytics enabled: {config['analytics']['collect_usage_metrics']}")
    logger.info(f" Alert on low scores: {config['notification']['alert_on_error']} if < {config['notification']['alert_threshold_score']}")
    return {
        "version": config["version"],
        "minimum_score_to_pass": config["minimum_score_to_pass"],
        "default_goal_model": config["default_goal_model"],
        "model_goals_supported": config["model_goals_supported"],
        "loaded_models": list(scorer.models.keys()),
        "skill_matching": config["skill_matching"],
        "performance": config["performance"],
        "logging": config["logging"],
        "api": config["api"],
        "analytics": config["analytics"],
        "notification": config["notification"]
    }

# Main scoring endpoint
@app.post("/score", response_model=ScoreResponse)
async def score_resume(request: ScoreRequest):
    """Score a resume against a goal and return insights."""
    config = app.state.config
    scorer = app.state.scorer

    # Normalize goal input
    input_goal = request.goal.strip().lower()

    #  Check if normalized goal is supported
    if input_goal not in config["model_goals_supported"]:
        logger.warning(f"Unsupported goal requested: {request.goal}, falling back to default")
        goal = config["default_goal_model"]
    else:
        goal = input_goal

    try:
        #  Score the resume using normalized goal
        result = scorer.score_resume(
            student_id=request.student_id,
            goal=goal,
            resume_text=request.resume_text
        )

        if config.get("log_score_details", False):
            logger.info(
                f"Scored resume for student {request.student_id}: "
                f"goal={request.goal}, score={result['score']:.2f}, "
                f"matched={len(result['matched_skills'])}, "
                f"missing={len(result['missing_skills'])}"
            )

        return result

    except Exception as e:
        logger.error(f"Error scoring resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to score resume: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)