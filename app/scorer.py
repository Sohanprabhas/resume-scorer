import os
import re
import json
import logging
import joblib
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("resume-scorer")

class ResumeScorer:
    """
    Main class for scoring resumes against various goals using ML models
    and skill-based analysis.
    """
    
    def __init__(self, model_dir,goals_path,config: Dict[str, Any], goals: Dict[str, List[str]]):
        """
        Initialize the ResumeScorer with configuration and goals data.
        
        Args:
            config: Configuration dictionary loaded from config.json
            goals: Dictionary of goals and their required skills
        """
        self.model_dir = model_dir
        self.config = config
        self.goals = goals
        self.models = {}
        self.shared_vectorizer = None
        self.goals_data = {}
        
        # Load models and vectorizers for each supported goal
        self._load_models()
        self._load_vectorizer()
        self._load_goals(goals_path)
        
        logger.info(f"ResumeScorer initialized with {len(self.models)} models")
    
    def _load_models(self) -> None:
        """Load shared TF-IDF vectorizer and goal-specific models."""
        model_dir = os.path.join(os.path.dirname(__file__), "model")

        # Load shared vectorizer
        try:
            vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
            self.shared_vectorizer = joblib.load(vectorizer_path)
            logger.info(f" Shared TF-IDF vectorizer loaded with {len(self.shared_vectorizer.vocabulary_)} features")
        except Exception as e:
            logger.error(f" Failed to load shared TF-IDF vectorizer: {str(e)}")
            self.shared_vectorizer = None

        # Load goal-specific models
        for goal in self.config["model_goals_supported"]:
            try:
                goal_filename = goal.lower().replace(" ", "_")
                model_path = os.path.join(model_dir, f"{goal_filename}_model.pkl")
                self.models[goal] = joblib.load(model_path)
                logger.info(f" Loaded model for goal: {goal}")
            except Exception as e:
                logger.error(f" Failed to load model for goal {goal}: {str(e)}")

    def _load_vectorizer(self):
        path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
        self.vectorizer = joblib.load(path)
    def _load_goals(self, goals_path):
        with open(goals_path, "r") as f:
            self.goals_data = json.load(f)

    def _extract_skills_from_resume(self, resume_text: str) -> List[str]:
        """
        Extract skills from the resume text by checking for each skill in the goals data.
        Uses pattern matching to find skills in the resume.
        
        Args:
            resume_text: The full text of the resume
            
        Returns:
            List of skills found in the resume
        """
        # Create a set of all unique skills across all goals
        all_skills = set()
        for skills in self.goals.values():
            all_skills.update(skills)
        
        # Find skills in the resume text
        found_skills = []
        resume_text_lower = resume_text.lower()
        
        for skill in all_skills:
            # Create a regex pattern that looks for the skill as a whole word
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, resume_text_lower):
                found_skills.append(skill)
                
        return found_skills
    
    def _get_matched_missing_skills(self, found_skills: List[str], goal: str) -> tuple:
        """
        Compare found skills with required skills for the goal.
        
        Args:
            found_skills: List of skills found in the resume
            goal: The target goal
            
        Returns:
            Tuple of (matched_skills, missing_skills)
        """
        if goal not in self.goals:
            logger.warning(f"Goal '{goal}' not found in goals data, using default")
            goal = self.config["default_goal_model"]
            
        required_skills = self.goals.get(goal, [])
        
        # Find matched and missing skills
        matched_skills = [skill for skill in found_skills if skill in required_skills]
        missing_skills = [skill for skill in required_skills if skill not in found_skills]
        
        return matched_skills, missing_skills
    
    def _generate_learning_path(self, missing_skills: List[str]) -> List[str]:
        """
        Generate a personalized learning path based on missing skills.
        
        Args:
            missing_skills: List of skills missing from the resume
            
        Returns:
            List of suggested learning activities
        """
        learning_path = []
        
        # Skill-specific learning recommendations
        skill_recommendations = {
            "Java": "Complete a Java programming course focusing on core concepts and OOP principles",
            "Python": "Learn Python fundamentals and practice with data structure exercises",
            "C++": "Take a C++ course covering memory management and STL",
            "Data Structures": "Study and implement common data structures like arrays, linked lists, trees and graphs",
            "Algorithms": "Practice algorithmic problem solving on platforms like LeetCode or HackerRank",
            "SQL": "Learn database design principles and practice complex SQL queries",
            "REST APIs": "Build a small project that consumes and creates REST APIs",
            "System Design": "Learn basic system design concepts and architecture patterns",
            "Docker": "Create and deploy containerized applications using Docker",
            "AWS": "Complete AWS Cloud Practitioner certification",
            "Azure": "Learn Azure fundamentals and deploy a small application",
            "Kubernetes": "Learn container orchestration with Kubernetes",
            "React": "Build a frontend application using React and modern JS",
            "JavaScript": "Master JavaScript fundamentals and ES6+ features",
            "TypeScript": "Learn TypeScript for adding static types to JavaScript",
            "Machine Learning": "Take an introductory ML course covering supervised and unsupervised learning",
            "Deep Learning": "Study neural networks and implement basic models with TensorFlow or PyTorch",
            "Numpy": "Practice numerical computing with NumPy",
            "Pandas": "Learn data manipulation and analysis with Pandas",
            "Scikit-learn": "Implement ML algorithms using scikit-learn",
            "TensorFlow": "Build and train neural networks with TensorFlow",
            "PyTorch": "Learn deep learning with PyTorch framework",
            "Linear Algebra": "Study vectors, matrices and linear transformations",
            "Statistics": "Learn probability theory and statistical methods",
            "Git": "Master version control with Git and GitHub",
            "CI/CD": "Set up continuous integration and deployment pipelines",
            "Agile": "Learn agile methodologies and scrum practices",
            "Communication": "Improve technical writing and presentation skills",
            "Team Work": "Practice collaborative development through open source contributions"
        }
        
        # Generate recommendations for each missing skill
        for skill in missing_skills:
            if skill in skill_recommendations:
                learning_path.append(skill_recommendations[skill])
            else:
                # Generic recommendation if specific one isn't available
                learning_path.append(f"Develop proficiency in {skill} through online courses and projects")
                
        # Add general advice if there are missing skills
        if missing_skills:
            learning_path.append("Create portfolio projects that showcase your skills in these areas")
            
        return learning_path
    
    def score_resume(self, student_id: str, goal: str, resume_text: str) -> Dict[str, Any]:
     logger.info(f"Processing resume for student_id={student_id}, goal='{goal}'")


     if not resume_text.strip():
      normalized_goal = goal.strip().casefold()
      goals_map = {k.casefold(): k for k in self.goals}
      actual_goal_key = goals_map.get(normalized_goal, self.config["default_goal_model"])
    
      required_skills = self.goals.get(actual_goal_key, [])
      return {
        "score": 0.0,
        "matched_skills": [],
        "missing_skills": required_skills,
        "suggested_learning_path": self._generate_learning_path(required_skills)
      }

    
    # Normalize goal to lowercase for uniformity
     normalized_goal = goal.strip().casefold()

    # Create case-insensitive maps
     goals_map = {k.casefold(): k for k in self.goals}
     models_map = {k.casefold(): k for k in self.models}
     supported_map = {k.casefold(): k for k in self.config["model_goals_supported"]}

    # Check if goal is supported
     if normalized_goal not in supported_map:
        logger.warning(f"Goal '{goal}' not supported, using default: {self.config['default_goal_model']}")
        normalized_goal = self.config["default_goal_model"].casefold()

    # Check if goal exists in goals data
     if normalized_goal not in goals_map:
        logger.warning(f"Goal '{goal}' not found in goals data, using default")
        normalized_goal = self.config["default_goal_model"].casefold()

     actual_goal_key = goals_map.get(normalized_goal, self.config["default_goal_model"])

    # Extract skills
     found_skills = self._extract_skills_from_resume(resume_text)
     matched_skills, missing_skills = self._get_matched_missing_skills(found_skills, actual_goal_key)

    # Score
     if normalized_goal in models_map and self.shared_vectorizer:
        X = self.shared_vectorizer.transform([resume_text])
        score = float(self.models[models_map[normalized_goal]].predict_proba(X)[0, 1])
     else:
        required_skills = self.goals.get(actual_goal_key, [])
        score = len(matched_skills) / max(len(required_skills), 1) if required_skills else 0.0

    # Learning path
     learning_path = self._generate_learning_path(missing_skills)

     return {
        "score": score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "suggested_learning_path": learning_path
    }





    def evaluate_passing(self, score: float) -> bool:
        """
        Determine if a score meets the minimum passing threshold.
        
        Args:
            score: The resume match score
            
passes the thresholdpip         Returns:
            Boolean indicating if the score install joblib
        """
        return score >= self.config["minimum_score_to_pass"]