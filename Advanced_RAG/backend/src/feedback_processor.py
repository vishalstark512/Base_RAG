import json
from typing import Dict, Any
from src.utils import setup_logging

logger = setup_logging()

def store_feedback(feedback: Dict[str, Any]):
    # Store feedback in a database or file
    # This is a placeholder implementation
    with open('feedback.json', 'a') as f:
        json.dump(feedback, f)
        f.write('\n')
    logger.info(f"Stored feedback for task {feedback['task_id']}")

def improve_system(feedback: Dict[str, Any]):
    # Use feedback to improve the system
    # This could involve adjusting model parameters, updating training data, etc.
    # This is a placeholder implementation
    if feedback['rating'] < 3:
        logger.info(f"Low rating received for task {feedback['task_id']}. Flagging for review.")
    else:
        logger.info(f"Positive feedback received for task {feedback['task_id']}")

def process_feedback(feedback: Dict[str, Any]):
    try:
        # Store feedback
        store_feedback(feedback)
        
        # Use feedback to improve the system
        improve_system(feedback)
        
        logger.info(f"Processed feedback for task {feedback['task_id']}")
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")

if __name__ == "__main__":
    # Test feedback processing
    test_feedback = {
        "task_id": "test_task_123",
        "rating": 4,
        "comments": "Good summary, but could be more concise."
    }
    process_feedback(test_feedback)