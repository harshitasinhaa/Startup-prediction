from joblib import load
import logging
from .custom_pipeline import CustomPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_nb_model(model_path='mlmodel/models/model_nb.joblib'):
    try:
        logging.info(f"Loading model from {model_path}")
        model = load(model_path)
        logging.info("Model loaded successfully")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def get_combined_pipeline_model():
    model_path = 'mlmodel/models/combined_pipeline.joblib'
    return load(model_path)