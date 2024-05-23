from django.apps import AppConfig

class MLModelConfig(AppConfig):
    name = 'mlmodel'
    model_nb = None  # Naive Bayes model

    def ready(self):
        if not self.model_nb:  # Load the model only if it hasn't been loaded already
            from .load_models import get_nb_model
            try:
                self.model_nb = get_nb_model()
            except Exception as e:
                # Handle exceptions, possibly logging them
                print(f"Failed to load the model: {str(e)}")
