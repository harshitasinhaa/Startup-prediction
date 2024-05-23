from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

class CustomPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, binary_model=None, model_open=None, model_closed=None, scaler=None, encoder=None, smote=None):
        self.binary_model = binary_model if binary_model else XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.model_open = model_open if model_open else MultinomialNB()
        self.model_closed = model_closed if model_closed else MultinomialNB()
        self.scaler = scaler if scaler else MinMaxScaler()
        self.encoder = encoder if encoder else LabelEncoder()
        self.smote = smote if smote else SMOTE(random_state=42)

    def fit(self, X, y):
        try:
            X_scaled = self.scaler.fit_transform(X)
            y_encoded = self.encoder.fit_transform(y)
            X_smote, y_smote = self.smote.fit_resample(X_scaled, y_encoded)

            y_binary = (y_smote > 1)
            self.binary_model.fit(X_smote, y_binary)

            open_mask = y_binary == 1
            closed_mask = y_binary == 0

            self.model_open.fit(X_smote[open_mask], y_smote[open_mask])
            self.model_closed.fit(X_smote[closed_mask], y_smote[closed_mask])

        except ValueError as e:
            raise ValueError(f"Error in fitting the model: {str(e)}")

        return self

    def predict(self, X):
        try:
            X_scaled = self.scaler.transform(X)
            binary_predictions = self.binary_model.predict(X_scaled)

            final_predictions = []
            for pred, features in zip(binary_predictions, X_scaled):
                if pred == 1:
                    final_pred = self.encoder.inverse_transform(self.model_open.predict([features]))[0]
                else:
                    final_pred = self.encoder.inverse_transform(self.model_closed.predict([features]))[0]
                final_predictions.append(final_pred)

        except ValueError as e:
            raise ValueError(f"Error in prediction: {str(e)}")

        return final_predictions
