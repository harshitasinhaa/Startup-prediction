from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class CustomPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, binary_model_1, binary_model_2, multiclass_model, label_encoder=None):
        self.binary_model_1 = binary_model_1
        self.binary_model_2 = binary_model_2
        self.multiclass_model = multiclass_model
        self.label_encoder = label_encoder

    def fit(self, X, y_binary, y_multiclass):
        self.binary_model_1.fit(X, y_binary)
        self.binary_model_2.fit(X, y_binary)

        multiclass_indices = (self.binary_model_1.predict(X) == 0) & (self.binary_model_2.predict(X) == 0)
        self.multiclass_model.fit(X[multiclass_indices], y_multiclass[multiclass_indices])
        return self

    def predict(self, X):
      final_predictions = np.empty(X.shape[0], dtype=int)

      pred_1 = self.binary_model_1.predict(X)
      pred_2 = self.binary_model_2.predict(X)
      
      for i in range(X.shape[0]):
          if pred_1[i] == 0 or pred_2[i] == 0:
              raw_pred = self.multiclass_model.predict(X[i:i+1])

              if isinstance(raw_pred[0], str):

                  final_predictions[i] = self.label_encoder.transform([raw_pred[0]])[0]
              else:
                  final_predictions[i] = raw_pred[0]
          else:

              final_predictions[i] = self.label_encoder.transform(['operating'])[0]

      return final_predictions
