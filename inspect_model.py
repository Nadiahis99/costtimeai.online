import joblib
import numpy as np

model = joblib.load('construction_model.pkl')
print(f"N Features: {model.n_features_in_}")
if hasattr(model, 'feature_names_in_'):
    print("Features:", list(model.feature_names_in_))
else:
    print("Feature names not saved in model object (older sklearn version?).")
