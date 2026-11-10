import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

class ConstructionModel:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.target_names = ["Cost Overrun Ratio", "Duration Overrun Ratio"]
        
    def build_model(self):
        self.model = RandomForestRegressor(
            n_estimators=100, 
            min_samples_leaf=10,
            random_state=42, 
            n_jobs=-1
        )
        
    def train(self, X_train, y_train):
        print(f"Training Multi-Output Model (Ratios)...")
        self.model.fit(X_train, y_train)
        
    def save(self, filepath):
        print(f"Saving model to {filepath}...")
        joblib.dump(self.model, filepath)
        joblib.dump(self.model, filepath, compress=3)
    def load(self, filepath):
        print(f"Loading model from {filepath}...")
        self.model = joblib.load(filepath)
        
    def evaluate(self, X_test, y_test_ratios, original_X_test):
        """
        X_test: scaled/encoded features used for prediction
        y_test_ratios: actual ratios (Ground Truth)
        original_X_test: DataFrame containing the 'Expected Total Cost' and 'Expected Duration' columns to reconstruction.
        """
        print(f"\nEvaluating Ratio-Based Model (Reconstructed Values):")
        pred_ratios = self.model.predict(X_test)
        
        # Extract Expected Values for reconstruction
        expected_cost = original_X_test['Expected Total Cost'].reset_index(drop=True)
        expected_duration = original_X_test['Expected Duration'].reset_index(drop=True)
        
        if hasattr(y_test_ratios, 'iloc'):
            actual_cost_ratio = y_test_ratios.iloc[:, 0].reset_index(drop=True)
            actual_dur_ratio = y_test_ratios.iloc[:, 1].reset_index(drop=True)
        else:
            actual_cost_ratio = y_test_ratios[:, 0]
            actual_dur_ratio = y_test_ratios[:, 1]
            
        true_actual_cost = actual_cost_ratio * expected_cost
        true_actual_dur = actual_dur_ratio * expected_duration
        
        # Predicted Absolute Values
        pred_cost_ratio = pred_ratios[:, 0]
        pred_dur_ratio = pred_ratios[:, 1]
        
        pred_actual_cost = pred_cost_ratio * expected_cost
        pred_actual_dur = pred_dur_ratio * expected_duration
        
        # Calculate Metrics on ABSOLUTE values (User cares about Money/Days, not Ratios)
        metrics = {}
        
        # Cost Metrics
        mae_cost = mean_absolute_error(true_actual_cost, pred_actual_cost)
        rmse_cost = np.sqrt(mean_squared_error(true_actual_cost, pred_actual_cost))
        r2_cost = r2_score(true_actual_cost, pred_actual_cost)
        
        print(f"\n--- Actual Cost Prediction (Derived) ---")
        print(f"MAE: {mae_cost:,.2f}")
        print(f"RMSE: {rmse_cost:,.2f}")
        print(f"R²: {r2_cost:.4f}")
        
        # Duration Metrics
        mae_dur = mean_absolute_error(true_actual_dur, pred_actual_dur)
        rmse_dur = np.sqrt(mean_squared_error(true_actual_dur, pred_actual_dur))
        r2_dur = r2_score(true_actual_dur, pred_actual_dur)
        
        print(f"\n--- Actual Duration Prediction (Derived) ---")
        print(f"MAE: {mae_dur:,.2f}")
        print(f"RMSE: {rmse_dur:,.2f}")
        print(f"R²: {r2_dur:.4f}")
        
        return metrics
        
    def predict(self, X):
        return self.model.predict(X)

def sensitivity_analysis(model, X_test, feature_names):
    print("\nRunning Sensitivity Analysis (on Reconstructed output)...")
    
    mean_row = X_test.mean().to_frame().T
    
    # Base Prediction (Ratio)
    base_pred_ratios = model.predict(mean_row)[0] # [CostRatio, DurRatio]
    

    exp_cost_idx = feature_names.index('Expected Total Cost') if 'Expected Total Cost' in feature_names else -1
    exp_dur_idx = feature_names.index('Expected Duration') if 'Expected Duration' in feature_names else -1
    
    base_exp_cost = mean_row.iloc[0, exp_cost_idx]
    base_exp_dur = mean_row.iloc[0, exp_dur_idx]
    
    base_abs_cost = base_pred_ratios[0] * base_exp_cost
    base_abs_dur = base_pred_ratios[1] * base_exp_dur
    
    print(f"Base Abs Prediction -> Cost: {base_abs_cost:,.0f}, Duration: {base_abs_dur:.1f}")

    for feature in feature_names:
        if feature in mean_row.columns:
            perturbed_row = mean_row.copy()
            
            # Perturb logic
            val = perturbed_row[feature].values[0]
            if val == 0 or val == 1:
                 perturbed_row[feature] = 1 - val
            else:
                 perturbed_row[feature] = val * 1.10
            
            # Predict new Ratios
            new_pred_ratios = model.predict(perturbed_row)[0]
            
            # Reconstruct using PERTURBED Expected values (if we perturbed Expectation)
            new_exp_cost = perturbed_row.iloc[0, exp_cost_idx]
            new_exp_dur = perturbed_row.iloc[0, exp_dur_idx]
            
            new_abs_cost = new_pred_ratios[0] * new_exp_cost
            new_abs_dur = new_pred_ratios[1] * new_exp_dur
            
            cost_change = ((new_abs_cost - base_abs_cost) / base_abs_cost) * 100
            dur_change = ((new_abs_dur - base_abs_dur) / base_abs_dur) * 100
            
            if abs(cost_change) > 0.01 or abs(dur_change) > 0.01:
                print(f"Sensitivity {feature:30}: Cost {cost_change:+.2f}% | Duration {dur_change:+.2f}%")
