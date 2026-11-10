import joblib
import pandas as pd
import numpy as np

# Load the saved model
print("Loading model from construction_model.pkl...")
model = joblib.load('construction_model.pkl')

print("Model loaded successfully!")

# Define the exact features expected by the model in order
features = [
    'Expected Total Cost', 'Expected Duration', 'Project Size (Area)', 
    'Complexity', 'Design Completeness', 'Sustainability Requirements', 
    'Material Price Index', 'Local Material Availability', 'Average Labor Cost', 
    'Expected Material Cost', 'Political Stability', 'BIM Accuracy', 
    'Value Engineering', 'Manager Experience', 
    'Type_Commercial ', 'Type_commercial', 'Type_service', 
    'Contract_Unit price '
]

# Example Data: A large Commercial project
sample_data = {
    'Expected Total Cost': [4000000],
    'Expected Duration': [180], 
    'Project Size (Area)': [1500],
    'Complexity': [2],
    'Design Completeness': [90], 
    'Sustainability Requirements': [1], 
    'Material Price Index': [120],
    'Local Material Availability': [0.8],
    'Average Labor Cost': [5000],
    'Expected Material Cost': [2000000],
    'Political Stability': [3], 
    'BIM Accuracy': [90],
    'Value Engineering': [1],
    'Manager Experience': [7], 
    
    'Type_Commercial ': [1], 
    'Type_commercial': [0], 
    'Type_service': [0],
    
    'Contract_Unit price ': [0] 
}

df_sample = pd.DataFrame(sample_data)

# Reorder to match model expectation exactly
df_sample = df_sample[features]

print("\nPredicting for New Project:")
print(f"Expected Cost: ${sample_data['Expected Total Cost'][0]:,}")
print(f"Expected Duration: {sample_data['Expected Duration'][0]} Days")

# Predict (Returns Ratios)
pred_ratios = model.predict(df_sample)
cost_ratio = pred_ratios[0][0]
duration_ratio = pred_ratios[0][1]

print(f"\nModel Output (Ratios):")
print(f"Cost Overrun Ratio: {cost_ratio:.4f}")
print(f"Duration Overrun Ratio: {duration_ratio:.4f}")

# Calculate Absolute
predicted_cost = cost_ratio * sample_data['Expected Total Cost'][0]
predicted_duration = duration_ratio * sample_data['Expected Duration'][0]

print("\n" + "="*40)
print(f"FINAL RESULTS:")
print(f"Predicted Actual Cost: ${predicted_cost:,.2f}")
print(f"Predicted Duration:    {predicted_duration:.2f} Days")
print("="*40)
