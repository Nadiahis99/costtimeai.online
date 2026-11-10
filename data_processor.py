import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import random


TARGET_COST_RATIO = 'Cost Overrun Ratio'
TARGET_DURATION_RATIO = 'Duration Overrun Ratio'

# Original absolute targets 
RAW_TARGET_COST = 'Actual Total Cost'
RAW_TARGET_DURATION = 'Actual Duration'

# Input Features (16)
INPUT_FEATURES = [
    'Project Type', 'Expected Total Cost', 'Expected Duration', 'Project Size (Area)',
    'Complexity', 'Design Completeness', 'Sustainability Requirements', 'Contract Type',
    'Material Price Index', 'Local Material Availability', 'Average Labor Cost',
    'Expected Material Cost', 'Political Stability', 'BIM Accuracy', 'Value Engineering',
    'Manager Experience'
]

class DataIngestion:
    def __init__(self, excel_path, csv_path):
        self.excel_path = excel_path
        self.csv_path = csv_path
        
    def load_data(self):
        print("Loading data files...")
        try:
            df_excel = pd.read_excel(self.excel_path)
            print(f"Excel loaded: {df_excel.shape}")
        except Exception as e:
            print(f"Error loading Excel: {e}")
            df_excel = pd.DataFrame()
            
        try:
            df_csv = pd.read_csv(self.csv_path)
            print(f"CSV loaded: {df_csv.shape}")
        except Exception as e:
            print(f"Warning loading CSV: {e}")
            df_csv = pd.DataFrame()
            
        return df_excel, df_csv

class DataCleaner:
    def normalize_columns(self, df_excel, df_csv):
        print("Normalizing and mapping columns...")
        
        # 1. Process CSV
        csv_map = {
            'expected_ total_cost': 'Expected Total Cost',
            'planned_ duration': 'Expected Duration',
            'project_size': 'Project Size (Area)',
            'actual_cost': 'Actual Total Cost',
            'actual_ duration': 'Actual Duration',
            'monthly_ labor_cost': 'Average Labor Cost',
            'material_ cost': 'Expected Material Cost',
            'manager_ experience': 'Manager Experience'
        }
        
        processed_csv = df_csv.rename(columns=csv_map).copy()
        
        for col in INPUT_FEATURES:
            if col not in processed_csv.columns:
                processed_csv[col] = np.nan
        
        # 2. Process Excel
        excel_map = {
            'Project Type': 'Project Type',
            'Planned  Budget\nالميزانية المخططة': 'Expected Total Cost',
            'Planned Duration (days)': 'Expected Duration',
            'Project Size (Area)': 'Project Size (Area)',
            'Complexity (1-5)': 'Complexity',
            'Design Completeness (%)': 'Design Completeness',
            'Sustainability Requirements (Yes/No)': 'Sustainability Requirements',
            'Contract Type': 'Contract Type',
            'Material Price Index': 'Material Price Index',
            'Local Material Availability (%)': 'Local Material Availability',
            'Average Labor Cost': 'Average Labor Cost',
            'Political/Economic Stability (1-5)': 'Political Stability',
            'BIM/QTO Accuracy (%)': 'BIM Accuracy',
            'Value Engineering (Yes/No)': 'Value Engineering',
            'Actual Budget\nالميزانية الفعلية': 'Actual Total Cost',
            'Actual Duration (days)': 'Actual Duration'
        }
        
        processed_excel = df_excel.rename(columns=excel_map).copy()
        
        combined_df = pd.concat([processed_csv, processed_excel], ignore_index=True)
        
        cols_to_keep = INPUT_FEATURES + [RAW_TARGET_COST, RAW_TARGET_DURATION]
        for col in cols_to_keep:
            if col not in combined_df.columns:
                combined_df[col] = np.nan
        
        # Clean / Impute
        print("Cleaning data...")
        for col in INPUT_FEATURES:
            if combined_df[col].dtype == 'object':
                mode_val = combined_df[col].mode()[0] if not combined_df[col].mode().empty else 'Unknown'
                combined_df[col] = combined_df[col].fillna(mode_val)
            else:
                median_val = combined_df[col].median()
                if pd.isna(median_val): median_val = 0
                combined_df[col] = combined_df[col].fillna(median_val)

        # STRICTLY RETURN ONLY KEPT COLUMNS
        return combined_df[cols_to_keep]

    def clean(self, df):
        return df

class DataAugmentor:
    def __init__(self, n_samples=5000):
        self.n_samples = n_samples

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.one_hot_encoder = None
        
    def prepare_features(self, df):
        print("Feature Engineering...")
        df = df.copy()
        
        # Calculate Ratios 
        epsilon = 1e-6
        df['Expected Total Cost'] = df['Expected Total Cost'].replace(0, epsilon)
        df['Expected Duration'] = df['Expected Duration'].replace(0, epsilon)
        
        if RAW_TARGET_COST in df.columns:
            df[TARGET_COST_RATIO] = df[RAW_TARGET_COST] / df['Expected Total Cost']
            
        if RAW_TARGET_DURATION in df.columns:
            df[TARGET_DURATION_RATIO] = df[RAW_TARGET_DURATION] / df['Expected Duration']
        
        # CRITICAL FIX: Drop Bad Targets BEFORE Filling
        cols_to_check = [c for c in [TARGET_COST_RATIO, TARGET_DURATION_RATIO] if c in df.columns]
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=cols_to_check)
        
        yes_no_cols = ['Sustainability Requirements', 'Value Engineering']
        for col in yes_no_cols:
            if col in df.columns:
               df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0)
        
        if 'Project Type' in df.columns:
            dummies = pd.get_dummies(df['Project Type'], prefix='Type', dtype=int)
            df = pd.concat([df, dummies], axis=1)
            df.drop('Project Type', axis=1, inplace=True)
            
        if 'Contract Type' in df.columns:
            dummies = pd.get_dummies(df['Contract Type'], prefix='Contract', dtype=int)
            df = pd.concat([df, dummies], axis=1)
            df.drop('Contract Type', axis=1, inplace=True)
            
        # FOOLPROOF: Keep only numeric columns
        print("Filtering for numeric columns only...")
        df = df.select_dtypes(include=[np.number])

        return df.fillna(0)
