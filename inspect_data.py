import pandas as pd
import os

files = [
    "/home/mohamed/Desktop/ai/Construction_Delay_Template 09-11-2025.xlsx",
    "/home/mohamed/Desktop/ai/from web site.csv"
]

for file_path in files:
    print(f"\n{'='*50}")
    print(f"Inspecting: {os.path.basename(file_path)}")
    print(f"{'='*50}")
    
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
            
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nFirst 3 rows:")
        print(df.head(3))
        print("\nInfo:")
        print(df.info())
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
