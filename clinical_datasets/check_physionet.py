import os, pandas as pd

data_dir = r"E:\rul_project\clinical_datasets\physionet2012\set-a"
out_file = r"E:\rul_project\clinical_datasets\physionet2012\Outcomes-a.txt"

print("=== OUTCOMES FILE ===")
outcomes = pd.read_csv(out_file)
outcomes.columns = [c.strip() for c in outcomes.columns]
print(f"Columns: {list(outcomes.columns)}")
print(outcomes.head(3).to_string())

files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
print(f"\n=== SAMPLE FILE: {files[0]} ===")
df = pd.read_csv(os.path.join(data_dir, files[0]))
print(f"Columns: {list(df.columns)}")
print(df.head(15).to_string())