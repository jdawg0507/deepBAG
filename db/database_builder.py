import os
import sqlite3
import pandas as pd
import zipfile
import glob

DB_PATH = os.path.join(os.path.dirname(__file__), "historical_nba.db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ZIP_FILE = os.path.join(DATA_DIR, "historical-nba-data-and-player-box-scores.zip")

def build_database():
    if not os.path.exists(ZIP_FILE):
        print(f"Error: {ZIP_FILE} not found. Please download the dataset first.")
        return

    print("Extracting ZIP file...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
        
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Found {len(csv_files)} CSV files.")
    
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    
    for csv_file in csv_files:
        table_name = os.path.basename(csv_file).replace(".csv", "").lower()
        print(f"Ingesting {csv_file} into table '{table_name}'...")
        
        try:
            # Use chunksize to avoid blowing up memory with 1GB dataset
            chunksize = 100000
            for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunksize, low_memory=False)):
                chunk.to_sql(table_name, conn, if_exists='replace' if i == 0 else 'append', index=False)
            print(f"  Successfully ingested {table_name}.")
        except Exception as e:
            print(f"  Error ingesting {csv_file}: {e}")
            
    conn.close()
    print("Database build complete!")

if __name__ == "__main__":
    build_database()
