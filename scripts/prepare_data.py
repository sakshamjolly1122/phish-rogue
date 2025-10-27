#!/usr/bin/env python3
"""
Data preparation script for PHISH-ROGUE.
Copies and renames raw CSV files to processed directory.
"""
import os
import shutil
import pandas as pd
from pathlib import Path

def main():
    # Define paths
    raw_dir = Path("raw")
    processed_dir = Path("data/processed")
    
    # Create processed directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # File mappings
    file_mappings = {
        "train (1).csv": "train.csv",
        "val_unseen (1).csv": "val_unseen.csv", 
        "test (2).csv": "test.csv"
    }
    
    print("Preparing data for PHISH-ROGUE...")
    
    for source_name, target_name in file_mappings.items():
        source_path = raw_dir / source_name
        target_path = processed_dir / target_name
        
        if source_path.exists():
            # Copy file
            shutil.copy2(source_path, target_path)
            print(f"✓ Copied {source_name} → {target_name}")
            
            # Validate CSV structure
            try:
                df = pd.read_csv(target_path)
                print(f"  Rows: {len(df)}")
                
                # Check for required columns
                if 'url' in df.columns and 'label' in df.columns:
                    print(f"  ✓ Contains required columns: url, label")
                else:
                    print(f"  ⚠ Missing required columns. Available: {list(df.columns)}")
                    
            except Exception as e:
                print(f"  ⚠ Error reading CSV: {e}")
        else:
            print(f"✗ Source file not found: {source_name}")
    
    print("\nData preparation completed!")
    print(f"Processed files are in: {processed_dir.absolute()}")

if __name__ == "__main__":
    main()
