#!/usr/bin/env python3
"""
Download Model Files for HealthAI Suite

This script downloads all required pre-trained models from Google Drive.
Usage: python models/download_models.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Google Drive file IDs and target filenames
MODELS = {
    # Add your Google Drive file IDs here
    # Format: 'filename.ext': 'GOOGLE_DRIVE_FILE_ID'
    'los_model.pkl': 'YOUR_GOOGLE_DRIVE_ID_1',
    'los_scaler.pkl': 'YOUR_GOOGLE_DRIVE_ID_2',
    'kmeans_cluster_model.pkl': 'YOUR_GOOGLE_DRIVE_ID_3',
    'cluster_scaler_final.pkl': 'YOUR_GOOGLE_DRIVE_ID_4',
    'xgboost_disease_model.json': 'YOUR_GOOGLE_DRIVE_ID_5',
    'association_rules.json': 'YOUR_GOOGLE_DRIVE_ID_6',
    'pneumonia_cnn_model.h5': 'YOUR_GOOGLE_DRIVE_ID_7',
}

def download_from_google_drive(file_id, output_path):
    """
    Download file from Google Drive using gdown.
    
    Args:
        file_id: Google Drive file ID
        output_path: Local path to save the file
    """
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])
        import gdown
    
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f"Downloading {output_path}...")
    gdown.download(url, str(output_path), quiet=False)
    print(f"✓ Downloaded {output_path}")

def main():
    """
    Download all model files.
    """
    models_dir = Path(__file__).parent
    
    print("="*60)
    print("HealthAI Suite - Model Downloader")
    print("="*60)
    print(f"\nModels directory: {models_dir}")
    print(f"Files to download: {len(MODELS)}\n")
    
    # Check if gdown is available
    try:
        import gdown
        print("✓ gdown is installed")
    except ImportError:
        print("Installing gdown (required for Google Drive downloads)...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])
    
    failed_downloads = []
    
    # Download each model
    for filename, file_id in MODELS.items():
        if file_id == 'YOUR_GOOGLE_DRIVE_ID_1' or file_id.startswith('YOUR_'):
            print(f"⚠ Skipping {filename} - File ID not configured")
            failed_downloads.append(filename)
            continue
        
        output_path = models_dir / filename
        
        # Skip if already exists
        if output_path.exists():
            print(f"✓ {filename} already exists")
            continue
        
        try:
            download_from_google_drive(file_id, output_path)
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            failed_downloads.append(filename)
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    
    if failed_downloads:
        print(f"\n⚠ {len(failed_downloads)} file(s) not downloaded:")
        for filename in failed_downloads:
            print(f"  - {filename}")
        print("\nPlease configure Google Drive file IDs in this script.")
        return 1
    else:
        print("\n✓ All models downloaded successfully!")
        return 0

if __name__ == '__main__':
    sys.exit(main())
