"""
Batch processing script to preprocess all patient CSV files
Run this after you have all patient CSVs in data/raw/
"""

import sys
from pathlib import Path

# Add paths
sys.path.append('src/data')
sys.path.append('configs')

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from preprocess_dreamt import DREAMTPreprocessor
import numpy as np


def process_all_patients(use_simplified_stages=True):
    """
    Process all patient CSV files in data/raw/
    
    Args:
        use_simplified_stages: If True, use 3-class (W/NREM/REM)
    """
    
    # Find all CSV files
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    
    if len(csv_files) == 0:
        print("ERROR: No CSV files found in data/raw/")
        print("Please place your patient CSV files there first.")
        return
    
    print(f"{'='*70}")
    print(f"BATCH PROCESSING: Found {len(csv_files)} patient file(s)")
    print(f"{'='*70}\n")
    
    # Initialize preprocessor
    preprocessor = DREAMTPreprocessor(use_simplified_stages=use_simplified_stages)
    
    # Track results
    results = []
    failed = []
    
    # Process each patient
    for i, csv_path in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Processing: {csv_path.name}")
        print("-" * 70)
        
        try:
            # Process patient
            save_path = PROCESSED_DATA_DIR / f"{csv_path.stem}_processed.npz"
            result = preprocessor.process_patient(str(csv_path), str(save_path))
            results.append(result)
            
            print(f"✅ SUCCESS: {csv_path.stem}")
            print(f"   Windows: {result['n_windows']}")
            print(f"   Shape: {result['X'].shape}")
            
        except Exception as e:
            print(f"❌ FAILED: {csv_path.stem}")
            print(f"   Error: {str(e)}")
            failed.append((csv_path.name, str(e)))
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Successfully processed: {len(results)}/{len(csv_files)} patients")
    
    if failed:
        print(f"❌ Failed: {len(failed)} patients")
        for name, error in failed:
            print(f"   - {name}: {error}")
    
    # Overall statistics
    if results:
        print(f"\nDataset Statistics:")
        total_windows = sum(r['n_windows'] for r in results)
        print(f"  Total windows: {total_windows:,}")
        print(f"  Average windows per patient: {total_windows/len(results):.1f}")
        
        # Label distribution across all patients
        all_labels = np.concatenate([r['y'].flatten() for r in results])
        unique, counts = np.unique(all_labels, return_counts=True)
        
        stage_names = {0: 'Wake', 1: 'NREM', 2: 'REM'}
        print(f"\n  Overall label distribution:")
        for u, c in zip(unique, counts):
            print(f"    {stage_names[u]}: {c:,} ({c/len(all_labels)*100:.1f}%)")
        
        print(f"\n📁 Processed files saved to: {PROCESSED_DATA_DIR}")
    
    return results, failed


def load_processed_patient(patient_id):
    """
    Load a processed patient file
    
    Args:
        patient_id: Patient identifier (e.g., 'S005_whole_df')
        
    Returns:
        Dictionary with X, y, and metadata
    """
    file_path = PROCESSED_DATA_DIR / f"{patient_id}_processed.npz"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Processed file not found: {file_path}")
    
    data = np.load(file_path, allow_pickle=True)
    
    return {
        'X': data['X'],
        'y': data['y'],
        'patient_id': str(data['patient_id']),
        'normalization_params': data['normalization_params'].item(),
        'n_windows': int(data['n_windows']),
        'window_size': int(data['window_size']),
        'sequence_length': int(data['sequence_length']),
        'n_classes': int(data['n_classes'])
    }


def load_all_processed_patients():
    """
    Load all processed patient files
    
    Returns:
        List of dictionaries with patient data
    """
    processed_files = list(PROCESSED_DATA_DIR.glob("*_processed.npz"))
    
    if len(processed_files) == 0:
        print("No processed files found. Run process_all_patients() first.")
        return []
    
    print(f"Loading {len(processed_files)} processed patient file(s)...")
    
    patients = []
    for file_path in processed_files:
        patient_id = file_path.stem.replace('_processed', '')
        data = load_processed_patient(patient_id)
        patients.append(data)
        print(f"  ✅ Loaded {patient_id}: {data['X'].shape}")
    
    return patients


if __name__ == "__main__":
    # Run batch processing
    results, failed = process_all_patients(use_simplified_stages=True)
    
    print("\n" + "="*70)
    print("READY FOR TRAINING!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Create train/val/test splits")
    print("  2. Create DataLoader")
    print("  3. Train WatchSleepNet model")
    print("\nTo load processed data in another script:")
    print("  from process_all_patients import load_all_processed_patients")
    print("  patients = load_all_processed_patients()")