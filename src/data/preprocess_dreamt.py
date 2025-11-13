"""
Data preprocessing pipeline for DREAMT dataset
Handles multi-rate signals, windowing, and label alignment
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

from config import (
    SLEEP_STAGES, SLEEP_STAGES_SIMPLIFIED, 
    EPOCH_LENGTH, WINDOW_SIZE, OVERLAP,
    FEATURES_TO_USE, FILTER_PREP_STAGE, FILTER_MISSING,
    SAMPLING_RATES, N_CLASSES
)


class DREAMTPreprocessor:
    """
    Preprocessor for DREAMT dataset
    Handles loading, cleaning, windowing, and preparing data for model
    """
    
    def __init__(self, use_simplified_stages=True):
        """
        Args:
            use_simplified_stages: If True, use 3-class (W/NREM/REM)
                                  If False, use 5-class (W/N1/N2/N3/R)
        """
        self.use_simplified_stages = use_simplified_stages
        self.stage_mapping = SLEEP_STAGES_SIMPLIFIED if use_simplified_stages else SLEEP_STAGES
        
    def load_patient_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load a single patient's CSV file
        
        Args:
            csv_path: Path to patient CSV file
            
        Returns:
            DataFrame with all signals and labels
        """
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data:
        - Remove preparation stage ('P')
        - Remove missing labels
        - Handle any NaN values
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        print("\nCleaning data...")
        initial_rows = len(df)
        
        # Filter out preparation stage
        if FILTER_PREP_STAGE:
            df = df[df['Sleep_Stage'] != 'P'].copy()
            print(f"Removed {initial_rows - len(df)} preparation stage rows")
        
        # Filter out missing labels
        if FILTER_MISSING:
            before = len(df)
            df = df[df['Sleep_Stage'] != 'Missing'].copy()
            removed = before - len(df)
            if removed > 0:
                print(f"Removed {removed} missing label rows")
        
        # Drop rows with NaN in critical columns
        critical_cols = FEATURES_TO_USE + ['Sleep_Stage']
        before = len(df)
        df = df.dropna(subset=critical_cols).copy()
        removed = before - len(df)
        if removed > 0:
            print(f"Removed {removed} rows with NaN values")
        
        print(f"Final cleaned data: {len(df)} rows")
        return df
    
    def align_labels_to_epochs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sleep stage labels are annotated every 30 seconds
        This function creates 30-second epochs and assigns labels
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            DataFrame with epoch labels
        """
        print("\nAligning labels to epochs...")
        
        # Get the timestamp in seconds
        df['timestamp_sec'] = df['TIMESTAMP']
        
        # Create epoch index (each epoch is 30 seconds)
        df['epoch'] = (df['timestamp_sec'] // EPOCH_LENGTH).astype(int)
        
        # For each epoch, assign the most common sleep stage
        # (Should be consistent within epoch, but this handles edge cases)
        df['epoch_label'] = df.groupby('epoch')['Sleep_Stage'].transform(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        )
        
        print(f"Created {df['epoch'].nunique()} epochs")
        
        return df
    
    def map_sleep_stages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map sleep stage strings to integer labels
        
        Args:
            df: DataFrame with 'epoch_label' column
            
        Returns:
            DataFrame with numeric 'label' column
        """
        print("\nMapping sleep stages to numeric labels...")
        
        if self.use_simplified_stages:
            df['label'] = df['epoch_label'].map(self.stage_mapping)
        else:
            df['label'] = df['epoch_label'].map(SLEEP_STAGES)
            # Remove unmapped stages (P, Missing which are -1, -2)
            df = df[df['label'] >= 0].copy()
        
        # Print label distribution
        print("\nLabel distribution:")
        label_counts = df.groupby(['epoch_label', 'label']).size()
        for (stage, label), count in label_counts.items():
            print(f"  {stage} (label {label}): {count} samples")
        
        return df
    
    def create_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Create overlapping windows for sequence modeling
        
        Each window contains WINDOW_SIZE epochs (e.g., 20 epochs = 10 minutes)
        Windows overlap by OVERLAP fraction
        
        Args:
            df: DataFrame with aligned epochs and labels
            
        Returns:
            X: Array of shape (n_windows, window_size, n_features, sequence_length)
            y: Array of shape (n_windows, window_size) - labels for each epoch in window
            epoch_ids: List of epoch IDs for each window
        """
        print(f"\nCreating windows (window_size={WINDOW_SIZE}, overlap={OVERLAP})...")
        
        # Get unique epochs
        epochs = sorted(df['epoch'].unique())
        n_epochs = len(epochs)
        
        # Calculate step size
        step = int(WINDOW_SIZE * (1 - OVERLAP))
        
        windows = []
        labels = []
        epoch_ids = []
        
        # Slide window across epochs
        for start_idx in range(0, n_epochs - WINDOW_SIZE + 1, step):
            end_idx = start_idx + WINDOW_SIZE
            window_epochs = epochs[start_idx:end_idx]
            
            # Extract data for this window
            window_data = df[df['epoch'].isin(window_epochs)]
            
            # For each epoch in window, extract features
            epoch_features = []
            epoch_labels = []
            
            for epoch in window_epochs:
                epoch_df = window_data[window_data['epoch'] == epoch]
                
                if len(epoch_df) == 0:
                    continue
                
                # Extract features (ACC_X, ACC_Y, ACC_Z, HR)
                features = epoch_df[FEATURES_TO_USE].values  # Shape: (samples_in_epoch, n_features)
                
                # Ensure we have data for the full epoch (30 seconds)
                # ACC is at 32Hz, so 30 seconds = 960 samples
                # HR is at 1Hz, so 30 seconds = 30 samples
                # We'll need to handle different sampling rates
                
                epoch_features.append(features)
                epoch_labels.append(epoch_df['label'].iloc[0])  # Label for this epoch
            
            if len(epoch_features) == WINDOW_SIZE:
                windows.append(epoch_features)
                labels.append(epoch_labels)
                epoch_ids.append(window_epochs)
        
        print(f"Created {len(windows)} windows")
        
        return windows, labels, epoch_ids
    
    def combine_features_to_single_channel(self, windows: List) -> List:
        """
        Combine multi-channel features (ACC_X, ACC_Y, ACC_Z, HR) into format for model
        Your WatchSleepNet expects single-channel input per segment
        
        We'll compute the magnitude of acceleration + HR as single channel
        
        Args:
            windows: List of windows, each containing list of epoch features
            
        Returns:
            Combined windows ready for model
        """
        print("\nCombining features into single channel...")
        
        combined_windows = []
        
        for window in windows:
            window_combined = []
            
            for epoch_features in window:
                # epoch_features shape: (variable_length, 4) for [ACC_X, ACC_Y, ACC_Z, HR]
                if len(epoch_features) > 0:
                    # Compute acceleration magnitude: sqrt(x^2 + y^2 + z^2)
                    acc_mag = np.sqrt(
                        epoch_features[:, 0]**2 + 
                        epoch_features[:, 1]**2 + 
                        epoch_features[:, 2]**2
                    )
                    
                    # For now, just use acceleration magnitude
                    # Can experiment with including HR later
                    combined = acc_mag
                else:
                    combined = np.array([])
                
                window_combined.append(combined)
            
            combined_windows.append(window_combined)
        
        return combined_windows
    
    def resample_to_fixed_length(self, windows: List, target_length: int = 960) -> np.ndarray:
        """
        Resample each epoch to fixed length
        Since ACC is 32Hz and epoch is 30s, we expect ~960 samples
        
        Args:
            windows: List of windows, each containing list of epoch features (1D arrays)
            target_length: Target number of samples per epoch
            
        Returns:
            X: Array of shape (n_windows, window_size, target_length)
        """
        print(f"\nResampling to fixed length ({target_length} samples per epoch)...")
        
        X = []
        
        for window in windows:
            window_resampled = []
            
            for epoch_features in window:
                # epoch_features shape: (variable_length,) - single channel
                n_samples = len(epoch_features)
                
                if n_samples == 0:
                    # No data, pad with zeros
                    resampled = np.zeros(target_length)
                elif n_samples < target_length:
                    # Interpolate up
                    x_old = np.linspace(0, 1, n_samples)
                    x_new = np.linspace(0, 1, target_length)
                    resampled = np.interp(x_new, x_old, epoch_features)
                elif n_samples > target_length:
                    # Downsample
                    indices = np.linspace(0, n_samples - 1, target_length).astype(int)
                    resampled = epoch_features[indices]
                else:
                    resampled = epoch_features
                
                window_resampled.append(resampled)
            
            X.append(np.array(window_resampled))
        
        X = np.array(X)  # Shape: (n_windows, window_size, target_length)
        
        print(f"Final shape: {X.shape}")
        return X
    
    def normalize_features(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Normalize features using z-score normalization
        
        Args:
            X: Array of shape (n_windows, window_size, sequence_length)
            
        Returns:
            X_normalized: Normalized array
            normalization_params: Dict with mean and std
        """
        print("\nNormalizing features...")
        
        # Calculate mean and std across all data
        mean = np.mean(X)
        std = np.std(X)
        
        # Normalize
        X = (X - mean) / (std + 1e-8)
        
        normalization_params = {
            'mean': mean,
            'std': std
        }
        
        print(f"Normalization: mean={mean:.2f}, std={std:.2f}")
        
        return X, normalization_params
    
    def process_patient(self, csv_path: str, save_path: str = None) -> Dict:
        """
        Full preprocessing pipeline for one patient
        
        Args:
            csv_path: Path to patient CSV
            save_path: Optional path to save processed data
            
        Returns:
            Dictionary with processed data and metadata
        """
        print(f"\n{'='*60}")
        print(f"Processing patient: {Path(csv_path).stem}")
        print(f"{'='*60}")
        
        # Load data
        df = self.load_patient_data(csv_path)
        
        # Clean data
        df = self.clean_data(df)
        
        # Align labels to epochs
        df = self.align_labels_to_epochs(df)
        
        # Map sleep stages
        df = self.map_sleep_stages(df)
        
        # Create windows
        windows, labels, epoch_ids = self.create_windows(df)
        
        # Combine features to single channel (for your model)
        windows = self.combine_features_to_single_channel(windows)
        
        # Resample to fixed length
        X = self.resample_to_fixed_length(windows)
        y = np.array(labels)
        
        # Normalize
        X, norm_params = self.normalize_features(X)
        
        # Package results
        result = {
            'X': X,  # Shape: (n_windows, window_size, sequence_length)
            'y': y,  # Shape: (n_windows, window_size)
            'epoch_ids': epoch_ids,
            'normalization_params': norm_params,
            'patient_id': Path(csv_path).stem,
            'n_windows': len(X),
            'window_size': X.shape[1],
            'sequence_length': X.shape[2],
            'n_classes': N_CLASSES
        }
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Number of windows: {len(X)}")
        print(f"{'='*60}\n")
        
        # Save if requested
        if save_path:
            np.savez(save_path, **result)
            print(f"Saved processed data to: {save_path}")
        
        return result


if __name__ == "__main__":
    # Example usage
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    
    # Find first patient CSV
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    
    if len(csv_files) == 0:
        print("No CSV files found in data/raw/")
        print("Please place your patient CSV files there first.")
    else:
        print(f"Found {len(csv_files)} patient file(s)")
        
        # Process first patient
        preprocessor = DREAMTPreprocessor(use_simplified_stages=True)
        
        csv_path = csv_files[0]
        save_path = PROCESSED_DATA_DIR / f"{csv_path.stem}_processed.npz"
        
        result = preprocessor.process_patient(str(csv_path), str(save_path))
        
        print("\nYou can now use this processed data for training!")
        print("The data is saved as a .npz file that can be loaded with np.load()")