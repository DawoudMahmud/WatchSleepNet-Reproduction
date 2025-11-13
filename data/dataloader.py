"""
PyTorch DataLoader for WatchSleepNet
Handles loading processed patient data for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import sys

sys.path.append('configs')
from config import BATCH_SIZE, RANDOM_SEED


class SleepDataset(Dataset):
    """
    Dataset for sleep stage classification
    Each sample is a window containing multiple epochs
    """
    
    def __init__(self, patients_data: List[Dict], augment=False):
        """
        Args:
            patients_data: List of patient dictionaries from load_all_processed_patients()
            augment: Whether to apply data augmentation (for training)
        """
        self.augment = augment
        
        # Concatenate all patients' data
        self.X = np.concatenate([p['X'] for p in patients_data], axis=0)
        self.y = np.concatenate([p['y'] for p in patients_data], axis=0)
        
        # Store patient IDs for each window (useful for leave-one-out CV)
        self.patient_ids = []
        for p in patients_data:
            self.patient_ids.extend([p['patient_id']] * p['n_windows'])
        
        print(f"Dataset created: {len(self)} windows from {len(patients_data)} patients")
        print(f"  X shape: {self.X.shape}")
        print(f"  y shape: {self.y.shape}")
        
        # Compute class weights for imbalanced data
        self.class_weights = self._compute_class_weights()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Returns:
            x: Window of shape (num_epochs, sequence_length)
            y: Labels of shape (num_epochs,)
            length: Actual number of valid epochs (for variable-length sequences)
        """
        x = self.X[idx]  # Shape: (num_epochs, sequence_length)
        y = self.y[idx]  # Shape: (num_epochs,)
        
        # Data augmentation (if enabled)
        if self.augment:
            x = self._augment(x)
        
        # Convert to tensors
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)
        
        # For your model: length is number of epochs (all valid since we don't have padding yet)
        length = torch.tensor(x.shape[0], dtype=torch.long)
        
        return x, y, length
    
    def _augment(self, x):
        """
        Simple data augmentation:
        - Random amplitude scaling
        - Random noise addition
        """
        # Random amplitude scaling (0.9 to 1.1)
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale
        
        # Random noise (small)
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.01, x.shape)
            x = x + noise
        
        return x
    
    def _compute_class_weights(self):
        """
        Compute class weights for handling imbalanced data
        """
        unique, counts = np.unique(self.y, return_counts=True)
        total = len(self.y.flatten())
        
        # Inverse frequency weighting
        weights = total / (len(unique) * counts)
        
        print(f"\nClass distribution:")
        stage_names = {0: 'Wake', 1: 'NREM', 2: 'REM'}
        for u, c, w in zip(unique, counts, weights):
            print(f"  {stage_names[u]}: {c:,} samples (weight: {w:.3f})")
        
        return torch.FloatTensor(weights)


def collate_fn(batch):
    """
    Custom collate function for variable-length sequences
    Pads sequences to the same length within a batch
    """
    xs, ys, lengths = zip(*batch)
    
    # Stack (all same length for now, but this handles future variable-length)
    xs = torch.stack(xs)  # (batch, num_epochs, sequence_length)
    ys = torch.stack(ys)  # (batch, num_epochs)
    lengths = torch.stack(lengths)  # (batch,)
    
    return xs, ys, lengths


def create_dataloaders(
    patients_data: List[Dict],
    train_patients: List[str],
    val_patients: List[str],
    test_patients: List[str] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        patients_data: All processed patient data
        train_patients: List of patient IDs for training
        val_patients: List of patient IDs for validation
        test_patients: List of patient IDs for testing (optional)
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Split data by patient
    train_data = [p for p in patients_data if p['patient_id'] in train_patients]
    val_data = [p for p in patients_data if p['patient_id'] in val_patients]
    test_data = [p for p in patients_data if p['patient_id'] in test_patients] if test_patients else []
    
    print(f"\nCreating dataloaders...")
    print(f"  Train: {len(train_data)} patients")
    print(f"  Val: {len(val_data)} patients")
    if test_data:
        print(f"  Test: {len(test_data)} patients")
    
    # Create datasets
    train_dataset = SleepDataset(train_data, augment=True)
    val_dataset = SleepDataset(val_data, augment=False)
    test_dataset = SleepDataset(test_data, augment=False) if test_data else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )
    
    return train_loader, val_loader, test_loader


def create_leave_one_out_dataloaders(
    patients_data: List[Dict],
    test_patient: str,
    val_patient: str = None,
    batch_size: int = BATCH_SIZE
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for leave-one-out cross-validation
    
    Args:
        patients_data: All processed patient data
        test_patient: Patient ID to use for testing
        val_patient: Patient ID to use for validation (if None, uses another patient)
        batch_size: Batch size
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    patient_ids = [p['patient_id'] for p in patients_data]
    
    # Select validation patient if not specified
    if val_patient is None:
        available = [pid for pid in patient_ids if pid != test_patient]
        if len(available) > 0:
            val_patient = available[0]
        else:
            raise ValueError("Not enough patients for leave-one-out split")
    
    # Remaining patients for training
    train_patients = [pid for pid in patient_ids if pid not in [test_patient, val_patient]]
    
    print(f"\nLeave-one-out split:")
    print(f"  Test: {test_patient}")
    print(f"  Val: {val_patient}")
    print(f"  Train: {train_patients}")
    
    return create_dataloaders(
        patients_data,
        train_patients=train_patients,
        val_patients=[val_patient],
        test_patients=[test_patient],
        batch_size=batch_size
    )


if __name__ == "__main__":
    # Example usage
    from process_all_patients import load_all_processed_patients
    
    # Load all processed patients
    patients = load_all_processed_patients()
    
    if len(patients) == 0:
        print("No processed data found. Run process_all_patients.py first.")
    else:
        # Example: Simple split (first 60% train, 20% val, 20% test)
        n_patients = len(patients)
        n_train = int(0.6 * n_patients)
        n_val = int(0.2 * n_patients)
        
        patient_ids = [p['patient_id'] for p in patients]
        train_ids = patient_ids[:n_train]
        val_ids = patient_ids[n_train:n_train+n_val]
        test_ids = patient_ids[n_train+n_val:]
        
        print(f"\nSplit: {n_train} train, {n_val} val, {len(test_ids)} test")
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            patients,
            train_patients=train_ids,
            val_patients=val_ids,
            test_patients=test_ids if test_ids else None
        )
        
        # Test a batch
        print("\nTesting dataloader...")
        for x, y, lengths in train_loader:
            print(f"  Batch x shape: {x.shape}")
            print(f"  Batch y shape: {y.shape}")
            print(f"  Lengths: {lengths}")
            break
        
        print("\n✅ DataLoader working correctly!")