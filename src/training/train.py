"""
Training script for WatchSleepNet
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import json
from datetime import datetime

# Add paths
sys.path.append('src')
sys.path.append('src/data')
sys.path.append('src/models')
sys.path.append('configs')
sys.path.append('data')

from config import *
from watchsleepnet import WatchSleepNet
from process_all_patients import load_all_processed_patients
from dataloader import create_dataloaders, create_leave_one_out_dataloaders


class Trainer:
    """
    Trainer class for WatchSleepNet
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        save_dir='results/models'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function with class weights
        class_weights = train_loader.dataset.class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        if USE_LR_SCHEDULER:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=LR_SCHEDULER_FACTOR,
                patience=LR_SCHEDULER_PATIENCE,
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f'results/runs/experiment_{timestamp}')
        
        print(f"\nTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Class weights: {class_weights.cpu().numpy()}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (x, y, lengths) in enumerate(pbar):
            # Move to device
            x = x.to(self.device)  # (batch, num_epochs, sequence_length)
            y = y.to(self.device)  # (batch, num_epochs)
            lengths = lengths  # Keep on CPU for packing
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x, lengths)  # (batch, num_epochs, num_classes)
            
            # Compute loss (only on valid positions)
            loss = self._compute_loss(outputs, y, lengths)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(outputs, dim=2)
            mask = self._create_mask(lengths, y.size(1)).to(self.device)
            correct += ((predicted == y) * mask).sum().item()
            total += mask.sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for x, y, lengths in pbar:
                # Move to device
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                outputs = self.model(x, lengths)
                
                # Compute loss
                loss = self._compute_loss(outputs, y, lengths)
                total_loss += loss.item()
                
                # Compute accuracy
                _, predicted = torch.max(outputs, dim=2)
                mask = self._create_mask(lengths, y.size(1)).to(self.device)
                correct += ((predicted == y) * mask).sum().item()
                total += mask.sum().item()
                
                # Store predictions for detailed metrics
                all_preds.extend(predicted[mask].cpu().numpy())
                all_labels.extend(y[mask].cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)
    
    def _compute_loss(self, outputs, targets, lengths):
        """
        Compute loss only on valid positions
        
        Args:
            outputs: (batch, seq_len, num_classes)
            targets: (batch, seq_len)
            lengths: (batch,) - actual lengths
        """
        batch_size, max_len, num_classes = outputs.shape
        
        # Flatten
        outputs_flat = outputs.reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)
        
        # Create mask for valid positions
        mask = self._create_mask(lengths, max_len).reshape(-1).to(outputs.device)
        
        # Filter valid positions
        outputs_valid = outputs_flat[mask]
        targets_valid = targets_flat[mask]
        
        return self.criterion(outputs_valid, targets_valid)
    
    def _create_mask(self, lengths, max_len):
        """Create mask for variable-length sequences"""
        batch_size = len(lengths)
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING")
        print(f"{'='*70}\n")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=True)
                print(f"  ✅ New best model saved!")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epoch(s)")
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=False)
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        self.writer.close()
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history as JSON"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss
        }
        
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)


def main():
    """Main training function"""
    
    # Set random seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load processed data
    print("\nLoading processed data...")
    patients = load_all_processed_patients()
    
    if len(patients) < 3:
        print("⚠️ Warning: You have fewer than 3 patients. Consider using leave-one-out CV.")
    
    # Split patients (simple split for now)
    patient_ids = [p['patient_id'] for p in patients]
    n_patients = len(patients)
    
    if n_patients >= 5:
        # Regular split
        n_train = max(1, int(0.6 * n_patients))
        n_val = max(1, int(0.2 * n_patients))
        
        train_ids = patient_ids[:n_train]
        val_ids = patient_ids[n_train:n_train+n_val]
        test_ids = patient_ids[n_train+n_val:]
        
        print(f"\nPatient split:")
        print(f"  Train: {train_ids}")
        print(f"  Val: {val_ids}")
        print(f"  Test: {test_ids}")
        
        train_loader, val_loader, test_loader = create_dataloaders(
            patients,
            train_patients=train_ids,
            val_patients=val_ids,
            test_patients=test_ids if test_ids else None,
            batch_size=BATCH_SIZE
        )
    else:
        # Leave-one-out for small datasets
        print(f"\n⚠️ Small dataset ({n_patients} patients) - using leave-one-out CV")
        test_patient = patient_ids[-1]
        val_patient = patient_ids[-2] if len(patient_ids) > 1 else None
        
        train_loader, val_loader, test_loader = create_leave_one_out_dataloaders(
            patients,
            test_patient=test_patient,
            val_patient=val_patient,
            batch_size=BATCH_SIZE
        )
    
    # Create model
    print("\nInitializing model...")
    model = WatchSleepNet(
        num_features=NUM_FEATURES,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=N_CLASSES,
        tcn_layers=TCN_LAYERS,
        use_tcn=USE_TCN,
        use_attention=USE_ATTENTION
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE
    )
    
    # Train
    trainer.train()
    
    print("\n✅ Training complete! Check results/models/ for saved models.")
    print("📊 View training curves: tensorboard --logdir results/runs")


if __name__ == "__main__":
    main()