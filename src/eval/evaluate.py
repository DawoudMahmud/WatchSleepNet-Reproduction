"""
Evaluation script for WatchSleepNet
Generates detailed metrics, confusion matrix, and per-class performance
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score
)
import json
from pathlib import Path
import sys

# Add paths
sys.path.append('src')
sys.path.append('src/data')
sys.path.append('src/models')
sys.path.append('configs')
sys.path.append('data')

from config import *
from watchsleepnet import WatchSleepNet
from process_all_patients import load_all_processed_patients
from dataloader import create_dataloaders


class ModelEvaluator:
    # ... rest stays the same
    """Evaluate trained WatchSleepNet model"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model_path = model_path
        self.stage_names = {0: 'Wake', 1: 'NREM', 2: 'REM'}
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model architecture
        self.model = WatchSleepNet(
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
        
        # Load weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Trained for {self.checkpoint['epoch'] + 1} epochs")
        print(f"   Best val accuracy: {self.checkpoint['val_acc']:.2f}%")
    
    def evaluate(self, data_loader, split_name='Test'):
        """
        Evaluate model on a dataset
        
        Returns:
            predictions, labels, metrics
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING ON {split_name.upper()} SET")
        print(f"{'='*60}")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for x, y, lengths in data_loader:
                # Move to device
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                outputs = self.model(x, lengths)  # (batch, num_epochs, num_classes)
                
                # Get predictions
                probs = torch.softmax(outputs, dim=2)
                _, predicted = torch.max(outputs, dim=2)
                
                # Create mask for valid positions
                max_len = y.size(1)
                mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
                mask = mask.to(self.device)
                
                # Store predictions and labels (only valid positions)
                all_preds.extend(predicted[mask].cpu().numpy())
                all_labels.extend(y[mask].cpu().numpy())
                all_probs.extend(probs[mask].cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_preds)
        labels = np.array(all_labels)
        probs = np.array(all_probs)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, labels, split_name)
        
        return predictions, labels, probs, metrics
    
    def _compute_metrics(self, predictions, labels, split_name):
        """Compute all evaluation metrics"""
        
        # Overall accuracy
        accuracy = accuracy_score(labels, predictions) * 100
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(labels, predictions)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        metrics = {
            'split': split_name,
            'accuracy': accuracy,
            'precision_macro': precision_macro * 100,
            'recall_macro': recall_macro * 100,
            'f1_macro': f1_macro * 100,
            'cohen_kappa': kappa,
            'per_class': {
                'precision': precision * 100,
                'recall': recall * 100,
                'f1': f1 * 100,
                'support': support
            },
            'confusion_matrix': cm
        }
        
        # Print results
        print(f"\n{split_name} Set Results:")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print(f"Macro Precision: {precision_macro*100:.2f}%")
        print(f"Macro Recall: {recall_macro*100:.2f}%")
        print(f"Macro F1-Score: {f1_macro*100:.2f}%")
        print(f"Cohen's Kappa: {kappa:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 60)
        for i in range(len(precision)):
            class_name = self.stage_names[i]
            print(f"{class_name:<10} {precision[i]*100:>10.2f}% {recall[i]*100:>10.2f}% "
                  f"{f1[i]*100:>10.2f}% {support[i]:>10}")
        
        return metrics
    
    def plot_confusion_matrix(self, cm, save_path='results/figures/confusion_matrix.png'):
        """Plot and save confusion matrix"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Plot
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=[self.stage_names[i] for i in range(len(cm))],
            yticklabels=[self.stage_names[i] for i in range(len(cm))],
            cbar_kws={'label': 'Percentage (%)'},
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Confusion matrix saved to: {save_path}")
        
        plt.close()
    
    def plot_per_class_metrics(self, metrics, save_path='results/figures/per_class_metrics.png'):
        """Plot per-class precision, recall, F1"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = [self.stage_names[i] for i in range(N_CLASSES)]
        x = np.arange(len(classes))
        width = 0.25
        
        precision = metrics['per_class']['precision']
        recall = metrics['per_class']['recall']
        f1 = metrics['per_class']['f1']
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Sleep Stage', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 100])
        
        plt.tight_layout()
        
        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Per-class metrics saved to: {save_path}")
        
        plt.close()
    
    def save_results(self, metrics, save_path='results/evaluation_results.json'):
        """Save evaluation results as JSON"""
        
        # Convert numpy arrays to lists for JSON serialization
        results = {
            'split': metrics['split'],
            'accuracy': float(metrics['accuracy']),
            'precision_macro': float(metrics['precision_macro']),
            'recall_macro': float(metrics['recall_macro']),
            'f1_macro': float(metrics['f1_macro']),
            'cohen_kappa': float(metrics['cohen_kappa']),
            'per_class': {
                'precision': metrics['per_class']['precision'].tolist(),
                'recall': metrics['per_class']['recall'].tolist(),
                'f1': metrics['per_class']['f1'].tolist(),
                'support': metrics['per_class']['support'].tolist()
            },
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'model_path': str(self.model_path),
            'config': {
                'use_tcn': USE_TCN,
                'use_attention': USE_ATTENTION,
                'hidden_dim': HIDDEN_DIM,
                'num_channels': NUM_CHANNELS
            }
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to: {save_path}")


def main():
    """Main evaluation function"""
    
    # Load processed data
    print("Loading processed data...")
    patients = load_all_processed_patients()
    
    if len(patients) == 0:
        print("❌ No processed data found. Run process_all_patients.py first.")
        return
    
    # Split patients (same as training)
    patient_ids = [p['patient_id'] for p in patients]
    n_patients = len(patients)
    
    if n_patients >= 5:
        n_train = max(1, int(0.6 * n_patients))
        n_val = max(1, int(0.2 * n_patients))
        
        train_ids = patient_ids[:n_train]
        val_ids = patient_ids[n_train:n_train+n_val]
        test_ids = patient_ids[n_train+n_val:]
    else:
        # Leave-one-out
        test_ids = [patient_ids[-1]]
        val_ids = [patient_ids[-2]] if len(patient_ids) > 1 else []
        train_ids = [pid for pid in patient_ids if pid not in test_ids + val_ids]
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        patients,
        train_patients=train_ids,
        val_patients=val_ids,
        test_patients=test_ids if test_ids else None,
        batch_size=BATCH_SIZE
    )
    
    # Initialize evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = ModelEvaluator('results/models/best_model.pth', device=device)
    
    # Evaluate on validation set
    val_preds, val_labels, val_probs, val_metrics = evaluator.evaluate(val_loader, 'Validation')
    
    # Plot results
    evaluator.plot_confusion_matrix(val_metrics['confusion_matrix'], 
                                   'results/figures/confusion_matrix_val.png')
    evaluator.plot_per_class_metrics(val_metrics,
                                    'results/figures/per_class_metrics_val.png')
    
    # Save results
    evaluator.save_results(val_metrics, 'results/evaluation_results_val.json')
    
    # If test set exists, evaluate on it too
    if test_loader:
        test_preds, test_labels, test_probs, test_metrics = evaluator.evaluate(test_loader, 'Test')
        evaluator.plot_confusion_matrix(test_metrics['confusion_matrix'],
                                       'results/figures/confusion_matrix_test.png')
        evaluator.plot_per_class_metrics(test_metrics,
                                        'results/figures/per_class_metrics_test.png')
        evaluator.save_results(test_metrics, 'results/evaluation_results_test.json')
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*60}")
    print("\nGenerated files:")
    print("  📊 results/figures/confusion_matrix_*.png")
    print("  📊 results/figures/per_class_metrics_*.png")
    print("  📄 results/evaluation_results_*.json")


if __name__ == "__main__":
    main()