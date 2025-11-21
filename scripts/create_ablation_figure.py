import matplotlib.pyplot as plt
import json
import numpy as np

# Load training histories for each experiment
# Adjust paths as needed based on where you saved them

configs = [
    "Full Model (TCN+Attn)",
    "TCN Only",
    "Attention Only", 
    "Minimal"
]

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot training loss
for i, (config, color) in enumerate(zip(configs, colors)):
    # Load your training history - adjust path
    # history = json.load(open(f'results/ablation_{i}/training_history.json'))
    # For now, placeholder
    epochs = range(1, 17)
    loss = np.random.rand(16) * 0.5 + 0.7 - i*0.05  # Placeholder
    ax1.plot(epochs, loss, label=config, color=color, linewidth=2)

ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Validation Loss', fontsize=12)
ax1.set_title('Validation Loss Across Configurations', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot validation accuracy
for i, (config, color) in enumerate(zip(configs, colors)):
    epochs = range(1, 17)
    acc = np.random.rand(16) * 10 + 55 + i*2  # Placeholder
    ax2.plot(epochs, acc, label=config, color=color, linewidth=2)

ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax2.set_title('Validation Accuracy Across Configurations', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ablation_training_curves.png', dpi=300, bbox_inches='tight')
print("✅ Ablation figure saved!")