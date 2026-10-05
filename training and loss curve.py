import matplotlib.pyplot as plt
import os

epochs = [20, 40, 60, 80, 100]
train_loss = [1.1447, 0.8765, 0.7234, 0.6345, 0.5534]
val_loss   = [1.2472, 0.9856, 0.8765, 0.8234, 0.8081]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs, train_loss, 'o-', color='#1f77b4', linewidth=2.5, markersize=9,
        markeredgecolor='white', markeredgewidth=1.2, label='Training Loss')
ax.plot(epochs, val_loss, 's-', color='#d62728', linewidth=2.5, markersize=9,
        markeredgecolor='white', markeredgewidth=1.2, label='Validation Loss')

for e, tl, vl in zip(epochs, train_loss, val_loss):
    ax.annotate(f'{tl:.4f}', (e, tl), textcoords='offset points', xytext=(0, -18),
                ha='center', va='top', fontsize=9, color='#1f77b4', fontweight='bold')
    ax.annotate(f'{vl:.4f}', (e, vl), textcoords='offset points', xytext=(0, 10),
                ha='center', va='bottom', fontsize=9, color='#d62728', fontweight='bold')

ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('Loss (MSE)', fontsize=13, fontweight='bold')
ax.set_title('Training and Validation Loss of the GNN Model', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(epochs)
ax.set_xlim(15, 105)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

plt.tight_layout()
out = os.path.join(os.path.expanduser("~"), "Desktop", "GNN_Training_Validation_Loss.png")
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"Saved: {out}")
plt.show()