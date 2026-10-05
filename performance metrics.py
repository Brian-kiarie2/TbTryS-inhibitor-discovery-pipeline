import matplotlib.pyplot as plt
import os

metrics = {
    'R²': 0.6447,
    'Pearson r': 0.8072,
    'Spearman ρ': 0.7918,
    'MAE': 0.5719,
    'RMSE': 0.7432,
}
colors = ['#2E86AB', '#048A81', '#8A2E39', '#F18F01', '#A23B72']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(metrics.keys(), metrics.values(), color=colors,
              edgecolor='black', linewidth=1.5, alpha=0.85)

for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.text(1, 0.85, 'p < 0.001', ha='center', fontsize=9, style='italic', color='#048A81')
ax.text(2, 0.85, 'p = 1.42e-181', ha='center', fontsize=9, style='italic', color='#8A2E39')

ax.set_ylabel('Score / Error Value', fontsize=12, fontweight='bold')
ax.set_title('GNN Performance Metrics on Validation Set (n = 840)', fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out = os.path.join(os.path.expanduser("~"), "Desktop", "GNN_Performance_Metrics.png")
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"Saved: {out}")
plt.show()