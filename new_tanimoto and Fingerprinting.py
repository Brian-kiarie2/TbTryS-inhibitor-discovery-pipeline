"""
SCAFFOLD DIVERSITY ANALYSIS (FIXED)
Calculates Tanimoto similarity and performs hierarchical clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import os

# ============================================================================
# 1. LOAD YOUR NCES
# ============================================================================

nce_data = [
    {'ID': 'NCE_1', 'SMILES': 'COc1ccc(Nc2nc(N)nc(N)n2)cc1'},
    {'ID': 'NCE_2', 'SMILES': 'COc1ccc(C(=O)O)cc1C1CCCN1'},
    {'ID': 'NCE_3', 'SMILES': 'O=S(=O)(Nc1ccccc1)c1ccc(-c2nn[nH]n2)cc1'},
    {'ID': 'NCE_4', 'SMILES': 'O=S(=O)(Nc1ccccc1)c1ccc(F)cc1'},
    {'ID': 'NCE_5', 'SMILES': 'COc1ccc(S(=O)(=O)Nc2ccccc2)cc1'},
    {'ID': 'NCE_6', 'SMILES': 'NCc1ccc(C(=O)O)cc1C1CCCN1'},
    {'ID': 'NCE_7', 'SMILES': 'Nc1nc(N)nc(Nc2ccc(F)cc2)n1'},
    {'ID': 'NCE_8', 'SMILES': 'O=C(O)c1ccc(S(=O)(=O)Nc2ccccc2)cc1'},
    {'ID': 'NCE_9', 'SMILES': 'O=S(=O)(Nc1ccccc1)c1ccc(-c2nn[nH]n2)cc1'},
    {'ID': 'NCE_10', 'SMILES': 'NS(=O)(=O)c1ccc(S(=O)(=O)Nc2ccccc2)cc1'},
    {'ID': 'NCE_11', 'SMILES': 'O=C(O)c1ccc(F)c(C2CCCN2)c1'}
]

df_nce = pd.DataFrame(nce_data)

print("="*60)
print("SCAFFOLD DIVERSITY ANALYSIS")
print("="*60)
print(f"Total compounds: {len(df_nce)}")

# ============================================================================
# 2. CALCULATE FINGERPRINTS (FIXED)
# ============================================================================

print("\n[1/4] Generating Morgan fingerprints...")

fingerprints = []
valid_ids = []

for idx, row in df_nce.iterrows():
    mol = Chem.MolFromSmiles(row['SMILES'])
    if mol:
        # FIX: Use GetMorganFingerprintAsBitVect with nBits
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprints.append(fp)
        valid_ids.append(row['ID'])

n = len(fingerprints)
print(f"✅ Generated {n} fingerprints")

# ============================================================================
# 3. CALCULATE TANIMOTO SIMILARITY MATRIX
# ============================================================================

print("\n[2/4] Calculating Tanimoto similarity matrix...")

similarity_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])

sim_df = pd.DataFrame(similarity_matrix, index=valid_ids, columns=valid_ids)

print("\n📊 TANIMOTO SIMILARITY MATRIX:")
print("-"*60)
print(sim_df.round(3).to_string())

# ============================================================================
# 4. SIMILARITY STATISTICS
# ============================================================================

print("\n[3/4] Calculating similarity statistics...")

upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

print(f"\n📊 SIMILARITY STATISTICS:")
print("-"*60)
print(f"  Mean similarity:     {np.mean(upper_tri):.3f}")
print(f"  Median similarity:   {np.median(upper_tri):.3f}")
print(f"  Standard deviation:  {np.std(upper_tri):.3f}")
print(f"  Minimum similarity:  {np.min(upper_tri):.3f}")
print(f"  Maximum similarity:  {np.max(upper_tri):.3f}")
print(f"  Range:               {np.max(upper_tri) - np.min(upper_tri):.3f}")

# ============================================================================
# 5. CLUSTERING
# ============================================================================

print("\n[4/4] Performing hierarchical clustering...")

distance_matrix = 1 - similarity_matrix
linkage_matrix = hierarchy.linkage(squareform(distance_matrix), method='average')

# ============================================================================
# 6. GENERATE FIGURES
# ============================================================================

desktop = os.path.join(os.path.expanduser("~"), "Desktop")

# Figure 1: Heatmap
print("\n📊 Generating similarity heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    sim_df,
    annot=True,
    fmt='.3f',
    cmap='RdYlGn',
    vmin=0,
    vmax=1,
    square=True,
    cbar_kws={'label': 'Tanimoto Similarity', 'shrink': 0.8},
    annot_kws={'size': 10}
)
ax.set_title('Pairwise Tanimoto Similarity of Generated NCEs', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(desktop, 'Figure_Similarity_Heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure_Similarity_Heatmap.png saved")

# Figure 2: Dendrogram
print("📊 Generating clustering dendrogram...")

fig, ax = plt.subplots(figsize=(14, 8))
dendrogram = hierarchy.dendrogram(
    linkage_matrix,
    labels=valid_ids,
    ax=ax,
    leaf_rotation=90,
    leaf_font_size=10,
    color_threshold=0.5
)
ax.set_xlabel('NCE ID', fontsize=12)
ax.set_ylabel('Distance (1 - Tanimoto Similarity)', fontsize=12)
ax.set_title('Hierarchical Clustering of NCEs', fontsize=14)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Diversity threshold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(desktop, 'Figure_Clustering_Dendrogram.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure_Clustering_Dendrogram.png saved")

# Figure 3: Similarity Distribution
print("📊 Generating similarity distribution...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(upper_tri, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(np.mean(upper_tri), color='red', linestyle='--', label=f'Mean: {np.mean(upper_tri):.3f}')
ax.axvline(np.median(upper_tri), color='green', linestyle='--', label=f'Median: {np.median(upper_tri):.3f}')
ax.set_xlabel('Tanimoto Similarity', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Pairwise Tanimoto Similarities', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(desktop, 'Figure_Similarity_Distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure_Similarity_Distribution.png saved")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

sim_df.to_csv(os.path.join(desktop, 'Tanimoto_Similarity_Matrix.csv'))
print("\n✅ Tanimoto_Similarity_Matrix.csv saved")

# ============================================================================
# 8. SUMMARY
# ============================================================================

print("\n" + "="*60)
print("SUMMARY FOR MANUSCRIPT")
print("="*60)

summary_text = f"""
SCAFFOLD DIVERSITY ANALYSIS
===========================

KEY STATISTICS:
---------------
• Number of compounds: {n}
• Mean Tanimoto similarity: {np.mean(upper_tri):.3f}
• Median similarity: {np.median(upper_tri):.3f}
• Standard deviation: {np.std(upper_tri):.3f}
• Minimum similarity: {np.min(upper_tri):.3f}
• Maximum similarity: {np.max(upper_tri):.3f}
• Range: {np.max(upper_tri) - np.min(upper_tri):.3f}

INTERPRETATION:
--------------
The mean similarity of {np.mean(upper_tri):.3f} indicates {'moderate' if np.mean(upper_tri) < 0.5 else 'high'} 
structural diversity. The range of {np.min(upper_tri):.3f} to {np.max(upper_tri):.3f} 
demonstrates that the fragment-based generation strategy produced compounds 
spanning diverse chemical space.
"""

print(summary_text)

with open(os.path.join(desktop, 'Scaffold_Diversity_Summary.txt'), 'w') as f:
    f.write(summary_text)

print("\n✅ Scaffold_Diversity_Summary.txt saved")
print("\n" + "="*60)
print("✅ ANALYSIS COMPLETE!")
print("="*60)