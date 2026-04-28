"""
Medical Data Clustering Analysis
Dataset: Wisconsin Breast Cancer Diagnostic Dataset
Features: 30 numeric features extracted from digitized images of fine needle aspirates (FNA)
          of breast masses (radius, texture, perimeter, area, smoothness, etc.)
Algorithms: K-Means, Hierarchical (Agglomerative), DBSCAN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             adjusted_rand_score, confusion_matrix, classification_report)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ──────────────────────────────────────────────
print("=" * 65)
print("  MEDICAL IMAGING DATA CLUSTERING ANALYSIS")
print("  Wisconsin Breast Cancer Diagnostic Dataset")
print("=" * 65)

data = load_breast_cancer()
X = data.data
y = data.target          # 0 = malignant, 1 = benign  (used only for evaluation)
feature_names = data.feature_names
target_names  = data.target_names

df = pd.DataFrame(X, columns=feature_names)
df['diagnosis'] = y
df['diagnosis_label'] = df['diagnosis'].map({0: 'Malignant', 1: 'Benign'})

print(f"\n📊 Dataset Overview")
print(f"   Samples   : {X.shape[0]}")
print(f"   Features  : {X.shape[1]}")
print(f"   Malignant : {(y == 0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"   Benign    : {(y == 1).sum()} ({(y==1).mean()*100:.1f}%)")
print(f"\n   Feature groups: mean | standard error | worst")
print(f"   (radius, texture, perimeter, area, smoothness, ...)")

# ──────────────────────────────────────────────
# 2. PRE-PROCESSING
# ──────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA()
pca_full.fit(X_scaled)
cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
n_comp   = np.argmax(cumvar >= 0.95) + 1
print(f"\n🔬 PCA: {n_comp} components explain ≥ 95 % variance")

pca2 = PCA(n_components=2)
pca5 = PCA(n_components=n_comp)
X_2d = pca2.fit_transform(X_scaled)
X_nd = pca5.fit_transform(X_scaled)
print(f"   2-D PCA variance explained: {pca2.explained_variance_ratio_.sum()*100:.1f}%")

# ──────────────────────────────────────────────
# 3. K-MEANS  —  elbow + silhouette to pick k
# ──────────────────────────────────────────────
print("\n🔄 Running K-Means (k = 2 … 8) …")
k_range   = range(2, 9)
inertias, sil_scores, db_scores = [], [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_nd)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_nd, labels))
    db_scores.append(davies_bouldin_score(X_nd, labels))

best_k = k_range.start + np.argmax(sil_scores)
print(f"   Best k by silhouette: k = {best_k}  (score = {max(sil_scores):.4f})")

km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_labels = km_best.fit_predict(X_scaled)

km_sil = silhouette_score(X_scaled, km_labels)
km_db  = davies_bouldin_score(X_scaled, km_labels)
km_ari = adjusted_rand_score(y, km_labels)
print(f"   Silhouette          : {km_sil:.4f}")
print(f"   Davies-Bouldin      : {km_db:.4f}")
print(f"   Adj. Rand Index     : {km_ari:.4f}")

# ──────────────────────────────────────────────
# 4. HIERARCHICAL CLUSTERING
# ──────────────────────────────────────────────
print("\n🌲 Running Agglomerative Clustering (Ward linkage) …")
agg = AgglomerativeClustering(n_clusters=2, linkage='ward')
agg_labels = agg.fit_predict(X_scaled)

agg_sil = silhouette_score(X_scaled, agg_labels)
agg_db  = davies_bouldin_score(X_scaled, agg_labels)
agg_ari = adjusted_rand_score(y, agg_labels)
print(f"   Silhouette          : {agg_sil:.4f}")
print(f"   Davies-Bouldin      : {agg_db:.4f}")
print(f"   Adj. Rand Index     : {agg_ari:.4f}")

# linkage matrix for dendrogram (use subsample for speed)
idx_sample = np.random.choice(len(X_scaled), 80, replace=False)
Z = linkage(X_scaled[idx_sample], method='ward')

# ──────────────────────────────────────────────
# 5. DBSCAN  —  tune eps via k-distance graph
# ──────────────────────────────────────────────
print("\n🔵 Running DBSCAN …")
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
dists, _ = nbrs.kneighbors(X_scaled)
k_dists  = np.sort(dists[:, -1])

eps_val  = float(np.percentile(k_dists, 90))
db_model = DBSCAN(eps=eps_val, min_samples=5)
db_labels= db_model.fit_predict(X_scaled)

n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise       = (db_labels == -1).sum()
print(f"   eps (90th pct)      : {eps_val:.4f}")
print(f"   Clusters found      : {n_clusters_db}")
print(f"   Noise points        : {n_noise} ({n_noise/len(db_labels)*100:.1f}%)")

if n_clusters_db > 1:
    mask = db_labels != -1
    db_sil = silhouette_score(X_scaled[mask], db_labels[mask])
    db_db  = davies_bouldin_score(X_scaled[mask], db_labels[mask])
    db_ari = adjusted_rand_score(y[mask], db_labels[mask])
    print(f"   Silhouette          : {db_sil:.4f}")
    print(f"   Davies-Bouldin      : {db_db:.4f}")
    print(f"   Adj. Rand Index     : {db_ari:.4f}")
else:
    db_sil = db_db = db_ari = None
    print("   (Only 1 cluster — metrics not applicable)")

# ──────────────────────────────────────────────
# 6. VISUALISATION  (10-panel figure)
# ──────────────────────────────────────────────
COLORS_2 = ['#E74C3C', '#2ECC71']
COLORS_N = plt.cm.tab10.colors

fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor('#0F1923')
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.42, wspace=0.35)

def ax_style(ax, title, xlabel='PC 1', ylabel='PC 2'):
    ax.set_facecolor('#1A2535')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color='#AAB8C2', fontsize=9)
    ax.set_ylabel(ylabel, color='#AAB8C2', fontsize=9)
    ax.tick_params(colors='#AAB8C2', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2C3E50')

# ── Panel 0: Ground truth ──
ax0 = fig.add_subplot(gs[0, 0])
for label, name, c in zip([1,0], ['Benign','Malignant'], ['#2ECC71','#E74C3C']):
    m = y == label
    ax0.scatter(X_2d[m,0], X_2d[m,1], c=c, s=18, alpha=0.7, label=name)
ax0.legend(fontsize=8, framealpha=0.3, labelcolor='white')
ax_style(ax0, '① Ground Truth (PCA 2D)')

# ── Panel 1: Elbow curve ──
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(list(k_range), inertias, 'o-', color='#3498DB', lw=2, ms=6)
ax1.axvline(best_k, color='#F39C12', ls='--', lw=1.5, label=f'Best k={best_k}')
ax1.legend(fontsize=8, framealpha=0.3, labelcolor='white')
ax_style(ax1, '② K-Means Elbow Curve', xlabel='k', ylabel='Inertia')

# ── Panel 2: Silhouette vs k ──
ax2 = fig.add_subplot(gs[0, 2])
bars = ax2.bar(list(k_range), sil_scores, color=['#F39C12' if k==best_k else '#3498DB' for k in k_range])
ax_style(ax2, '③ Silhouette Score vs k', xlabel='k', ylabel='Silhouette')

# ── Panel 3: K-Means clusters ──
ax3 = fig.add_subplot(gs[1, 0])
for lbl in np.unique(km_labels):
    m = km_labels == lbl
    ax3.scatter(X_2d[m,0], X_2d[m,1], c=COLORS_N[lbl], s=18, alpha=0.7, label=f'C{lbl}')
ax3.legend(fontsize=8, framealpha=0.3, labelcolor='white')
ax_style(ax3, f'④ K-Means (k={best_k}) Clusters')

# ── Panel 4: K-Means confusion heatmap ──
ax4 = fig.add_subplot(gs[1, 1])
# align cluster 0 → majority class
map_km = {}
for lbl in np.unique(km_labels):
    majority = pd.Series(y[km_labels==lbl]).mode()[0]
    map_km[lbl] = majority
km_mapped = np.array([map_km[l] for l in km_labels])
cm_km = confusion_matrix(y, km_mapped)
sns.heatmap(cm_km, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred M','Pred B'],
            yticklabels=['True M','True B'],
            ax=ax4, cbar=False, annot_kws={'color':'white','size':12})
ax4.set_facecolor('#1A2535')
ax4.set_title('⑤ K-Means Confusion Matrix', color='white', fontsize=11, fontweight='bold', pad=8)
ax4.tick_params(colors='#AAB8C2', labelsize=8)

# ── Panel 5: Hierarchical clusters ──
ax5 = fig.add_subplot(gs[1, 2])
for lbl in np.unique(agg_labels):
    m = agg_labels == lbl
    ax5.scatter(X_2d[m,0], X_2d[m,1], c=COLORS_2[lbl], s=18, alpha=0.7, label=f'C{lbl}')
ax5.legend(fontsize=8, framealpha=0.3, labelcolor='white')
ax_style(ax5, '⑥ Agglomerative (Ward) Clusters')

# ── Panel 6: Dendrogram ──
ax6 = fig.add_subplot(gs[2, 0])
ax6.set_facecolor('#1A2535')
dendrogram(Z, ax=ax6, truncate_mode='lastp', p=20,
           color_threshold=0.7*max(Z[:,2]),
           above_threshold_color='#95A5A6',
           leaf_font_size=7)
ax6.set_title('⑦ Dendrogram (Ward, n=80)', color='white', fontsize=11, fontweight='bold', pad=8)
ax6.tick_params(colors='#AAB8C2', labelsize=7)
for spine in ax6.spines.values():
    spine.set_edgecolor('#2C3E50')

# ── Panel 7: DBSCAN k-distance ──
ax7 = fig.add_subplot(gs[2, 1])
ax7.plot(k_dists, color='#9B59B6', lw=1.5)
ax7.axhline(eps_val, color='#F39C12', ls='--', lw=1.5, label=f'eps={eps_val:.2f}')
ax7.legend(fontsize=8, framealpha=0.3, labelcolor='white')
ax_style(ax7, '⑧ DBSCAN k-Distance Graph', xlabel='Points (sorted)', ylabel='5-NN Distance')

# ── Panel 8: DBSCAN clusters ──
ax8 = fig.add_subplot(gs[2, 2])
unique_dbl = np.unique(db_labels)
for lbl in unique_dbl:
    m   = db_labels == lbl
    col = '#555555' if lbl == -1 else COLORS_N[lbl % 10]
    lname = 'Noise' if lbl == -1 else f'C{lbl}'
    ax8.scatter(X_2d[m,0], X_2d[m,1], c=col, s=18, alpha=0.6, label=lname)
ax8.legend(fontsize=8, framealpha=0.3, labelcolor='white')
ax_style(ax8, '⑨ DBSCAN Clusters')

# ── Panel 9: Algorithm comparison bar chart ──
ax9 = fig.add_subplot(gs[3, :])
ax9.set_facecolor('#1A2535')
algos   = ['K-Means', 'Agglomerative', 'DBSCAN']
sil_v   = [km_sil,  agg_sil,  db_sil  if db_sil  else 0]
ari_v   = [km_ari,  agg_ari,  db_ari  if db_ari  else 0]
db_v    = [km_db,   agg_db,   db_db   if db_db   else 0]

x = np.arange(len(algos))
w = 0.25
b1 = ax9.bar(x - w, sil_v, w, label='Silhouette ↑', color='#2ECC71', alpha=0.85)
b2 = ax9.bar(x,     ari_v, w, label='Adj. Rand ↑',  color='#3498DB', alpha=0.85)
b3 = ax9.bar(x + w, db_v,  w, label='Davies-Bouldin ↓', color='#E74C3C', alpha=0.85)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                 f'{h:.3f}', ha='center', va='bottom', color='white', fontsize=8)

ax9.set_xticks(x)
ax9.set_xticklabels(algos, color='white', fontsize=11)
ax9.set_title('⑩ Algorithm Comparison (higher Silhouette & ARI = better; lower DB = better)',
              color='white', fontsize=11, fontweight='bold', pad=8)
ax9.tick_params(colors='#AAB8C2')
ax9.legend(fontsize=9, framealpha=0.3, labelcolor='white')
for spine in ax9.spines.values():
    spine.set_edgecolor('#2C3E50')

# Main title
fig.suptitle('Clustering Analysis — Wisconsin Breast Cancer Diagnostic Data\n'
             'K-Means  |  Agglomerative  |  DBSCAN',
             color='white', fontsize=15, fontweight='bold', y=0.98)

plt.savefig('/mnt/user-data/outputs/medical_clustering.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("\n✅ Figure saved.")

# ──────────────────────────────────────────────
# 7. SUMMARY TABLE
# ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL METRICS SUMMARY")
print("=" * 65)
print(f"{'Algorithm':<22} {'Silhouette':>11} {'Davies-Bouldin':>15} {'Adj. Rand':>10}")
print("-" * 65)
print(f"{'K-Means (k=' + str(best_k) + ')':<22} {km_sil:>11.4f} {km_db:>15.4f} {km_ari:>10.4f}")
print(f"{'Agglomerative':<22} {agg_sil:>11.4f} {agg_db:>15.4f} {agg_ari:>10.4f}")
if db_sil:
    print(f"{'DBSCAN':<22} {db_sil:>11.4f} {db_db:>15.4f} {db_ari:>10.4f}")
else:
    print(f"{'DBSCAN':<22} {'N/A':>11} {'N/A':>15} {'N/A':>10}")
print("=" * 65)

print("""
📝 Interpretation:
  • Silhouette  → closer to 1.0 is better (cluster separation)
  • Davies-Bouldin → closer to 0.0 is better (intra vs inter cluster)
  • Adj. Rand   → closer to 1.0 is better (match with true labels)
""")
EOF
