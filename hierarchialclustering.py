import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# ✅ Step 1: Create a sample dataset (Similar to Advertisement dataset)
data = {
    'TV': [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8],
    'Radio': [37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 2.1, 2.6],
    'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4, 75.0, 23.5, 11.6, 1.0, 21.2]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# ✅ Step 2: Perform Hierarchical Clustering using Complete Linkage
# - linkage() performs hierarchical clustering
# - 'complete' specifies that complete linkage is used (farthest point distance)
Z = linkage(df, method='complete')

# ✅ Step 3: Plot the Dendrogram
plt.figure(figsize=(10, 7))

# - dendrogram() creates a visual representation of the hierarchy
# - Z contains the linkage matrix representing hierarchical clustering
dendrogram(Z, labels=np.arange(1, len(df) + 1))

plt.title("Hierarchical Clustering Dendrogram (Complete Linkage)")
plt.xlabel("Data Point Index")
plt.ylabel("Distance")
plt.grid(True)
plt.show()
