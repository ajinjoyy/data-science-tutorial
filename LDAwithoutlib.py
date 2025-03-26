import numpy as np

# Step 1: Define the data (2x2 matrix) and class labels
data = np.array([[1, 2], [2, 4], [4, 6], [6, 8]])
labels = np.array([0, 0, 1, 1])  # Two classes: 0 and 1

# Separate the data by class
class_0 = data[labels == 0]
class_1 = data[labels == 1]

# Step 2: Calculate the class means
mean_0 = np.mean(class_0, axis=0)
mean_1 = np.mean(class_1, axis=0)
overall_mean = np.mean(data, axis=0)

print("Class 0 Mean:", mean_0)
print("Class 1 Mean:", mean_1)
print("Overall Mean:", overall_mean)

# Step 3: Compute the Within-Class Scatter Matrix (S_W)
S_W = np.zeros((2, 2))

for x in class_0:
    diff = (x - mean_0).reshape(2, 1)
    S_W += np.dot(diff, diff.T)

for x in class_1:
    diff = (x - mean_1).reshape(2, 1)
    S_W += np.dot(diff, diff.T)

print("\nWithin-Class Scatter Matrix (S_W):\n", S_W)

# Step 4: Compute the Between-Class Scatter Matrix (S_B)
mean_diff = (mean_0 - mean_1).reshape(2, 1)
S_B = np.dot(mean_diff, mean_diff.T)

print("\nBetween-Class Scatter Matrix (S_B):\n", S_B)

# Step 5: Calculate the LDA projection vector
# Using the formula: inv(S_W) * S_B
S_W_inv = np.linalg.inv(S_W)
eig_values, eig_vectors = np.linalg.eig(np.dot(S_W_inv, S_B))

# Select the eigenvector corresponding to the largest eigenvalue
lda_vector = eig_vectors[:, np.argmax(eig_values)]
print("\nLDA Projection Vector (Direction):", lda_vector)

# Step 6: Project the data onto the LDA vector
projected_data = np.dot(data, lda_vector)
print("\nProjected Data (1D representation):\n", projected_data)
