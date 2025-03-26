import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Example 2x2 matrix
data = np.array([[1, 2], [2, 4], [4, 6], [6, 8]])
labels = np.array([0, 0, 1, 1])  # Two classes (0 and 1)

# Apply LDA using sklearn
lda = LDA(n_components=1)
#lda.fit_transform() computes LDA by maximizing the separation between 
# the two classes and reducing dimensions.
transformed_data = lda.fit_transform(data, labels)

print("LDA Transformed Data:\n", transformed_data)
