import numpy as np
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris()
X = data.data[:, :2]  # Using only Sepal Length and Sepal Width
y = data.target

# Convert to binary classification (-1 for Setosa, 1 for Non-Setosa)
y = np.where(y == 0, -1, 1)

# Support Vector Machine Class
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initialize the SVM model with hyperparameters.

        Parameters:
        - learning_rate: Step size for gradient descent.
        - lambda_param: Regularization parameter to prevent overfitting.
        - n_iters: Number of iterations for training.
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Train the SVM using Stochastic Gradient Descent.

        Parameters:
        - X: Feature matrix (samples, features)
        - y: Target vector (binary labels: -1 or 1)
        """
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # Ensure labels are in -1 and 1

        # Initialize weights and bias to zero
        self.w = np.zeros(n_features)
        self.b = 0

        # Training loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Check classification using hinge loss condition
                # Condition: y_i * (w·x_i - b) >= 1
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # Correct classification → Only apply regularization
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Misclassified → Update weights and bias
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        """
        Predict labels using the trained model.

        Parameters:
        - X: Feature matrix for prediction

        Returns:
        - Predicted labels (-1 or 1)
        """
        # Calculate the linear function: w·x - b
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# Initialize and train the model
svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X, y)

# Predict and print results
predictions = svm.predict(X)
print("Predicted Labels:", predictions)

# Calculate accuracy
accuracy = np.sum(predictions == y) / len(y) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Display weights and bias
print("Weights (w):", svm.w)
print("Bias (b):", svm.b)




