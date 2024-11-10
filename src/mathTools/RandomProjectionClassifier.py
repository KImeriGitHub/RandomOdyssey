import numpy as np
from math import sqrt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class RandomProjectionClassifier:
    def __init__(self, num_random_features=2000, regularization=30, max_iter=10, verbose=True, random_state=None):
        """
        Initializes the RandomProjectionClassifier.

        Parameters:
        - num_random_features: Number of random features (NRnd).
        - regularization: Regularization parameter (g).
        - max_iter: Maximum iterations for regularization tuning.
        - verbose: If True, prints progress during tuning.
        - random_state: Seed for reproducibility.
        """
        self.NRnd = num_random_features
        self.g = regularization
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.W = None
        self.K = None
        self.rhs = None
        self.num_classes = None
        self.best_g = None
        self.best_acc = None

    def one_hot_encode(self, Y, num_classes):
        """
        One-hot encodes the labels.

        Parameters:
        - Y: Array of labels.
        - num_classes: Number of distinct classes.

        Returns:
        - One-hot encoded label matrix.
        """
        Nfeatures = Y.shape[0]
        Y_encoded = np.zeros((Nfeatures, num_classes))
        Y_encoded[np.arange(Nfeatures), Y] = 1
        return Y_encoded

    def fit(self, X_train, Y_train):
        """
        Fits the model to the training data.

        Parameters:
        - X_train: Training data of shape (num_samples, 28, 28).
        - Y_train: Training labels of shape (num_samples,).
        """
        Nfeatures = X_train.shape[0]
        self.num_classes = len(np.unique(Y_train))

        # One-hot encode Y_train
        Ytrain = self.one_hot_encode(Y_train, self.num_classes)

        # Reshape images to vectors and normalize
        U = X_train.reshape((Nfeatures, -1)).astype("float64")

        # Initialize W with complex random values
        rng = np.random.default_rng(self.random_state)
        self.W = (rng.standard_normal((self.NRnd, U.shape[1])) + 
                  1j * rng.standard_normal((self.NRnd, U.shape[1]))) / sqrt(self.NRnd)

        # Compute XMat = |W * U^T|
        XMat = (np.abs(self.W @ U.T)).T  # Shape: (Nfeatures, NRnd)

        # Compute K = XMat^T * XMat
        self.K = XMat.T @ XMat  # Shape: (NRnd, NRnd)

        # Compute rhs = XMat^T * Ytrain
        self.rhs = XMat.T @ Ytrain  # Shape: (NRnd, num_classes)

    def predict(self, X_test):
        """
        Predicts class labels for the test data.

        Parameters:
        - X_test: Test data of shape (num_samples, 28, 28).

        Returns:
        - Predicted labels of shape (num_samples,).
        """
        Nfeatures = X_test.shape[0]

        # Reshape and normalize
        Utest = X_test.reshape((Nfeatures, -1)).astype("float64")

        # Compute XtilMat = |W * Utest^T|
        XtilMat = (np.abs(self.W @ Utest.T)).T  # Shape: (Nfeatures, NRnd)

        # Compute Ytil = XtilMat * (K + gI)^(-1) * rhs
        Ytil = XtilMat @ np.linalg.solve((self.K + self.g * np.eye(self.K.shape[0])), self.rhs)

        # Predictions via argmax
        return np.argmax(Ytil, axis=1)

    def tune_regularization(self, X_val, Y_val, low_start=0.01, high_start=30, max_iter=None):
        """
        Tunes the regularization parameter 'g' to maximize accuracy.

        Parameters:
        - X_val: Validation data of shape (num_samples, 28, 28).
        - Y_val: Validation labels of shape (num_samples,).
        - low: Lower bound for 'g'.
        - high: Upper bound for 'g'.
        - max_iter: Maximum iterations for tuning. If None, uses self.max_iter.
        """
        if max_iter is None:
            max_iter = self.max_iter

        Nfeatures = X_val.shape[0]
        Y_true = Y_val

        # Reshape and normalize
        Utest = X_val.reshape((Nfeatures, -1)).astype("float64")
        XtilMat = (np.abs(self.W @ Utest.T)).T  # Shape: (Nfeatures, NRnd)

        # Initial accuracies at the bounds
        acc_low = self.compute_accuracy(g=low_start, Y_test=Y_true, XtilMat=XtilMat)
        acc_high = self.compute_accuracy(g=high_start, Y_test=Y_true, XtilMat=XtilMat)

        best_acc = acc_low if acc_low > acc_high else acc_high
        best_g = low_start if acc_low > acc_high else high_start

        low, high = low_start, high_start

        for i in range(max_iter):
            g = (low + high) / 2
            acc = self.compute_accuracy(g=g, Y_test=Y_true, XtilMat=XtilMat)

            if acc > best_acc:
                best_acc, best_g = acc, g
        
            if (acc > acc_low) != (acc > acc_high):
                # If acc is greater than one bound but not the other
                if acc > acc_low:
                    low, acc_low = g, acc
                else:
                    high, acc_high = g, acc
            else:
                if acc_low > acc_high:
                    high, acc_high = g, acc
                else:
                    low, acc_low = g, acc

            if self.verbose:
                print(f"Iteration {i+1}: acc = {best_acc:.4f}, g = {best_g:.4f}, low = {low:.4f}, high = {high:.4f}")

        self.best_g = best_g
        self.best_acc = best_acc
        self.g = best_g  # Update the regularization parameter to the best found

        if self.verbose:
            print(f"Best accuracy: {self.best_acc:.4f} with g = {self.best_g:.4f}")

    def compute_accuracy(self, g, Y_test, X_test=None, XtilMat=None):
        """
        Computes the accuracy for a given regularization parameter 'g'.

        Parameters:
        - g: Regularization parameter.
        - Y_true: True labels.
        - X_test: Test data of shape (num_samples, 28, 28). Required if XtilMat is not provided.
        - XtilMat: Precomputed projection matrix for validation data. Optional.

        Returns:
        - Accuracy as a float.
        """
        if XtilMat is not None:
            Ytil = XtilMat @ np.linalg.solve((self.K + g * np.eye(self.K.shape[0])), self.rhs)
        elif X_test is not None:
            Nfeatures = X_test.shape[0]
            Utest = (X_test.reshape((Nfeatures, -1)) / np.max(X_test)).astype("float64")
            XtilMat_computed = (np.abs(self.W @ Utest.T)).T  # Shape: (Nfeatures, NRnd)
            Ytil = XtilMat_computed @ np.linalg.solve((self.K + g * np.eye(self.K.shape[0])), self.rhs)
        else:
            raise ValueError("Either XtilMat or X_test must be provided.")

        return np.mean(np.argmax(Ytil, axis=1) == Y_test)
