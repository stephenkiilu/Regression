import math

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        for _ in range(self.epochs):
            for idx in range(n_samples):
                linear = sum(self.weights[j] * X[idx][j] for j in range(n_features)) + self.bias
                y_pred = self._sigmoid(linear)
                error = y_pred - y[idx]
                for j in range(n_features):
                    self.weights[j] -= self.lr * error * X[idx][j]
                self.bias -= self.lr * error

    def predict_proba(self, X):
        probs = []
        for x in X:
            linear = sum(w * xj for w, xj in zip(self.weights, x)) + self.bias
            probs.append(self._sigmoid(linear))
        return probs

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

if __name__ == "__main__":
    # Example usage with OR function
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 1]
    model = LogisticRegression(lr=0.1, epochs=1000)
    model.fit(X, y)
    preds = model.predict(X)
    print("Predictions:", preds)

