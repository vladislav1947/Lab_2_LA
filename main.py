import numpy as np

#const
ACCURACY = 85
ALPHA = 0.1 # learning rate
EPOCHS = 1000 # learning times

def sampling(m, n):
    target = 0.5
    error = 0.01
    while True:
        X = np.random.rand(m, n)
        true_w = np.random.randn(n)
        true_b = np.random.randn()
        Z = X @ true_w + true_b
        Y = (Z > 0).astype(int)
        if abs(np.mean(Y) - target) < error:
            return X, Y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, w, b):
    Z = X @ w + b
    Y = sigmoid(Z)
    return Y

def train(X, Y, EPOCHS, ALPHA):
    m, n = X.shape
    w = np.random.randn(n)
    b = 0.0

    for epoch in range(EPOCHS):
        Y_new = forward(X, w, b)

        prediction = (Y_new >= 0.5).astype(int)
        accuracy = np.mean(prediction == Y) * 100
        print(f"Epoch {epoch+1}: Accuracy = {accuracy:.2f}%")

        if accuracy >= 85:
            print("Достигнута нужная точность")
            np.savez("model.npz", w=w, b=b)
            break

        error = Y_new - Y
        dw = X.T @ error / m
        db = np.sum(error) / m

        w -= ALPHA * dw
        b -= ALPHA * db

    return w, b

def main():
    m = 1_000_000
    n = 30
    X, Y = sampling(m, n)

    w, b = train(X, Y, EPOCHS, ALPHA)

if "__name__" == "__main__":
    main()