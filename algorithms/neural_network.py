import csv
import math
import random
import argparse

# =========================================================
# Neural Network Classifier (MLP)
# Accuracy: 0.5549132947976878
# =========================================================


# CSV Loader
def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        data = list(reader)
    return data[0], data[1:]


# Preprocessing + Normalize (0~1)
def preprocess_dataset(train_header, train_rows, test_header, test_rows):
    label_index = 0

    X_train_raw, y_train_raw = [], []
    for row in train_rows:
        y_train_raw.append(row[label_index])
        X_train_raw.append(row[1:])

    X_test_raw, y_test_raw = [], []
    for row in test_rows:
        y_test_raw.append(row[label_index])
        X_test_raw.append(row[1:])

    label_map = {"p": 0, "e": 1}
    y_train = [label_map[v] for v in y_train_raw]
    y_test = [label_map[v] for v in y_test_raw]

    num_features = len(X_train_raw[0])
    feature_maps = []

    # categorical → numeric
    for col in range(num_features):
        vals = [(r[col] if r[col] != "" else "<EMPTY>") for r in X_train_raw]
        uniq = sorted(set(vals))
        mapping = {v: i for i, v in enumerate(uniq)}
        feature_maps.append(mapping)

    # encode
    X_train = []
    for row in X_train_raw:
        X_train.append([
            feature_maps[col][row[col] if row[col] != "" else "<EMPTY>"]
            for col in range(num_features)
        ])

    X_test = []
    for row in X_test_raw:
        X_test.append([
            feature_maps[col].get(row[col] if row[col] != "" else "<EMPTY>", -1)
            for col in range(num_features)
        ])

    # ===== Normalize 0~1 =====
    mins = [min(col) for col in zip(*X_train)]
    maxs = [max(col) for col in zip(*X_train)]

    for i in range(len(X_train)):
        X_train[i] = [
            (x - mins[j]) / (maxs[j] - mins[j] + 1e-9)
            for j, x in enumerate(X_train[i])
        ]

    for i in range(len(X_test)):
        X_test[i] = [
            (x - mins[j]) / (maxs[j] - mins[j] + 1e-9)
            for j, x in enumerate(X_test[i])
        ]

    return X_train, y_train, X_test, y_test


def accuracy(y_true, y_pred):
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


# MLP (1 Hidden Layer)
class MLP:
    def __init__(self, input_dim, hidden_dim=10, lr=0.0005):
        self.lr = lr

        # 더 작은 초기화
        self.W1 = [[(random.random() - 0.5) * 0.01 for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim

        self.W2 = [(random.random() - 0.5) * 0.01 for _ in range(hidden_dim)]
        self.b2 = 0.0

    def relu(self, x): return max(0, x)

    # 안정 sigmoid
    def sigmoid(self, x):
        if x < -100:
            return 0.0
        if x > 100:
            return 1.0
        return 1 / (1 + math.exp(-x))

    def forward(self, x):
        self.h_raw = [sum(w_i * x_i for w_i, x_i in zip(w, x)) + b for w, b in zip(self.W1, self.b1)]
        self.h = [self.relu(v) for v in self.h_raw]

        o_raw = sum(w * h for w, h in zip(self.W2, self.h)) + self.b2
        return self.sigmoid(o_raw)

    def backward(self, x, y, out):
        d_out = (out - y)

        for i in range(len(self.W2)):
            self.W2[i] -= self.lr * d_out * self.h[i]
        self.b2 -= self.lr * d_out

        d_h = []
        for i in range(len(self.h)):
            grad = d_out * self.W2[i]
            if self.h_raw[i] <= 0: grad = 0
            d_h.append(grad)

        for i in range(len(self.W1)):
            for j in range(len(self.W1[0])):
                self.W1[i][j] -= self.lr * d_h[i] * x[j]
            self.b1[i] -= self.lr * d_h[i]

    def predict(self, X):
        return [1 if self.forward(x) >= 0.5 else 0 for x in X]

    def fit(self, X, y, epochs=5):
        for e in range(epochs):
            for x_i, y_i in zip(X, y):
                out = self.forward(x_i)
                self.backward(x_i, y_i, out)
            print(f"[INFO] Epoch {e+1} completed")


# 실행부
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()

    print("[INFO] Loading CSV...")
    train_header, train_rows = load_csv(args.train)
    test_header, test_rows = load_csv(args.test)

    print("[INFO] Preprocessing...")
    X_train, y_train, X_test, y_test = preprocess_dataset(
        train_header, train_rows, test_header, test_rows
    )

    print("[INFO] Training MLP...")
    model = MLP(input_dim=len(X_train[0]), hidden_dim=10, lr=0.0005)
    model.fit(X_train, y_train, epochs=5)

    print("[INFO] Predicting...")
    y_pred = model.predict(X_test)

    acc = accuracy(y_test, y_pred)
    print("Neural Network Accuracy:", acc)
