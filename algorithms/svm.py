import csv
import math
import random
import argparse

# ========================================================
# Support Vector Machine Classifier
# Accuracy: 0.5417314840590152
# =========================================================


# CSV 로더
def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        data = list(reader)
    return data[0], data[1:]


# 전처리
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

    # p=0, e=1 → SVM 에서는 -1, +1 로 변환
    label_map = {"p": -1, "e": 1}
    y_train = [label_map[v] for v in y_train_raw]
    y_test = [label_map[v] for v in y_test_raw]

    num_features = len(X_train_raw[0])
    feature_maps = []

    for col in range(num_features):
        vals = [(r[col] if r[col] != "" else "<EMPTY>") for r in X_train_raw]
        uniq = sorted(set(vals))
        mapping = {v: i for i, v in enumerate(uniq)}
        feature_maps.append(mapping)

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

    return X_train, y_train, X_test, y_test


# Accuracy
def accuracy(y_true, y_pred):
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


# Linear SVM (Hinge Loss + SGD)
class LinearSVM:
    def __init__(self, input_dim, lr=0.0001, C=1.0, epochs=5):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.w = [(random.random() - 0.5) * 0.01 for _ in range(input_dim)]
        self.b = 0.0

    def fit(self, X, y):
        for ep in range(self.epochs):
            for x_i, y_i in zip(X, y):
                margin = y_i * (sum(w_j * x_j for w_j, x_j in zip(self.w, x_i)) + self.b)
                if margin >= 1:
                    # only regularization term
                    for j in range(len(self.w)):
                        self.w[j] -= self.lr * (2 * self.w[j])
                else:
                    # hinge loss update
                    for j in range(len(self.w)):
                        self.w[j] -= self.lr * (2 * self.w[j] - self.C * y_i * x_i[j])
                    self.b += self.lr * self.C * y_i
            print(f"[INFO] Epoch {ep+1} done")

    def predict(self, X):
        preds = []
        for x in X:
            v = sum(w_j * x_j for w_j, x_j in zip(self.w, x)) + self.b
            preds.append(1 if v >= 0 else -1)
        return preds


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

    print("[INFO] Training SVM...")
    model = LinearSVM(input_dim=len(X_train[0]), lr=0.0001, C=1.0, epochs=5)
    model.fit(X_train, y_train)

    print("[INFO] Predicting...")
    y_pred = model.predict(X_test)

    # convert back to 0/1 for accuracy
    y_test_binary = [1 if yy == 1 else 0 for yy in y_test]
    y_pred_binary = [1 if yy == 1 else 0 for yy in y_pred]

    acc = accuracy(y_test_binary, y_pred_binary)
    print("SVM Accuracy:", acc)
