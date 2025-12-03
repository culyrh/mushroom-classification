import csv
import math
import argparse
from collections import Counter

# =========================================================
# K-Nearest Neighbors Classifier
# Accuracy (k=5): 0.8819532004781476 
# =========================================================


# 1. CSV 로더 (세미콜론)
def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        data = list(reader)
    header = data[0]
    rows = data[1:]
    return header, rows


# 2. 전처리
def preprocess_dataset(train_header, train_rows, test_header, test_rows):
    label_index = 0

    # --- Train 분리 ---
    X_train_raw, y_train_raw = [], []
    for row in train_rows:
        y_train_raw.append(row[label_index])
        X_train_raw.append(row[1:])

    # --- Test 분리 ---
    X_test_raw, y_test_raw = [], []
    for row in test_rows:
        y_test_raw.append(row[label_index])
        X_test_raw.append(row[1:])

    # Label 인코딩
    label_map = {"p": 0, "e": 1}
    y_train = [label_map[v] for v in y_train_raw]
    y_test = [label_map[v] for v in y_test_raw]

    # === 범주형 인코딩 ===
    num_features = len(X_train_raw[0])
    feature_maps = []

    for col in range(num_features):
        values = [
            row[col] if row[col] != "" else "<EMPTY>"
            for row in X_train_raw
        ]
        uniq = sorted(set(values))
        mapping = {v: i for i, v in enumerate(uniq)}
        feature_maps.append(mapping)

    # Train 인코딩
    X_train = []
    for row in X_train_raw:
        enc = []
        for col in range(num_features):
            val = row[col] if row[col] != "" else "<EMPTY>"
            enc.append(feature_maps[col][val])
        X_train.append(enc)

    # Test 인코딩
    X_test = []
    for row in X_test_raw:
        enc = []
        for col in range(num_features):
            val = row[col] if row[col] != "" else "<EMPTY>"
            mapping = feature_maps[col]
            enc.append(mapping[val] if val in mapping else -1)
        X_test.append(enc)

    return X_train, y_train, X_test, y_test


# 3. 평가 지표
def accuracy(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


# 4. KNN 구현
class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # 유클리디안 거리
    def _distance(self, x1, x2):
        s = 0
        for a, b in zip(x1, x2):
            d = (a - b)
            s += d * d
        return math.sqrt(s)

    def predict(self, X):
        preds = []
        for x in X:
            # 모든 거리 계산
            distances = []
            for train_x, label in zip(self.X_train, self.y_train):
                dist = self._distance(x, train_x)
                distances.append((dist, label))

            # 거리순 정렬
            distances.sort(key=lambda z: z[0])

            # K개 선택
            k_nearest = distances[:self.k]

            # 다수결
            labels = [label for _, label in k_nearest]
            pred = Counter(labels).most_common(1)[0][0]
            preds.append(pred)

        return preds


# 5. 실행부
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    print("[INFO] Loading CSV...")
    train_header, train_rows = load_csv(args.train)
    test_header, test_rows = load_csv(args.test)

    print("[INFO] Preprocessing...")
    X_train, y_train, X_test, y_test = preprocess_dataset(
        train_header, train_rows,
        test_header, test_rows
    )

    print(f"[INFO] Training KNN (k={args.k})")
    model = KNNClassifier(k=args.k)
    model.fit(X_train, y_train)

    print("[INFO] Predicting...")
    y_pred = model.predict(X_test)

    acc = accuracy(y_test, y_pred)
    print(f"KNN Accuracy (k={args.k}): {acc}")
