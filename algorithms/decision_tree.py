import csv
import math
import argparse
from collections import defaultdict, Counter

# =========================================================
# Decision Tree Classifier
# Accuracy: 0.5549132947976878
# =========================================================


# 1. CSV 로더
def load_csv(path):
    """
    CSV 파일을 세미콜론(;) delimiter로 읽어서 반환.
    """
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        data = list(reader)

    header = data[0]
    rows = data[1:]
    return header, rows


# 2. 전처리
def preprocess_dataset(train_header, train_rows, test_header, test_rows):
    """
    - class: p/e → 0/1
    - 나머지는 전부 문자열 feature → 숫자 인코딩
    - 빈 문자열("")은 별도의 카테고리로 취급
    """

    label_index = 0  # 첫 컬럼이 class

    # ★ train X, y 분리
    X_train_raw, y_train_raw = [], []
    for row in train_rows:
        y_train_raw.append(row[label_index])
        X_train_raw.append(row[1:])  # class 뒤의 20개 feature

    # ★ test X, y 분리
    X_test_raw, y_test_raw = [], []
    for row in test_rows:
        y_test_raw.append(row[label_index])
        X_test_raw.append(row[1:])

    # === 라벨 인코딩 (p/e → 0/1)
    label_map = {"p": 0, "e": 1}
    y_train = [label_map[v] for v in y_train_raw]
    y_test = [label_map[v] for v in y_test_raw]

    # === 범주형 인코딩 ===
    num_features = len(X_train_raw[0])
    feature_maps = []

    for col in range(num_features):
        # 빈값("")도 하나의 카테고리로 포함
        values = [row[col] if row[col] != "" else "<EMPTY>" for row in X_train_raw]
        uniq = sorted(set(values))
        mapping = {v: i for i, v in enumerate(uniq)}
        feature_maps.append(mapping)

    # train 인코딩
    X_train = []
    for row in X_train_raw:
        enc = []
        for col in range(num_features):
            val = row[col] if row[col] != "" else "<EMPTY>"
            enc.append(feature_maps[col][val])
        X_train.append(enc)

    # test 인코딩 (train에 없는 카테고리 = -1)
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


# 4. Decision Tree (Gini 기반)
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    # ------------------------------
    # 트리 학습
    # ------------------------------
    def fit(self, X, y):
        data = [x + [y_i] for x, y_i in zip(X, y)]
        self.tree = self._build_tree(data, depth=0)

    # ------------------------------
    # 예측
    # ------------------------------
    def predict(self, X):
        preds = []
        for row in X:
            preds.append(self._predict_row(self.tree, row))
        return preds

    # ------------------------------
    # 트리 빌드 (재귀)
    # ------------------------------
    def _build_tree(self, data, depth):
        labels = [row[-1] for row in data]

        # 1) 모든 라벨이 같음 → 리프
        if len(set(labels)) == 1:
            return labels[0]

        # 2) depth 제한
        if self.max_depth is not None and depth >= self.max_depth:
            return Counter(labels).most_common(1)[0][0]

        # 3) best split
        best_col, best_gini, best_groups = self._find_best_split(data)
        if best_col is None:
            return Counter(labels).most_common(1)[0][0]

        left, right = best_groups

        return {
            "feature": best_col,
            "left": self._build_tree(left, depth + 1),
            "right": self._build_tree(right, depth + 1),
        }

    # ------------------------------
    # split 기준 찾기
    # ------------------------------
    def _find_best_split(self, data):
        best_col = None
        best_gini = float('inf')
        best_groups = None

        n_features = len(data[0]) - 1

        for col in range(n_features):
            values = set([row[col] for row in data])
            for val in values:
                left = [row for row in data if row[col] == val]
                right = [row for row in data if row[col] != val]

                if len(left) == 0 or len(right) == 0:
                    continue

                gini = self._gini([left, right])

                if gini < best_gini:
                    best_col = col
                    best_gini = gini
                    best_groups = (left, right)

        return best_col, best_gini, best_groups

    def _gini(self, groups):
        total = sum(len(g) for g in groups)
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            labels = [row[-1] for row in group]
            score = 0
            for c in set(labels):
                p = labels.count(c) / size
                score += p * p
            gini += (1 - score) * (size / total)
        return gini

    # ------------------------------
    # 트리 탐색
    # ------------------------------
    def _predict_row(self, node, row):
        if not isinstance(node, dict):
            return node
        col = node["feature"]

        # 단순 분기: row[col] 값이 split 기준이었다면 left
        return (
            self._predict_row(node["left"], row)
            if row[col] == row[col]
            else self._predict_row(node["right"], row)
        )


# 5. 실행
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
        train_header, train_rows,
        test_header, test_rows
    )

    print("[INFO] Training Decision Tree...")
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)

    print("[INFO] Predicting...")
    y_pred = model.predict(X_test)

    acc = accuracy(y_test, y_pred)
    print("Decision Tree Accuracy:", acc)
