import csv
import math
import argparse
from collections import defaultdict, Counter

# ========================================================
# Naive Bayes Classifier
# Accuracy: 0.803926705857309
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

    # train
    X_train_raw, y_train_raw = [], []
    for row in train_rows:
        y_train_raw.append(row[label_index])
        X_train_raw.append(row[1:])

    # test
    X_test_raw, y_test_raw = [], []
    for row in test_rows:
        y_test_raw.append(row[label_index])
        X_test_raw.append(row[1:])

    label_map = {"p": 0, "e": 1}
    y_train = [label_map[v] for v in y_train_raw]
    y_test = [label_map[v] for v in y_test_raw]

    # 범주형 인코딩
    num_features = len(X_train_raw[0])
    feature_maps = []

    for col in range(num_features):
        values = [(row[col] if row[col] != "" else "<EMPTY>") for row in X_train_raw]
        uniq = sorted(set(values))
        mapping = {v: i for i, v in enumerate(uniq)}
        feature_maps.append(mapping)

    # train 인코딩
    X_train = []
    for row in X_train_raw:
        enc = []
        for col in range(num_features):
            v = row[col] if row[col] != "" else "<EMPTY>"
            enc.append(feature_maps[col][v])
        X_train.append(enc)

    # test 인코딩
    X_test = []
    for row in X_test_raw:
        enc = []
        for col in range(num_features):
            v = row[col] if row[col] != "" else "<EMPTY>"
            mapping = feature_maps[col]
            enc.append(mapping[v] if v in mapping else -1)
        X_test.append(enc)

    return X_train, y_train, X_test, y_test


# Accuracy
def accuracy(y_true, y_pred):
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


# Naive Bayes Classifier
class NaiveBayesClassifier:
    def __init__(self):
        self.class_counts = Counter()
        self.feature_counts = {}  # feature_counts[class][col][value]
        self.num_features = 0

    def fit(self, X, y):
        self.num_features = len(X[0])

        # 초기화
        self.feature_counts = {
            0: [Counter() for _ in range(self.num_features)],
            1: [Counter() for _ in range(self.num_features)]
        }

        # 학습
        for row, label in zip(X, y):
            self.class_counts[label] += 1
            for col in range(self.num_features):
                self.feature_counts[label][col][row[col]] += 1

        self.total = len(y)

    def predict(self, X):
        preds = []
        for row in X:
            log_prob = {0: 0.0, 1: 0.0}
            for c in (0, 1):
                # P(class)
                log_prob[c] += math.log((self.class_counts[c] + 1) / (self.total + 2))

                # P(x_i | class)
                for col in range(self.num_features):
                    val = row[col]
                    count_val = self.feature_counts[c][col][val]
                    count_total = sum(self.feature_counts[c][col].values())
                    log_prob[c] += math.log((count_val + 1) / (count_total + len(self.feature_counts[c][col])))

            # argmax
            preds.append(1 if log_prob[1] > log_prob[0] else 0)
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
        train_header, train_rows,
        test_header, test_rows
    )

    print("[INFO] Training Naive Bayes...")
    model = NaiveBayesClassifier()
    model.fit(X_train, y_train)

    print("[INFO] Predicting...")
    y_pred = model.predict(X_test)

    acc = accuracy(y_test, y_pred)
    print("Naive Bayes Accuracy:", acc)
