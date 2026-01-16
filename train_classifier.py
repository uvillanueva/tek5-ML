import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


DATA_PATH = "mental_health_dataset.csv"
TARGET = "mental_health_risk"

TEST_SIZE = 0.25
RANDOM_STATE = 7


def present_basic_information(df):
    print("Number of features:", df.shape[1] - 1)
    print("Number of samples:", df.shape[0])
    print("Target variable:", TARGET)
    print()


def preprocess_fit_transform(X_train: pd.DataFrame):
    X = X_train.copy()

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_medians = {}
    for c in num_cols:
        col = pd.to_numeric(X[c], errors="coerce")
        med = col.median()
        med = float(med) if pd.notna(med) else 0.0
        num_medians[c] = med
        X[c] = col.fillna(med).astype(float)

    cat_modes = {}
    for c in cat_cols:
        s = X[c].astype(str)
        mode = s.mode(dropna=True)
        m = str(mode.iloc[0]) if len(mode) else ""
        cat_modes[c] = m
        X[c] = s.fillna(m)

    X_num = X[num_cols].astype(float) if num_cols else pd.DataFrame(index=X.index)
    X_cat = pd.get_dummies(X[cat_cols], dummy_na=False) if cat_cols else pd.DataFrame(index=X.index)

    means, stds = {}, {}
    for c in num_cols:
        m = float(X_num[c].mean())
        s = float(X_num[c].std(ddof=0))
        if s == 0.0:
            s = 1.0
        means[c] = m
        stds[c] = s
        X_num[c] = (X_num[c] - m) / s

    X_proc = pd.concat([X_num, X_cat], axis=1)
    features = X_proc.columns.tolist()

    state = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_medians": num_medians,
        "cat_modes": cat_modes,
        "means": means,
        "stds": stds,
        "features": features,
    }

    return X_proc.to_numpy(dtype=np.float64), state


def preprocess_transform(X_test: pd.DataFrame, state):
    X = X_test.copy()

    for c in state["num_cols"]:
        col = pd.to_numeric(X[c], errors="coerce").fillna(state["num_medians"][c]).astype(float)
        X[c] = (col - state["means"][c]) / state["stds"][c]

    for c in state["cat_cols"]:
        X[c] = X[c].astype(str).fillna(state["cat_modes"][c])

    X_num = X[state["num_cols"]].astype(float) if state["num_cols"] else pd.DataFrame(index=X.index)
    X_cat = pd.get_dummies(X[state["cat_cols"]], dummy_na=False) if state["cat_cols"] else pd.DataFrame(index=X.index)

    X_proc = pd.concat([X_num, X_cat], axis=1)
    X_proc = X_proc.reindex(columns=state["features"], fill_value=0.0)

    return X_proc.to_numpy(dtype=np.float64)


def encode_labels(y_train, y_test):
    classes = sorted(pd.Series(y_train).astype(str).unique().tolist())
    mapping = {c: i for i, c in enumerate(classes)}
    y_train_enc = pd.Series(y_train).astype(str).map(mapping).to_numpy(dtype=np.int64)
    y_test_enc = pd.Series(y_test).astype(str).map(mapping).to_numpy(dtype=np.int64)
    return y_train_enc, y_test_enc, classes


class MACHINE_LEARNING_TEK5_MODEL:
    def __init__(self, lr=0.15, epochs=1200, l2=1e-3, batch_size=512, seed=7):
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.W = None
        self.b = None

    def _softmax(self, z):
        z = np.asarray(z, dtype=np.float64)
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        k = int(np.max(y) + 1)

        self.W = rng.normal(0, 0.01, size=(d, k)).astype(np.float64)
        self.b = np.zeros((1, k), dtype=np.float64)

        Y = np.eye(k, dtype=np.float64)[y]

        for _ in range(self.epochs):
            idx = rng.permutation(n)
            for i in range(0, n, self.batch_size):
                xb = X[idx[i:i + self.batch_size]]
                yb = Y[idx[i:i + self.batch_size]]

                probs = self._softmax(xb @ self.W + self.b)
                grad = (probs - yb) / xb.shape[0]

                self.W -= self.lr * (xb.T @ grad + self.l2 * self.W)
                self.b -= self.lr * np.sum(grad, axis=0, keepdims=True)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        probs = self._softmax(X @ self.W + self.b)
        return np.argmax(probs, axis=1).astype(np.int64)


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    present_basic_information(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train, state = preprocess_fit_transform(X_train_df)
    X_test = preprocess_transform(X_test_df, state)

    y_train_enc, y_test_enc, class_names = encode_labels(y_train, y_test)

    model = MACHINE_LEARNING_TEK5_MODEL(seed=RANDOM_STATE)

    t0 = time.perf_counter()
    model.fit(X_train, y_train_enc)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = time.perf_counter() - t1

    acc = accuracy_score(y_test_enc, y_pred)
    cm = confusion_matrix(y_test_enc, y_pred)

    print("Accuracy:", round(float(acc), 3))
    print("Training time (s):", round(float(train_time), 3))
    print("Prediction time (s):", round(float(pred_time), 3))
    print()

    print("Per-class metrics:")
    for i, cls in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        support = int(cm[i, :].sum())

        print(cls)
        print("Precision:", round(float(precision), 3))
        print("Recall:", round(float(recall), 3))
        print("Support:", support)
        print()

    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
