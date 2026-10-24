from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_validate,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from features import load_segments, SR, DURATION, HOP


RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5


def make_model():
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# WAVファイルごとにデータを読み込む。
X, labels, groups = [], [], []
for label, folder in ((0, "normal"), (1, "abnormal")):
    paths = sorted(Path("data", folder).glob("*.wav"))
    if not paths:
        raise ValueError(f"data/{folder} にWAVファイルがありません")

    for path in paths:
        segments = load_segments(str(path))
        if not segments:
            print(f"スキップ（2秒未満）: {path}")
            continue

        for _, features in segments:
            X.append(features)
            labels.append(label)
            groups.append(str(path))

X = np.asarray(X)
y = np.asarray(labels)
groups = np.asarray(groups)

print(f"全データ読み込み完了: 正常({np.sum(y == 0)}), 異常({np.sum(y == 1)})")

if len(np.unique(y)) != 2:
    raise ValueError("正常音と異常音の両方に2秒以上のWAVが必要です")

# ファイル単位で層化したホールドアウト分割を行う。
file_labels = {}
for group, label in zip(groups, y):
    file_labels[group] = int(label)

files = np.asarray(list(file_labels.keys()))
file_y = np.asarray([file_labels[f] for f in files])
class_file_counts = np.bincount(file_y, minlength=2)
if np.min(class_file_counts) < 2:
    raise ValueError(
        "学習・評価分割には正常音・異常音それぞれ2個以上のWAVファイルが必要です。"
    )

train_files, test_files = train_test_split(
    files,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=file_y,
)

train_mask = np.isin(groups, train_files)
test_mask = np.isin(groups, test_files)
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
groups_train = groups[train_mask]

print(f"学習区間数: {len(X_train)}（正常: {np.sum(y_train == 0)}, 異常: {np.sum(y_train == 1)}）")
print(f"評価区間数: {len(X_test)}（正常: {np.sum(y_test == 0)}, 異常: {np.sum(y_test == 1)}）")

# 学習データ内で交差検証する。
train_file_labels = {
    f: file_labels[f] for f in train_files
}
train_file_y = np.asarray([train_file_labels[f] for f in train_files])
min_train_files_per_class = int(
    np.min(np.bincount(train_file_y, minlength=2))
)
n_splits = min(N_SPLITS, min_train_files_per_class)

if n_splits >= 2:
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores = cross_validate(
        make_model(),
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
        n_jobs=1,
        error_score="raise",
    )
    print(f"\n--- {n_splits}分割 交差検証（学習データ内、WAV単位） ---")
    for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
        values = scores[f"test_{metric}"]
        print(f"{metric}: {values.mean():.4f} ± {values.std():.4f}")
else:
    print(
        "\n警告: 学習データ内の正常音または異常音のWAVが1ファイルしかないため、"
        "交差検証を実行できません。"
    )

# 最終モデルの学習と予測
model = make_model()
model.fit(X_train, y_train)

# 予測確率を取得
y_proba = model.predict_proba(X_test)

# ★ポイント: デフォルトの0.5ではなく、閾値を調整して異常検出しやすくする
threshold = 0.35
y_pred = (y_proba[:, 1] >= threshold).astype(int)

print(f"\n--- ホールドアウト評価（評価データ / 閾値={threshold}） ---")
print(f"accuracy:          {accuracy_score(y_test, y_pred):.4f}")
print(f"balanced_accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"precision:         {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"recall:            {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"f1:                {f1_score(y_test, y_pred, zero_division=0):.4f}")
print("混同行列（行=正解、列=予測、ラベル順=[正常, 異常]）:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["正常", "異常"],
        zero_division=0,
    )
)

joblib.dump(
    {
        "model": model,
        "sr": SR,
        "duration": DURATION,
        "hop": HOP,
    },
    "model.joblib",
)
print("\n学習完了: model.joblib を保存しました")
