import numpy as np
import librosa

SR = 22050
DURATION = 2.0
HOP = 1.0


def _mean_std(features):
    """各特徴量のフレーム方向の平均と標準偏差を連結する。"""
    features = np.asarray(features)
    if features.ndim == 1:
        features = features[np.newaxis, :]
    return np.concatenate(
        [np.mean(features, axis=1), np.std(features, axis=1)]
    )


def extract_features(y, sr=SR):
    """
    2秒程度の音声区間から音量・音色・周波数変化の特徴量を抽出する。

    既存特徴量:
      - MFCC (20係数)
      - スペクトル重心
      - ゼロ交差率
      - RMS

    追加特徴量:
      - スペクトル帯域幅
      - スペクトルロールオフ
      - スペクトルフラットネス
      - オンセット強度（時間的な変化の目安）

    各特徴量についてフレーム方向の平均と標準偏差を使用する。
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError("空の音声データは特徴量を抽出できません")

    # 既存の特徴量
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    # 追加の特徴量
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    flatness = librosa.feature.spectral_flatness(y=y)

    # オンセット強度は、短い立ち上がりや音の変化の目安になる
    onset = librosa.onset.onset_strength(y=y, sr=sr)

    feature_groups = (
        mfcc,
        centroid,
        zcr,
        rms,
        bandwidth,
        rolloff,
        flatness,
        onset,
    )

    result = np.concatenate([_mean_std(item) for item in feature_groups])
    return result.astype(np.float32)


def segment_audio(y, sr=SR):
    size = int(DURATION * sr)
    hop = int(HOP * sr)

    if len(y) < size:
        return []

    starts = list(range(0, len(y) - size + 1, hop))
    last = len(y) - size
    if starts[-1] != last:
        starts.append(last)

    return [
        (start / sr, extract_features(y[start:start + size], sr))
        for start in starts
    ]


def load_segments(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    return segment_audio(y, sr)
