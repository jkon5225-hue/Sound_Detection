# streamlit run app.py

import io
from pathlib import Path
import joblib
import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from features import SR, DURATION, HOP, segment_audio

st.set_page_config(page_title='Abnormal Sound Detection AI', page_icon='🎧', layout='wide')
st.title('🎧 Abnormal Sound Detection AI')
st.caption('録音済みWAVファイルからモーターの異常音候補を検出します。')

@st.cache_resource
def load_model(path, mtime):
    obj = joblib.load(path)  # 信頼できる自作モデルのみ読み込む
    if not isinstance(obj, dict) or 'model' not in obj:
        raise ValueError('モデル形式が違います。付属のtrain.pyで再学習してください。')
    if (obj.get('sr'), obj.get('duration'), obj.get('hop')) != (SR, DURATION, HOP):
        raise ValueError('学習時と解析時の音声設定が一致しません。')
    return obj['model']

model_path = Path('model.joblib')
if not model_path.exists():
    st.warning('model.joblib がありません。先に正常音・異常音を用意して python train.py を実行してください。')
    st.stop()

try:
    model = load_model(str(model_path.resolve()), model_path.stat().st_mtime_ns)
except Exception as e:
    st.error(f'モデルの読み込みに失敗しました: {e}')
    st.stop()

# デフォルト値を 0.70 から検証済みの 0.30 に変更
threshold = st.sidebar.slider('異常判定しきい値', 0.0, 1.0, 0.35, 0.01)

upload = st.file_uploader('解析するWAVファイルを選択', type=['wav'])
if upload is None:
    st.info('WAVファイルをアップロードしてください。')
    st.stop()

raw = upload.getvalue()
st.audio(raw, format='audio/wav')
try:
    audio, original_sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
    if audio.shape[0] == 0 or original_sr <= 0:
        raise ValueError('空の音声です')
    if not np.isfinite(audio).all():
        raise ValueError('音声に不正な値が含まれています')
    y = audio.mean(axis=1)
    y = librosa.resample(y, orig_sr=original_sr, target_sr=SR) if original_sr != SR else y
    if len(y) < int(DURATION * SR):
        raise ValueError('2秒以上の録音ファイルを指定してください')
except Exception as e:
    st.error(f'WAV読み込みエラー: {e}')
    st.stop()

st.write(f'録音時間: **{len(audio) / original_sr:.2f}秒** / 元のサンプリング周波数: **{original_sr} Hz**')
segments = segment_audio(y)
X = np.stack([feature for _, feature in segments])
if 1 not in model.classes_:
    st.error('モデルに異常音クラスがありません。')
    st.stop()
scores = model.predict_proba(X)[:, list(model.classes_).index(1)]
df = pd.DataFrame({
    '開始秒': [round(start, 3) for start, _ in segments],
    '終了秒': [round(start + DURATION, 3) for start, _ in segments],
    '異常スコア': scores.round(4),
})
df['判定'] = np.where(df['異常スコア'] >= threshold, '異常候補', '正常候補')

c1, c2, c3 = st.columns(3)
c1.metric('最大異常スコア', f'{scores.max():.3f}')
c2.metric('異常候補区間数', f'{int((scores >= threshold).sum())} / {len(scores)}')
c3.metric('総合判定', '異常候補あり' if (scores >= threshold).any() else '異常候補なし')

st.subheader('異常スコアの時系列')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['開始秒'], y=df['異常スコア'], mode='lines+markers', name='異常スコア'))
fig.add_hline(y=threshold, line_dash='dash', annotation_text='判定しきい値')
fig.update_layout(xaxis_title='解析窓の開始時刻（秒）', yaxis_title='異常スコア', yaxis_range=[0, 1])
st.plotly_chart(fig, width='stretch')

st.subheader('音声波形')
stride = max(1, len(y) // 15000)
times = np.arange(0, len(y), stride) / SR
wave = go.Figure(go.Scatter(x=times, y=y[::stride], mode='lines', name='波形'))
wave.update_layout(xaxis_title='時間（秒）', yaxis_title='振幅')
st.plotly_chart(wave, width='stretch')

st.subheader('メルスペクトログラム')
mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64)
mel_db = librosa.power_to_db(mel, ref=np.max)
mel_time = librosa.times_like(mel_db, sr=SR)
heat = go.Figure(go.Heatmap(x=mel_time, y=librosa.mel_frequencies(n_mels=64, fmax=SR / 2), z=mel_db, colorbar_title='dB'))
heat.update_layout(xaxis_title='時間（秒）', yaxis_title='周波数（Hz)')
st.plotly_chart(heat, width='stretch')

st.subheader('異常候補区間')
alerts = df[df['判定'] == '異常候補']
if alerts.empty:
    st.success('設定したしきい値を超える区間はありませんでした。')
else:
    st.dataframe(alerts, width='stretch', hide_index=True)
with st.expander('全区間の判定結果'):
    st.dataframe(df, width='stretch', hide_index=True)
st.download_button('CSVをダウンロード', data=df.to_csv(index=False).encode('utf-8-sig'), file_name='motor_sound_result.csv', mime='text/csv')
st.caption('スコアは故障確率ではありません。回転数・負荷・録音環境を揃え、独立した録音データで精度を検証してください。')
