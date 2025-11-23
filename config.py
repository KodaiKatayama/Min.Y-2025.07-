"""
config.py
論文 "Structure As Search" TSP-20 再現実験用
完全版ハイパーパラメータ設定
"""

import torch

# ==========================
# 1. 問題設定 (TSP-20)
# ==========================
NUM_NODES = 20          # 都市数 n [cite: 115]
INPUT_DIM = 2           # 座標データの次元 (x, y) [cite: 44]
SEED = 42               # 再現性確保のため


# ==========================
# 2. データセット設定
# ==========================
TRAIN_SIZE = 100000     # 学習データ数 [cite: 117]
VAL_SIZE = 1000         # 検証データ数 [cite: 118]
TEST_SIZE = 1000        # テストデータ数 [cite: 118]
BATCH_SIZE = 128        # (論文非記載のため一般的な値を設定)
DISTRIBUTION = "uniform" # データ分布 [cite: 117]


# ==========================
# 3. モデル構造 (SAGs for TSP-20)
# ==========================
# [cite: 125] に基づく設定
HIDDEN_DIM = 128        # 隠れ層の次元
NUM_LAYERS = 2          # GNNの層数
SCATTERING_CHANNELS = 6 # [cite: 126]
LOWPASS_CHANNELS = 2    # [cite: 126]

# --- 数式パラメータ (論文 Eq.6, Eq.7) ---
# 論文の実験設定に具体的な数値がないため、デフォルト1.0として定義
# 距離行列の変換: A = exp(-D / s) [cite: 90]
SCALE_S = 1.0

# ロジットのスケーリング: F = alpha * tanh(...) [cite: 94]
SCALE_ALPHA = 1.0


# ==========================
# 4. Gumbel-Sinkhorn (微分可能置換)
# ==========================
# Sinkhornの反復回数 (l) [cite: 122]
SINKHORN_ITER = 60

# 温度パラメータ (tau) [cite: 121]
# TSP-20の探索範囲: {2.0, 3.0, 4.0, 5.0}
TAU = 2.0

# ノイズスケール (gamma または epsilon scale) [cite: 121]
# TSP-20の探索範囲: {0.005, 0.01, 0.05, 0.1, 0.2, 0.3}
NOISE_SCALE = 0.1


# ==========================
# 5. 学習・最適化 (Training & Optimization)
# ==========================
EPOCHS = 300            # 最大エポック数 [cite: 122]
LR = 1e-3               # 学習率 (0.001) [cite: 128]
WEIGHT_DECAY = 1e-4     # 正則化項 (lambda) [cite: 127]
OPTIMIZER = "Adam"      # [cite: 127]

# スケジューリング設定
WARMUP_EPOCHS = 15      # ウォームアップ期間 
PATIENCE = 50           # Early Stoppingの忍耐値 

# 勾配クリッピング
# 論文では "adaptive gradient clipping" と記載 
# 具体的な閾値がないため、PyTorchの一般的なclip_grad_norm_を使用想定
USE_GRAD_CLIP = True
MAX_GRAD_NORM = 1.0     # 一般的なデフォルト値 (必要に応じて調整)

# デバイス設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"