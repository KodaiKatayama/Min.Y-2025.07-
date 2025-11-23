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



if __name__ == "__main__":
    
    # 論文の表記と対応付けるための辞書
    param_map = {
        "NUM_NODES": "N (都市数)",
        "INPUT_DIM": "入力次元",
        "TRAIN_SIZE": "学習データ数",
        "BATCH_SIZE": "バッチサイズ",
        "HIDDEN_DIM": "隠れ層次元",
        "NUM_LAYERS": "GNN層数",
        "SCALE_S": "s (距離スケーリング)",
        "SCALE_ALPHA": "α (ロジットスケーリング)",
        "SINKHORN_ITER": "l (Sinkhorn反復)",
        "TAU": "τ (温度)",
        "NOISE_SCALE": "γ (ノイズスケール)",
        "EPOCHS": "最大エポック数",
        "LR": "LR (学習率)",
        "WEIGHT_DECAY": "λ (Weight Decay)",
        "OPTIMIZER": "Optimizer",
        "DEVICE": "デバイス",
    }
    
    print("=" * 35)
    print(f" Structure As Search: TSP-{NUM_NODES} 設定 ")
    print("=" * 35)

    # configモジュール内の大文字変数を取得
    config_vars = {k: v for k, v in globals().items() if k.isupper() and k not in ['USE_GRAD_CLIP', 'MAX_GRAD_NORM']}
    
    for var_name, description in param_map.items():
        if var_name in config_vars:
            # 論文の表記に合わせて出力
            print(f"{description:<20}: {config_vars[var_name]}")

    print("-" * 35)
    print(f"{'Warmup期間':<20}: {WARMUP_EPOCHS} epochs")
    print(f"{'Early Stopping Patience':<20}: {PATIENCE}")
    print(f"{'勾配クリッピング':<20}: {'有効' if USE_GRAD_CLIP else '無効'}")
    if USE_GRAD_CLIP:
         print(f"{'最大勾配ノルム':<20}: {MAX_GRAD_NORM}")
    print("=" * 35)