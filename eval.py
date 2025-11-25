import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt # 描画ライブラリ

from data import TSPDataset
from model import TSPModel
from utils import get_cyclic_matrix, Hungarian

# --- 設定エリア ---
NUM_NODES = 20
TEST_SAMPLES = 1000 # 論文通りテストは1000インスタンス [cite: 673]
BATCH_SIZE = 32
TAU = 2.0
ALPHA = 10.0
SAVE_PATH = 'best_model.pth' 
HISTORY_FILE = 'training_history.npz'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 論文のベンチマーク値 (TSP-20) ---
OPTIMAL_LEN = 3.83 # Concordeの平均 [cite: 870]
GREEDY_NN_LEN = 4.51 # Greedy Nearest Neighborの平均 [cite: 870]
PAPER_MODEL_LEN = 4.06 # 論文モデルの平均 [cite: 748, 870]
# -------------------------------------

def get_test_distances(model, V_matrix):
    """ベストモデルをロードし、テストデータ1000問の距離をリストで取得する"""
    dataset = TSPDataset(TEST_SAMPLES, NUM_NODES) # 新しいテストデータを生成
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    all_distances = []
    
    with torch.no_grad():
        for batch in dataloader:
            points = batch['points'].to(DEVICE)
            distances = batch['distance'].to(DEVICE)
            
            logits = model(points)
            
            # P = Hungarian(-F/τ)
            P = Hungarian(-(logits / TAU)) 
            
            V_batch = V_matrix.unsqueeze(0)
            hard_adj = torch.matmul(torch.matmul(P, V_batch), P.transpose(1, 2))
            
            # 各インスタンスの距離を計算
            dist_per_instance = torch.sum(distances * hard_adj, dim=(1, 2))
            all_distances.extend(dist_per_instance.cpu().numpy().tolist())
            
    return np.array(all_distances)

def plot_results(model):
    # 学習履歴をロード
    try:
        history = np.load(HISTORY_FILE)
    except FileNotFoundError:
        print("Error: Training history not found. Run train.py first.")
        return

    # 1. 図1の再現 (学習曲線)
    plt.figure(figsize=(12, 5))
    plt.plot(history['loss'], label='Training Loss', color='blue')
    plt.plot(history['dist'], label='Real Distance (Evaluation)', color='red')
    plt.title(f'Figure 1: Training History (TSP-{NUM_NODES})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Distance')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. 図2の再現 (距離分布 - ヒストグラム)
    # V行列を用意
    V = get_cyclic_matrix(NUM_NODES).to(DEVICE)
    
    # テスト距離を取得
    test_distances = get_test_distances(model, V)
    
    plt.figure(figsize=(8, 5))
    plt.hist(test_distances, bins=20, density=True, alpha=0.6, label=f'Current Model ($\mu$={np.mean(test_distances):.2f})', color='purple')
    
    # 論文のベンチマーク値を縦線で表示
    plt.axvline(OPTIMAL_LEN, color='green', linestyle='--', label=f'Optimal ($\mu$={OPTIMAL_LEN})')
    plt.axvline(GREEDY_NN_LEN, color='red', linestyle='--', label=f'Greedy NN ($\mu$={GREEDY_NN_LEN})')
    
    plt.title(f'Figure 2: Tour Length Distribution (TSP-{NUM_NODES})')
    plt.xlabel('Tour Length')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # モデルの初期化とロード
    model = TSPModel(input_dim=2, hidden_dim=128, num_nodes=NUM_NODES, alpha=ALPHA).to(DEVICE)
    try:
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        print(f"Loaded best model from {SAVE_PATH}.")
        plot_results(model)
    except FileNotFoundError:
        print("\n--- ERROR ---")
        print(f"Model file not found at {SAVE_PATH}. Please run train.py first.")
        print("-------------")