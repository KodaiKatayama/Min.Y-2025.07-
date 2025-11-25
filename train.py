import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np # 新規: 平均値を計算するため

from data import TSPDataset
from model import TSPModel
from utils import get_cyclic_matrix, sample_gumbel, sinkhorn, Hungarian

# --- 設定エリア ---
NUM_NODES = 20        # 都市の数
NUM_SAMPLES = 1000    
BATCH_SIZE = 32       
EPOCHS = 100          
LR = 1e-3             
ALPHA = 10.0          
TAU = 2.0             
GAMMA = 0.1           
SINKHORN_ITERS = 60
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- ロギングと保存設定 ---
BEST_DIST = float('inf') # 最小距離の初期値
SAVE_PATH = 'best_model.pth' # 保存ファイル名
# 学習履歴を記録するリスト
history = {'loss': [], 'dist': []} 
# -------------------------

dataset = TSPDataset(NUM_SAMPLES, NUM_NODES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = TSPModel(input_dim=2, hidden_dim=128, num_nodes=NUM_NODES, alpha=ALPHA).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
V = get_cyclic_matrix(NUM_NODES).to(DEVICE)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    total_distance = 0
    
    for batch in dataloader:
        points = batch['points'].to(DEVICE)
        distances = batch['distance'].to(DEVICE)
        
        optimizer.zero_grad()

        # --- 順伝播とLoss計算 (変更なし) ---
        logits = model(points)
        noise = sample_gumbel(logits.shape).to(DEVICE)
        noisy_logits = (logits + GAMMA * noise) / TAU
        T = sinkhorn(noisy_logits, n_iters=SINKHORN_ITERS)
        
        V_batch = V.unsqueeze(0) 
        soft_adj = torch.matmul(torch.matmul(T, V_batch), T.transpose(1, 2))
        loss = torch.sum(distances * soft_adj) / BATCH_SIZE
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # --- 評価 (実際の距離計算) ---
        with torch.no_grad():
            # 決定論的なデコード (ノイズなし)
            deterministic_logits = logits / TAU
            P = Hungarian(-deterministic_logits) 
            
            hard_adj = torch.matmul(torch.matmul(P, V_batch), P.transpose(1, 2))
            current_dist_batch = torch.sum(distances * hard_adj, dim=(1, 2))
            total_distance += current_dist_batch.mean().item()
        
    avg_loss = total_loss / len(dataloader)
    avg_dist = total_distance / len(dataloader) # 各エポックの平均距離

    # 1. 履歴を記録
    history['loss'].append(avg_loss)
    history['dist'].append(avg_dist)

    # 2. ベストモデルを保存 (最低の検証距離を達成したモデルを保存)
    if avg_dist < BEST_DIST:
        print(f"    *New Best Found! Saving model to {SAVE_PATH} (Dist: {BEST_DIST:.4f} -> {avg_dist:.4f})")
        BEST_DIST = avg_dist
        torch.save(model.state_dict(), SAVE_PATH) # パラメータを保存

    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Real Dist = {avg_dist:.4f} (Best: {BEST_DIST:.4f})")

# 最終結果の保存 (図の生成に利用)
np.savez('training_history.npz', loss=history['loss'], dist=history['dist'])