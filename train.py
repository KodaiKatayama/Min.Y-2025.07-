import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import TSPDataset
from model import SimpleTSPModel

def train():
    # --- 1. 設定 (Hyperparameters) ---
    num_nodes = 20       # TSP-20 (20都市の問題)
    num_samples = 2000   # 学習データの数 (論文ではもっと多いですが、実験用に少なめで)
    batch_size = 64      # 一度に解く問題数
    epochs = 100         # 学習回数 (論文では300回以上 [cite: 122])
    learning_rate = 1e-3 # 学習率 [cite: 128]
    
    # GPUが使えるならGPUを使う
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # --- 2. データとモデルの準備 ---
    print("データを生成中...")
    dataset = TSPDataset(num_samples=num_samples, num_nodes=num_nodes, seed=42)
    
    # DataLoader: データをバッチサイズごとにまとめて運んでくれる運び屋
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SimpleTSPModel(num_nodes=num_nodes).to(device)
    
    # Optimizer: モデルのパラメータを更新する指導教官 (Adamを使用)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 3. 学習ループ (Training Loop) ---
    print("学習開始！")
    model.train() # 「学習モード」にスイッチ

    for epoch in range(epochs):
        total_loss = 0
        
        for batch in dataloader:
            # データをデバイスへ移動
            distances = batch['distance'].to(device) # Shape: (Batch, N, N)
            
            # --- Forward (順伝播) ---
            # 1. モデルに距離行列を入れて、ソフト置換行列 T を得る
            soft_perm = model(distances) # (Batch, N, N)
            
            # --- Loss Calculation (損失計算) ---
            # 式: Loss = <D, T V T^T> (距離行列 D と、生成されたルート行列の内積)
            
            # 正準サイクル行列 V を取り出し、バッチサイズ分に増やす
            V = model.matrix_V
            V_batch = V.unsqueeze(0).expand(distances.size(0), -1, -1) # (Batch, N, N)
            
            # 行列演算: T @ V @ T^T を計算
            # bmm (Batch Matrix Multiplication) は「バッチごとの行列積」です
            
            # Step 1: T @ V
            TV = torch.bmm(soft_perm, V_batch)
            
            # Step 2: (T @ V) @ T_transpose
            # これで「ソフト隣接行列（どの都市からどの都市へ行くか）」ができます
            soft_adjacency = torch.bmm(TV, soft_perm.transpose(1, 2))
            
            # Step 3: 距離行列 D とかけ算して合計する
            # これが「期待される移動距離」になります
            loss = torch.sum(distances * soft_adjacency) / distances.size(0) # バッチ平均をとる
            
            # --- Backward (逆伝播) ---
            optimizer.zero_grad() # 前回の勾配をリセット
            loss.backward()       # 勾配を計算
            optimizer.step()      # パラメータを更新
            
            total_loss += loss.item()
            
        # 10エポックごとに経過を表示
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} | 平均移動距離 (Loss): {avg_loss:.4f}")

    print("学習終了！")
    
    # モデルを保存しておく
    torch.save(model.state_dict(), "my_tsp_model.pth")
    print("モデルを 'my_tsp_model.pth' に保存しました。")

if __name__ == "__main__":
    train()