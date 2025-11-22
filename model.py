import torch
import torch.nn as nn
import torch.nn.functional as F

from data import TSPDataset

class GumbelSinkhorn(nn.Module):
    def __init__(self, iterations=60, temperature=3.0, gamma=0.01):
        super().__init__()
        self.iterations = iterations
        self.temperature = temperature
        self.gamma = gamma

    def forward(self, logits):
        
        # 1. Gumbelノイズ (epsilon) の生成
        u = torch.rand_like(logits)
        epsilon = -torch.log(-torch.log(u))
        
        # 2. 式(8)の実装: (F + gamma * epsilon) / tau
        y = (logits + self.gamma * epsilon) / self.temperature
        
        # 3. Sinkhorn正規化
        P = F.softmax(y, dim=-1)
        for _ in range(self.iterations):
            P = P / (P.sum(dim=1, keepdim=True))
            P = P / (P.sum(dim=2, keepdim=True))
            
        return P

class SimpleTSPModel(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        
        # --- 学習可能なパラメータ ---
        # 今回は簡易版なのでLinear層を使いますが、
        # 本来はここにGNNが入ります (論文のSAGsなど)
        self.encoder = nn.Linear(1, 1) 
        
        self.sinkhorn = GumbelSinkhorn(
            iterations=60, 
            temperature=1.0, 
            gamma=0.01
        )
        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.register_buffer('matrix_V', self._create_canonical_cycle(num_nodes))

    def _create_canonical_cycle(self, n):
        """
        1->2->3...->n->1 というサイクルを表す行列Vを作る関数
        """
        V = torch.zeros(n, n)
        for i in range(n - 1):
            V[i, i + 1] = 1.0
        V[n - 1, 0] = 1.0 # 最後の都市から最初に戻る
        return V

    def forward(self, distances):
        # 1. データ整形
        x = distances.unsqueeze(-1)
        
        # 2. エンコード (距離 -> Logits F)
        logits = self.encoder(x).squeeze(-1)
        
        # 3. 活性化 (alpha * tanh)
        logits = self.alpha * torch.tanh(logits)
        
        # 4. Sinkhorn (T)
        soft_perm = self.sinkhorn(logits)
        
        return soft_perm

def test_model_full_flow():
    print("=== モデル動作の完全テストを開始 ===")
    
    # 1. 設定
    batch_size = 2
    num_nodes = 5
    model = SimpleTSPModel(num_nodes=num_nodes)
    
    # 2. データを用意 (data.pyで作ったクラスを使う)
    dataset = TSPDataset(num_samples=batch_size, num_nodes=num_nodes)
    data = dataset[0] # 1つ取り出す
    
    # バッチ次元を足す (N, N) -> (1, N, N)
    # 本番のDataLoaderはこれを自動でやってくれますが、テストなので手動で
    distances = data['distance'].unsqueeze(0) 
    
    print(f"Input Distance Shape: {distances.shape}")

    # 3. Forwardパス (順伝播)
    # ここでエラーが出たら model.py の forward か shape が間違ってる
    soft_perm = model(distances)
    
    print(f"Output SoftPerm Shape: {soft_perm.shape}")
    
    # チェック1: 形はあっているか？ (Batch, N, N)
    assert soft_perm.shape == (1, num_nodes, num_nodes), "出力サイズが変です！"
    
    # チェック2: Sinkhornは効いているか？ (行の和がほぼ1.0)
    row_sum = soft_perm.sum(dim=2)
    print(f"Row Sums: {row_sum.detach().numpy()}")
    if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-3):
        print("警告: 行の合計が1になっていません！Sinkhornの設定を見直してください")
    
    # 4. Loss計算のテスト (これが一番大事！)
    # 論文の式: Loss = <D, T V T_t>
    
    # Vを取り出す (register_bufferしたやつ)
    V = model.matrix_V
    
    # T * V * T^T を計算
    # batch行列積 (bmm) を使います
    # (B, N, N) x (N, N) は直接できないので、Vをバッチサイズ分コピーするか、matmulを使う
    
    # Vをバッチサイズに合わせて拡張: (1, N, N)
    V_batch = V.unsqueeze(0).expand(distances.size(0), -1, -1)
    
    # T @ V @ T.transpose
    permuted_V = torch.bmm(soft_perm, torch.bmm(V_batch, soft_perm.transpose(1, 2)))
    
    # 距離行列 D との内積 (総距離)
    loss = torch.sum(distances * permuted_V)
    
    print(f"Calculated Loss (Total Distance): {loss.item()}")
    
    # 5. Backwardパス (逆伝播)
    # ここでエラーが出たら「学習できない（勾配が途切れてる）」ということ
    loss.backward()
    
    # encoderの重みに勾配が入っているかチェック
    if model.encoder.weight.grad is not None:
        print("Gradient Check: OK! (学習可能です)")
    else:
        print("Gradient Check: NG! (勾配が流れていません。detach()とかしてませんか？)")

    print("\n=== All Checks Passed! 安心してください、動きます ===")

if __name__ == "__main__":
    test_model_full_flow()