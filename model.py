import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSinkhorn(nn.Module):
    def __init__(self, iterations=60, temperature=1.0, gamma=0.01):
        """
        Args:
            iterations (int): Sinkhornの反復回数 (論文では60~80)
            temperature (float): tau (論文では1.0~5.0付近)
            gamma (float): ノイズの強さ係数 (論文では0.005~0.3程度) 
        """
        super().__init__()
        self.iterations = iterations
        self.temperature = temperature
        self.gamma = gamma # 修正: noise_scale -> gamma

    def forward(self, logits):
        # logits shape: (batch, n, n)
        
        # 1. Gumbelノイズ (epsilon) の生成
        u = torch.rand_like(logits)
        epsilon = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        
        # 2. 式(8)の実装: (F + gamma * epsilon) / tau
        y = (logits + self.gamma * epsilon) / self.temperature
        
        # 3. Sinkhorn正規化
        P = F.softmax(y, dim=-1)
        for _ in range(self.iterations):
            P = P / (P.sum(dim=1, keepdim=True) + 1e-20)
            P = P / (P.sum(dim=2, keepdim=True) + 1e-20)
            
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
            gamma=0.01 # 論文の推奨値の一つ
        )
        
        # 論文の alpha (スケーリングパラメータ)
        self.alpha = nn.Parameter(torch.tensor(10.0))
        
        # --- 固定パラメータ (学習しない) ---
        # 正準サイクル行列 V をここで作っておきます
        # 式(2),(3)にある 1->2->3...->n->1 という行列です [cite: 49-67]
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

# --- 動作確認用 ---
if __name__ == "__main__":
    # モデルのテスト
    n = 5
    model = SimpleTSPModel(num_nodes=n)
    
    # V行列が正しく作れているか確認
    print("--- 正準サイクル行列 V (Canonical Cycle) ---")
    print(model.matrix_V)
    # 期待値:
    # [[0, 1, 0, 0, 0],
    #  [0, 0, 1, 0, 0], ...
    #  [1, 0, 0, 0, 0]] となっていればOK