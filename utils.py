import torch
from scipy.optimize import linear_sum_assignment

def get_adjacency_matrix(distance_matrix, scale):
    adjacency_matrix = torch.exp(-distance_matrix / scale)
    return adjacency_matrix

def sample_gumbel(shape):
    u = torch.rand(shape)
    return -torch.log(-torch.log(u))

def sinkhorn(logit, n_iters):
    T = torch.exp(logit)
    for _ in range(n_iters):
        T = T / torch.sum(T, dim=-1, keepdim=True)
        T = T / torch.sum(T, dim=-2, keepdim=True)
    return T

def Hungarian(logits):
    logits_np = logits.detach().cpu().numpy()
    batch_size, n, _ = logits.shape
    
    P_list = []

    for i in range(batch_size):
        row, col = linear_sum_assignment(logits_np[i])
        P = torch.zeros(n, n)
        P[row, col] = 1
        P_list.append(P)
    return torch.stack(P_list).to(logits.device)

def get_cyclic_matrix(num_nodes):
    matrix = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes - 1):
        matrix[i, i + 1] = 1
    matrix[num_nodes - 1, 0] = 1
    return matrix