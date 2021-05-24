#%%
import torch
import pandas as pd


shapes = pd.read_csv('../datasets/shapes.csv')
gamma = .1
# %%
def time_gemm(m, n, k, b):
    device = torch.device('cuda')
    A = torch.randn(b, m, k, device=device)
    B = torch.randn(k, n, device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    C = A @ B
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

def time_coo(m, n, k, b, gamma):
    device = torch.device('cuda')
    SpA = torch.randn(b, m, k, device=device).flatten()
    B = torch.randn(b, k, n, device=device)
    perm = torch.randperm(SpA.size(0))
    SpA[perm[:int(SpA.size(0) * (1-gamma))]] = 0
    SpA = SpA.view(b,m,k)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    SpA = SpA.to_sparse()
    end.record()
    torch.cuda.synchronize()

    fmt_conversion = start.elapsed_time(end)
    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start2.record()
    C = torch.bmm(SpA, B)
    end2.record()
    torch.cuda.synchronize()
    spmm_time = start2.elapsed_time(end2)
    return fmt_conversion, spmm_time



# %%
info = {'Size A': [], 'Size B': [], 'GEMM Time':[], 'D2S Time': [], 'SpMM Time': []}
for index, row in shapes.iterrows():
    m = row['m']
    n = row['n']
    k = row['k']
    b = row['b']
    info['Size A'].append((b, m, k))
    info['Size B'].append((k,n))
    gemm_rt = time_gemm(m,n,k,b)
    fmt_rt, spmm_rt = time_coo(m,n,k,b,gamma)
    info['GEMM Time'].append(gemm_rt)
    info['SpMM Time'].append(spmm_rt)
    info['D2S Time'].append(fmt_rt)

# %%
