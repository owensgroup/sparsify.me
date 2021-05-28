#%%
import torch
import pandas as pd


shapes = pd.read_csv('../../datasets/shapes.csv')
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
import matplotlib.pyplot as plt
import numpy as np
total_spmm_time = [a + b  for a,b in zip(info['SpMM Time'], info['D2S Time'])]
labels = ['Layer' + str(x) for x in range(len(total_spmm_time))]
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, info['GEMM Time'][1:10], width, label='GEMM Runtime')
rects2 = ax.bar(x + width/2, total_spmm_time[1:10], width, label='Total SpMM Runtime')

ax.set_ylabel('GPU Time (ms)')
ax.set_title('GEMM Runtime vs SpMM Runtime (90% Sparsity)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1)
ax.bar_label(rects2)
plt.show()
# %%
