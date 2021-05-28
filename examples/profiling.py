#%%
import pandas as pd
import subprocess
shapes = pd.read_csv('../datasets/shapes.csv')

def time_gemm(m, n, k, b):
    # run_cmd = './bin/gemm %d %d %d %d' % (m,n,k,b)
    elapsed = subprocess.run(["./bin/gemm", str(m), str(n), str(k), str(b)], capture_output=True, text=True)
    return float(elapsed.stdout)

def time_sparsify(m, k):
    elapsed = subprocess.run(["./bin/sparsify", str(m), str(k)], capture_output=True, text=True)
    return float(elapsed.stdout)

def time_spmm(m, n, k, b):
    elapsed = subprocess.run(["./bin/spmm", str(m), str(n), str(k), str(b)], capture_output=True, text=True)
    return float(elapsed.stdout)

#%%
d = {
    'm': [],
    'n':[],
    'k':[],
    'b':[],
    'gemm': [],
    'prune': [],
    'spmm': []
    }

for idx, row in shapes.iterrows():
    m = row['m']
    n = row['n']
    k = row['k']
    b = row['b']
    d['m'].append(m)
    d['n'].append(n)
    d['k'].append(k)
    d['b'].append(b)
    d['gemm'].append(time_gemm(m,n,k,b))
    d['prune'].append(time_sparsify(m,k))
    d['spmm'].append(time_spmm(m,n,k,b))
#%%
df = pd.DataFrame(d, columns=['m','n','k','b','gemm','prune','spmm'])
df.to_csv('compare.csv')
# %%
import matplotlib.pyplot as plt
import numpy as np
labels = ['L' + str(x) for x in range(len(d['m']))]

def make_plot(labels, gemm_times, spmm_times):
    x = np.arange(labels)
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, gemm_times, width, label='GEMM Runtime')
    rects2 = ax.bar(x + width/2, spmm_times, width, label='SpMM Runtime')
    ax.bar()