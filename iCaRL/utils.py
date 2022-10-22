import torch

def cal_NN(ft, q): 
    score = torch.sum((ft.reshape(ft.shape[0], 1, ft.shape[1]).repeat(1, q.shape[0], 1) - q) ** 2, axis=2)
    return torch.min(score, dim=1)[0].tolist() 

def get_norm2(x): 
    return torch.sqrt(torch.sum(x ** 2))

def reduce(p, m_, m): 
    q = [] 

    for i in range(0, len(p), m_): 
        j = i 
        while j < i + m: 
            q.append(p[j])
            j += 1
    return q


def construct_by_ft(m, sub_ds, feature):
    import time 
    t0 = time.time()
    MAX_VAL = 1e20 ##  
    n = len(sub_ds)
    mu = torch.sum(torch.concat([feature[i].reshape(1, -1) for i in range(n)]), axis=0) / n 
    p = [] 
    vis = {}
    sum_score = torch.zeros(1, len(feature[0]))

    for i in range(0, m): 
        min_val = MAX_VAL
        min_idx = 0 
        for j in range(0, n):
            vis.setdefault(j, 0)
            if vis[j] == 1: continue  
            val = get_norm2(mu - (feature[j] + sum_score) / (i + 1)) 
            if min_val > val: min_val = val; min_idx = j 

        sum_score += feature[min_idx].reshape(1, -1) 
        p.append(sub_ds[min_idx][0]) 
        vis[min_idx] = 1
    print("weew:\n", time.time()-t0)

    return p 

def construct_by_score(m, sub_ds, e):
    import time 
    t0 = time.time()
    MAX_VAL = 1e20 ##  
    n = len(sub_ds)
    e = (e - torch.mean(e)) / torch.std(e)
    mu = torch.mean(e)
    p = [sub_ds[i][0] for i in sorted(list(range(n)), key = lambda x: abs(mu - e[x]))[:m]] 
    print("weew:\n", time.time()-t0)
    return p 
# import torch 
# x = torch.ones(1, 4)
# y = torch.ones(1, 4)
# y=  torch.sum(torch.concat([x, y]), axis=0)
# construct() 

