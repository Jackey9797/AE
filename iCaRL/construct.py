import torch
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


def construct(m, sub_ds, feature):
    MAX_VAL = 1e20 ##  
    n = len(sub_ds)
    mu = torch.sum(torch.concat([feature[i].reshape(1, -1) for i in range(n)]), axis=0) / n 
    # print(mu) #TODO
    p = [] 
    vis = {}
    sum_score = torch.zeros(1, len(feature[0]))

    for i in range(0, m): 
        min_val = MAX_VAL
        min_idx = 0 
        for j in range(0, n):
            vis.setdefault(j, 0)
            if vis[j] == 1: continue  
            val = get_norm2(mu - (feature[j] + sum_score)) 
            if min_val > val: min_val = val; min_idx = j 

        sum_score += feature[min_idx].reshape(1, -1) 
        p.append(sub_ds[min_idx][0]) 
        vis[min_idx] = 1

    return p 

# import torch 
# x = torch.ones(1, 4)
# y = torch.ones(1, 4)
# y=  torch.sum(torch.concat([x, y]), axis=0)
# construct() 
