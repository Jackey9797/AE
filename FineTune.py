import os 
import numpy as np
import random
import torch
import sys 
import dill as pickle 

from models.ws_ae import autoencoder
from data import dataset 
from torch.utils.data import DataLoader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子

def train_valid(model, ds):
    # setup_seed(3323)    
    from torch.utils.data import Subset
    AUC_score = []


    data_stream = int(len(ds) / 7000) - 1
    week_n = 7000

    for stream in range(1, data_stream + 1): 
        pre = (stream - 1) * 7000
        ds_1 = Subset(ds, range(pre, pre + week_n)) # 按周划分数据
        ds_2 = Subset(ds, range(pre + week_n, pre + 2 * week_n))
        train_dl = DataLoader(ds_1, shuffle=True, batch_size=week_n) # 直接把一周所有数据当一个batch输入
        test_dl = DataLoader(ds_2, shuffle=False, batch_size=week_n) 

        #---- train loop
        if stream <= 35:  
            model.train()
            for i in range(model.epochs): 
                for idx, (X, y) in enumerate(train_dl, 1): 
                    # print(X.shape, y.shape)
                    model.optimizer.zero_grad()
                    X_pred = model(X)
                    e = model.loss_func_e(X_pred, X) # 计算异常分数

                    idx_sorted = list(range(len(X))) 
                    idx_sorted.sort(key=lambda x: e[x], reverse=True) # 按异常分数降序排序
                    choose_num = int(len(X) * model.a) 
                    choose_mask = torch.zeros(e.shape) 
                    choose_mask[idx_sorted[:choose_num]] = 1  # 前a% 的mask为1 

                    if model.setting == 'new': 
                        if stream <= 35: y = torch.zeros_like(y)  
                    loss = torch.sum(e * (1 - y) * choose_mask) + torch.sum(torch.maximum(model.a0 - e, torch.tensor([0])) * y * choose_mask) 
                    # print(loss.shape)          
                    # loss
                    loss.backward() # 

                    model.optimizer.step()
                    if idx % 1 == 0: 
                        print("loss now:", loss)  

        model.eval() ## 应该影响不大把
        anomaly_score = []
        from sklearn.metrics import roc_auc_score
        for idx, (X, y) in enumerate(test_dl, 1):  # 测试
            with torch.no_grad():
                X_pred = model(X)  
                loss = model.loss_func_e(X_pred, X)
                loss_list = loss.detach().tolist()      
                anomaly_score += loss_list

            # print(len(y.tolist()), len(loss.detach().tolist()))
            # AUC_score = roc_auc_score(y.tolist(), loss.detach().tolist()) 


        with open(model.exp_name + "Week" + str(stream) +  "_result.txt", 'w') as f: 
            # f.write("AUC score:", str(AUC_score) + '\n')
            for idx in range(len(anomaly_score)): 
                f.write(str(anomaly_score[idx]) + '\n')   
        print("\nWeek %d finished!\n" % stream)
    return model

def FineTune(exp_name, setting): 
    x, y = None , None

    if not os.path.exists('data/x.pkl'):
        x, y = dataset.generate_stream_data()
        with open('data/x.pkl', 'wb') as f:
            pickle.dump(x, f)
        with open('data/y.pkl', 'wb') as f:
            pickle.dump(y, f)
    else : 
        with open('data/x.pkl', 'rb') as f:
            x = pickle.load(f)
        with open('data/y.pkl', 'rb') as f:
            y = pickle.load(f)

    ds = dataset.manyWeek(x, y) 

    lr = 1e-3
    epochs = 20
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Using %s !" % str(device))
    # batch_size = 16 #! 小一点好
    input_size = 502 #* vectore size

    # model = autoencoder(input_size).to(device)  
    model = autoencoder(input_size) 
    model.optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    model.loss_func_e = lambda x, y: torch.sqrt(torch.sum((x - y) ** 2, axis=1)) 
    model.setting = setting
    model.loss_func = None 
    model.metric = None 
    model.epochs = epochs 
    model.a = 0.2 #* 前a% 弱监督数据 百分比 超参
    model.a0 = 5

    if not os.path.exists(os.path.join("exp2", exp_name)):
        os.makedirs(os.path.join("exp2", exp_name)) 
    exp_name = os.path.join(os.path.join("exp2", exp_name), exp_name)
    model.exp_name = exp_name
    model = train_valid(model, ds)