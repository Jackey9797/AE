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

def get_reg(x, y): 
    if y == None: return 0 
    return torch.sum((x - y) ** 2)

def train_valid(model):
    # setup_seed(3323)    
    from torch.utils.data import Subset
    AUC_score = []
    p = []

    data_stream = int(len(ds) / 7000) - 1
    data_stream = 70 #!
    week_n = 7000

    for stream in range(1, data_stream + 1): 
        pre = (stream - 1) * 7000
        ds_1 = Subset(ds, range(pre, pre + week_n)) # 按周划分数据
        ds_2 = Subset(ds, range(pre + week_n, pre + 2 * week_n))
        train_dl = DataLoader(ds_1, shuffle=True, batch_size=week_n) # 直接把一周所有数据当一个batch输入
        test_dl = DataLoader(ds_2, shuffle=False, batch_size=week_n) 

        # if stream == 1: 
        #     dis_loss = None  
        # else : 
        #     old_sample = torch.concat([x.reshape(1,-1) for x in p], axis=0)
        #     q = model(old_sample).detach()
        #     dis_loss = q
        # #---- train loop 
        # model.train()
        # tmp = []
        # for i in range(epochs): 
        #     for idx, (X, y) in enumerate(train_dl, 1): 
        #         # print(X.shape, y.shape)
        #         model.optimizer.zero_grad()
        #         X_pred, ft = model(X, return_feature=True)
        #         e = model.loss_func_e(X_pred, X) # 计算异常分数

        #         idx_sorted = list(range(len(X))) 
        #         idx_sorted.sort(key=lambda x: e[x], reverse=True) # 按异常分数降序排序
        #         choose_num = int(len(X) * model.a) 
        #         choose_mask = torch.zeros(e.shape) 
        #         choose_mask[idx_sorted[:choose_num]] = 1  # 前a% 的mask为1 

        #         output = None
        #         if dis_loss != None: output = model(old_sample)

        #         loss = torch.sum(e * (1 - y) * choose_mask) + torch.sum(torch.maximum(model.a0 - e, torch.tensor([0])) * y * choose_mask) + get_reg(output, dis_loss) 
        #         # print(loss.shape)          
        #         # loss
        #         loss.backward() # 

        #         model.optimizer.step()
        #         if idx % 1 == 0: 
        #             print("loss now:", loss)  

    
        # from iCaRL.construct import construct, reduce
        # import math     
        # if stream != 1: 
        #     p = reduce(p,math.floor(model.m / (stream - 1)),  math.floor(model.m / stream)) 

        # p += construct(math.floor(model.m / stream), ds_1, ft)
        # print(stream, len(p))

     

        model.eval() ## 应该影响不大把
        anomaly_score = []
        from sklearn.metrics import roc_auc_score
        for idx, (X, y) in enumerate(test_dl, 1):  # 测试
            with torch.no_grad():
                X_pred = model(X)  
                loss = model.loss_func_e(X_pred, X)
                loss_list = loss.detach().tolist()      
                # ! anomaly_score += loss_list
                # anomaly_score += torch.rand(7000).tolist()
                anomaly_score += torch.sum(X > 0.0001, axis=1).tolist()

            # print(len(y.tolist()), len(loss.detach().tolist()))
            # AUC_score = roc_auc_score(y.tolist(), loss.detach().tolist()) 


        with open(exp_name + "Week" + str(stream) +  "_result.txt", 'w') as f: 
            # f.write("AUC score:", str(AUC_score) + '\n')
            for idx in range(len(anomaly_score)): 
                f.write(str(anomaly_score[idx]) + '\n')   
        print("\nWeek %d finished!\n" % stream)
    return model

if __name__ == '__main__':
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

    lr = 1e-4
    epochs = 20
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Using %s !" % str(device))
    # batch_size = 16 #! 小一点好
    input_size = 502 #* vectore size

    # model = autoencoder(input_size).to(device)  
    model = autoencoder(input_size) 
    model.optimizer = torch.optim.SGD(model.parameters(),lr = lr,weight_decay=0.00001)
    model.loss_func_e = lambda x, y: torch.sqrt(torch.sum((x - y) ** 2, axis=1)) 
    model.loss_func = None 
    model.metric = None 
    model.a = 0.2 #* 前a% 弱监督数据 百分比 超参
    model.a0 = 5
    model.m = 1400

    exp_name = "exp_icarl_all_data_None_exp"
    if not os.path.exists(os.path.join("exp", exp_name)):
        os.makedirs(os.path.join("exp", exp_name)) 
    exp_name = os.path.join(os.path.join("exp", exp_name), exp_name)

    model = train_valid(model)
    # torch.save(model.state_dict(), open(exp_name + '.pth', "wb")) #! 训练则开

    # # model.load_state_dict(torch.load(open(exp_name + '.pth', "rb"))) #! 只测试则开 
    
    # anomaly_score = []
    # GT_score = []

    # model.eval() 
    # for idx, X in enumerate(test_dl, 1): 
    #     X_pred = model(X)  
    #     loss = model.loss_func(X_pred, X)             
    #     anomaly_score.append(loss.detach().item())
    #     GT_score.append(y_test[idx - 1])

    # # torch.save(model.state_dict(), open(exp_name + '.pth', "wb"))
    # with open(exp_name + '.txt', "w") as f:
    #     for score, score_ in zip(anomaly_score, GT_score):
    #         f.write(str(score) + ' ' + str(score_) + '\n')

    
