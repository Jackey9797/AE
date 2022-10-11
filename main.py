import os 
import numpy as np
import random
import torch
import sys 
import dill as pickle 

from models.ae import autoencoder
from data import dataset 
from torch.utils.data import DataLoader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子

def train(model):
    setup_seed(3323) # exp0: 32 exp1:320 exp2:330  exp3:3323  exp4:3323     exp5:3323 

#----- 数据读入

    

#----- 模型构建
    


    

    for epoch in range(1, epochs + 1): 
        #---- train loop 
        model.train()
        for idx, X in enumerate(train_dl, 1): 
            model.optimizer.zero_grad()

            X_pred = model(X)  
            loss = model.loss_func(X_pred, X)             
            loss.backward()

            model.optimizer.step()
            if idx % 7 == 0: 
                print("loss now:", loss)  
    return model

if __name__ == '__main__':
    x_train = None 
    x_test = None 
    y_test = None

    if not os.path.exists('data/x_train.pkl'):
        x_train, x_test, y_test = dataset.generate_data()
        with open('data/x_train.pkl', 'wb') as f:
            pickle.dump(x_train, f)
        with open('data/x_test.pkl', 'wb') as f:
            pickle.dump(x_test, f)
        with open('data/y_test.pkl', 'wb') as f:
            pickle.dump(y_test, f)
    else : 
        with open('data/x_train.pkl', 'rb') as f:
            x_train = pickle.load(f)
        with open('data/x_test.pkl', 'rb') as f:
            x_test = pickle.load(f)
        with open('data/y_test.pkl', 'rb') as f:
            y_test = pickle.load(f)

    ds_train = dataset.oneWeek(x_train) 
    ds_test = dataset.oneWeek(x_test)  
    

    lr = 1e-3
    epochs = 1
    batch_size = 16
    input_size = ds_train[0].shape[0]

    model = autoencoder(input_size)  
    model.batch_size = batch_size 
    model.optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    model.loss_func = lambda x, y: torch.sum((x - y) ** 2) / x.shape[0] 
    model.metric = None #TODO 

    exp_name = "singleBaseline"
    if not os.path.exists(os.path.join("exp", exp_name)):
        os.makedirs(os.path.join("exp", exp_name)) 
    exp_name = os.path.join(os.path.join("exp", exp_name), exp_name)

    train_dl = DataLoader(ds_train, shuffle=True, batch_size=model.batch_size)
    test_dl = DataLoader(ds_test, shuffle=False)
    model = train(model)
    torch.save(model.state_dict(), open(exp_name + '.pth', "wb")) #! 训练则开

    # model.load_state_dict(torch.load(open(exp_name + '.pth', "rb"))) #! 只测试则开 
    
    anomaly_score = []
    GT_score = []

    model.eval() 
    for idx, X in enumerate(test_dl, 1): 
        X_pred = model(X)  
        loss = model.loss_func(X_pred, X)             
        anomaly_score.append(loss.detach().item())
        GT_score.append(y_test[idx - 1])

    # torch.save(model.state_dict(), open(exp_name + '.pth', "wb"))
    with open(exp_name + '.txt', "w") as f:
        for score, score_ in zip(anomaly_score, GT_score):
            f.write(str(score) + ' ' + str(score_) + '\n')

    
