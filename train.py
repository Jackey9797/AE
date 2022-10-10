import os
import argparse
import dill as pickle

import torch 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import random_split

def train_step(model, x, y): 
    model.train() 
    model.optimizer.zero_grad()

    pred = model(x) 

    loss = model.loss_func(pred, y) + 0.00015 * get_L2_loss(model) 
    metric = model.metric_func(pred, y)

    loss.backward() 

    model.optimizer.step() #? loss 和 optimize各自扮演的角色
    return loss.item(), metric.item()

def valid_step(model, x, y):
    model.eval() #? 实现

    with torch.no_grad(): 
        pred = model(x)
        loss = model.loss_func(pred, y)
        metric = model.metric_func(pred, y)
    return loss.item(), metric.item()

def train_model(model, epoch, train_dl, valid_dl, log_step_freq, valid=True, dir="checkpoints", plt_curve=False): 
    metric_name = model.metric_name 
    df_history = pd.DataFrame(columns = ["epoch", "loss", metric_name, "val_loss","val_" + metric_name])
    train_loss_list = []
    valid_loss_list = []
    best_ACC = 0.0
    print("Start Training...\n")

    for i in range(1, epoch + 1): 
        #*-----------------------train loop-----------------------------------------
        print("EPOCH %d Begins!" % i)

        cnt = 0
        loss_sum = 0 
        metric_sum = 0

        step = 1
        for step, (x, y) in enumerate(train_dl, 1): 
            loss, metric = train_step(model, x, y) 
            
            loss_sum += loss 
            metric_sum += metric 
            cnt += len(y) 

            if step % log_step_freq == 0: 
                print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") % 
                        (step, loss / len(y), metric / len(y)))  
        
        #*-----------------------valid loop-----------------------------------------
        val_loss_sum = 0
        val_metric_sum = 0
        val_step = 1 
        val_cnt = 0

        for val_step, (x, y) in enumerate(valid_dl, 1): 
            val_loss, val_metric = valid_step(model, x, y) 
            val_loss_sum += val_loss 
            val_metric_sum += val_metric 
            val_cnt += len(y)
        # print(loss_sum)
        
        #*-----------------------check info-----------------------------------------

        ATL = loss_sum / cnt 
        AVL = val_loss_sum / val_cnt
        T_ACC = metric_sum / cnt
        V_ACC = val_metric_sum / val_cnt

        info = (i, ATL, T_ACC, 
                AVL, V_ACC) 

        df_history.loc[epoch - 1] = info 

        if V_ACC > best_ACC: 
            best_ACC = V_ACC 
            torch.save(obj=model.state_dict(), f= dir + "/net.pth")

        # 打印epoch级别日志
        # print(cnt)
        train_loss_list.append(loss_sum / cnt)
        valid_loss_list.append(val_loss_sum / val_cnt)
        if plt_curve : 
            plt.plot(np.arange(i) + 1, train_loss_list, label="Train Loss")
            plt.plot(np.arange(i) + 1, valid_loss_list, label="Valid Loss")
            plt.legend()
            myfig = plt.gcf()
            myfig.savefig("Train.png")
            plt.show()

        else : print(("EPOCH = %d, Avg Train loss = %.6f , "+ "Train " + metric_name + \
            " = %.6f \n           Avg Valid loss = %.6f , " + "Valid " + metric_name + " = %.6f ")
            %info, "\n")

    
    print("Training Finished!\n")
    return df_history, train_loss_list, valid_loss_list, best_ACC