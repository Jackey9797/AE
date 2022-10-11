#!/usr/bin/env python
# coding: utf-8

# In[11]:


import matplotlib.pyplot as plt
import pandas as pd 
import os 

file_dir = os.path.abspath(__file__) # py
# file_dir = "dataset.py" # ipynb

base_dir = os.path.dirname(file_dir)

def get_df_data():
    dirs = os.listdir(base_dir + "/feature")
    df = None 
    flag = False

    for dir in dirs: 
        if dir[:4] == 'part': 
            df_ = pd.read_csv(base_dir + "/feature/" + dir) 
            if flag == False : df = df_; flag = True 
            else : df = pd.concat([df, df_], axis=0) 
    return df 


# In[ ]:


df = get_df_data()

# 查看每一周数据中的的异常数
# for i in range(1, 71):
#     df_w2 = df[df['week_id'] == i] 
#     try : 
#         print(df_w2['insider'].value_counts()[1])
#     except: 
#         print(i, "all normal")


# In[ ]:


def generate_data():
    proc = lambda x: x.drop(['user', 'day', 'week_id', 'starttime', 'endtime', 'insider'], axis=1) ##TODO  sessionid
    normalize = lambda x: x.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6) )

    df = get_df_data()  

    df_w1 = df[df['week_id'] == 22] 
    df_w2 = pd.concat([df[df['week_id'] == 23], df[df['week_id'] == 24], df[df['week_id'] == 25]] , axis=0)

    y_test = df_w2['insider'].to_numpy()

    df_w1 = normalize(df_w1)
    df_w2 = normalize(df_w2)

    x_train = proc(df_w1).to_numpy()
    x_test = proc(df_w2).to_numpy()

    return x_train, x_test, y_test   

def generate_stream_data():
    proc = lambda x: x.drop(['user', 'day', 'week_id', 'starttime', 'endtime', 'insider'], axis=1) ##TODO  sessionid
    normalize = lambda x: x.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6) )

    df = get_df_data()  
    df.sort_values(by="week_id" , inplace=True, ascending=True)
    # df_w2 = pd.concat([df[df['week_id'] == 22] , df[df['week_id'] == 23], df[df['week_id'] == 24], df[df['week_id'] == 25]] , axis=0)
    df_w2 = df #! 改这里 

    y_data = df_w2['insider'].to_numpy()

    df_w2 = normalize(df_w2)

    x_data = proc(df_w2).to_numpy()

    return x_data, y_data   


# In[ ]:

import torch
class oneWeek: 
    def __init__(self, data) -> None:
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32) 

    def __len__(self): 
        return len(self.data) 

    def __getitem__(self, idx): 
        return self.data[idx]

class manyWeek: 
    def __init__(self, data, label) -> None:
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32) 
        self.label = torch.tensor(label, dtype=torch.uint8)
        # self.data = self.data.reshape(-1, 1, 502)
        # self.label = self.label.reshape(-1, 1)

    def __len__(self): 
        return len(self.label) 

    def __getitem__(self, idx):
        # self.data[idx].unsqueeze(0) 
        # print(idx, self.data[idx].shape)
        return self.data[idx], self.label[idx]  ##TODO dataset test

# In[ ]:


# get_ipython().system('jupyter nbconvert --to script dataset.ipynb')

