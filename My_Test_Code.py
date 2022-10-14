import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from My_Model import get_model
# from My_Model_Residual import get_model
from My_Pytorch_Pretreatment import My_DataSet
from My_Opt import opt
from My_Tools import evalute_with_graph
from tqdm import *


def main():
    train_opt = opt
    train_opt['load_weights'] = 1
    train_opt['dropout'] = 0.1

    print('Loading DataSet...')
    # train_data = My_DataSet(train_opt['train_data_path'], train_opt['device'], train_opt['n_class'])
    train_opt['val_data_path'] = r'C:\Users\Administrator\Desktop\zhangchi\0707,length=200,baseline,data\My_Pytorch_Data_My_Model_30s.pickle'
    val_data = My_DataSet(train_opt['val_data_path'], train_opt['device'], train_opt['n_class'])
    # ------------训练测试集换---------------------
    # train_dataloader = DataLoader(train_data, batch_size=train_opt['batch_size'], drop_last=False, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=train_opt['batch_size'], drop_last=False, shuffle=True)
    # -------------------------------------------
    print('Finished!')
    print('Loading model...')
    model = get_model(train_opt)# .to(device=train_opt['device'])
    # model = model.load_state_dict(torch.load(r'C:\Users\Administrator\Desktop\zhangchi\0703My Best Model\operating console\0708 1st weight 83.669% 64c\5,83.669%,1.197611.mdl'))
    model = model.to(device=train_opt['device'], dtype=torch.float64)
    print('Finished!')
    print('Training Begin...')
    val_acc = evalute_with_graph(model, val_dataloader)
    print('val_acc:', ' {:.4%}'.format(val_acc))

    print('Finished!')


if __name__ == "__main__":
    main()