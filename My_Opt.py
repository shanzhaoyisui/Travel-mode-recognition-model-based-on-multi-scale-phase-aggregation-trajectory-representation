import torch.nn as nn


opt = {
    'train_data_path': r'\My_Pytorch_Data_My_Model_mixed_train.pickle',  # 训练集存储路径
    'test_data_path': r'\My_Pytorch_Data_My_Model_mixed_test.pickle',  # 测试集存储路径
    'val_data_path': r'\My_Pytorch_Data_My_Model_mixed_test.pickle',  # 验证集集存储路径
    'subtrain_data_path': r'\My_Pytorch_Data_My_Model_mixed_train.pickle',  # 训练集子集
    'load_state_dict': r'\0718 4th weight mixed data',  # 测试权重加载路径
    'save_state_dict': r'\weight',  # 训练权重存储路径
    'weight_logs': r'\weight_logs.csv',  # 训练结果存储位置
    'lr': 0.0001, 
    'n_class': 5,  
    'epochs': 100,
    'batch_size': 64,
    'device': 'cuda',
    'load_weights': 0,  
    'attn_in_dim': 1,
    'attn_out_dim': 1,
    'conv_out_dim': 32,
    'mlp_hidden_dim': 32,  
    'n_in_feature': 60,  
    'n_tree': 80,
    'tree_depth': 10,
    'fc_out_dim': 60,
    'tree_feature_rate': 1.0,
    'jointly_training': True,
    'mlp_ratio': 4.,
    'qkv_bias': False,
    'qk_scale': None,
    'drop': 0.,
    'attn_drop': 0.,  
    'drop_path': 0.1, 
    'drop_out': 0., 
    'act_layer': nn.GELU,
    'norm_layer': nn.BatchNorm2d,
    'mode': 'conv',
    'feature_num': 32,  
    'full_connect_mid_dim_1': 16,
    'num_attn_layer': 1  

}
