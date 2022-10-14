from sklearn.model_selection import train_test_split
# from My_Tools import open_data_pickle
import pickle
import os

def open_data_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data

filedir = r''  # 数据存储路径
filename_10s = os.path.join(filedir, 'My_Pytorch_Data_My_Model_10s_test.pickle')
filename_20s = os.path.join(filedir, 'My_Pytorch_Data_My_Model_20s_test.pickle')
filename_30s = os.path.join(filedir, 'My_Pytorch_Data_My_Model_30s_test.pickle')

data_10s = open_data_pickle(filename_10s)
data_20s = open_data_pickle(filename_20s)
data_30s = open_data_pickle(filename_30s)

train_data = data_10s + data_20s + data_30s

with open(os.path.join(filedir, 'My_Pytorch_Data_My_Model_mixed_test.pickle'), 'wb') as f:
    pickle.dump(train_data, f)