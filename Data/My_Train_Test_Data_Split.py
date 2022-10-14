from sklearn.model_selection import train_test_split
# from My_Tools import open_data_pickle
import pickle
import os

def open_data_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data

filedir = r'' # 数据存储路径
filename = os.path.join(filedir, 'My_Pytorch_Data_My_Model_10s.pickle')

data = open_data_pickle(filename)

Train_data, Test_data = train_test_split(data, test_size=0.20, random_state=7)

with open(os.path.join(filedir, 'My_Pytorch_Data_My_Model_10s_train.pickle'), 'wb') as f:
    pickle.dump(Train_data, f)

with open(os.path.join(filedir, 'My_Pytorch_Data_My_Model_10s_test.pickle'), 'wb') as f:
    pickle.dump(Test_data, f)