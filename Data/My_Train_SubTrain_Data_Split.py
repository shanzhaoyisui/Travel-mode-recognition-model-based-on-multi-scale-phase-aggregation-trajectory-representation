from sklearn.model_selection import train_test_split
# from My_Tools import open_data_pickle
import pickle
import os

def open_data_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data

filedir = r''  # 数据存储路径
filename = os.path.join(filedir, 'My_Train_Data.pickle')

data = open_data_pickle(filename)

_, SubTrain_data = train_test_split(data, test_size=0.30, random_state=7)

# with open(os.path.join(filedir, 'My_Train_Data.pickle'), 'wb') as f:
#     pickle.dump(Train_data, f)

with open(os.path.join(filedir, 'My_SubTrain_Data.pickle'), 'wb') as f:
    pickle.dump(SubTrain_data, f)