import numpy as np
import pickle
# from geopy.distance import vincenty
from geopy import distance
import os
import math
from tqdm import *

filedir = r'C:\Users\Administrator\Desktop\zhangchi\My TMI Project\operating console'
filename = os.path.join(filedir, 'Revised_Trajectory_Label_Array.pickle')

with open(filename, 'rb') as f:
    Trajectory_Label_Array = pickle.load(f)

Total_30sec_Instance = []
Total_20sec_Instance = []
Total_10sec_Instance = []
Total_1sec_Instance = []

def cut_seg_in_nsec(seg, n_time):
    start_time = seg[0][2]
    out_seg = []
    out_seg.append(seg[0])
    for i in range(len(seg) - 1):
        if seg[i + 1][2] - seg[i][2] < 0.:
            continue
        end_time = seg[i + 1][2]
        seg_delta_time = (end_time - start_time) * 24. * 3600
        if seg_delta_time < n_time - 2.:
            pass
        elif seg_delta_time >= n_time - 2.:
        # else:
            out_seg.append(seg[i + 1])
            start_time = end_time

    return out_seg

for z in tqdm(range(len(Trajectory_Label_Array))):
    Descriptive_Stat = []
    Data = Trajectory_Label_Array[z]  # 一个人的全部轨迹
    if len(Data) == 0:
        continue

    delta_time = []
    for i in range(len(Data) - 1):  # 主要得到的是相对时间
        delta_time.append((Data[i+1, 2] - Data[i, 2]) * 24. * 3600)
        if delta_time[i] == 0:
            # 防止产生无限速度。所以使用非常短的时间=0.1秒。
            delta_time[i] = 0.1
        # A = (Data[i, 0], Data[i, 1])
        # B = (Data[i + 1, 0], Data[i + 1, 1])
        # tempSpeed.append(vincenty(xi, xj).meters/sum_time[i])
        # tempSpeed.append(distance.distance(A, B).meters / delta_time[i])
    delta_time.append(3)
    InstanceNumber = []
    # Label: For each created instance, we need only one mode to be assigned to.
    # Remove the instance with less than 10 GPS points. Break the whole user's trajectory into trips with min_trip
    # Also break the instance with more than threshold GPS points into more instances
    Data_All_Instance = []  # Each of its element is a list that shows the x for each instance (lat, long, time)
    Data_30sec_Instance = []
    Data_20sec_Instance = []
    Data_10sec_Instance = []
    Data_1sec_Instance = []
    Label = []
    min_trip_time = 20 * 60  # 20 minutes equal to 1200 seconds
    threshold = 200  # fixed of number of GPS points for each instance
    i = 0
    while i <= (len(Data) - 1):
        No = 0
        ModeType = Data[i, 3]
        Counter = 0  # 轨迹点的长度
        # 索引：在创建实例时保存实例索引，并在删除中串联所有索引
        index = []
        # 首先，我们始终有一个具有一个GPS点的实例。
        sum_delta_time = 0.
        avg_delta_time = 0.
        while i <= (len(Data) - 1) and Data[i, 3] == ModeType:  # and Counter < threshold:
            if delta_time[i] <= min_trip_time and delta_time[i] > 0:
                Counter += 1
                index.append(i)
                sum_delta_time += delta_time[i]
                i += 1
            # elif delta_time[i] < 0:
            #     i += 1
            #     break
            else:
                Counter += 1
                index.append(i)
                i += 1
                break
        if Counter > 0:
            avg_delta_time = sum_delta_time / Counter

        if Counter >= 25:  # Remove all instances that have less than 10 GPS points# I
            InstanceNumber.append(Counter)
            Data_For_Instance = [Data[i, 0:4] for i in index]  # 连标签一块传进去
            # Data_For_Instance = np.array(Data_For_Instance, dtype=float)
            Data_All_Instance.extend(Data_For_Instance)  # 这里换成两个列表拼接到一块而不是大包小
            # Label.append(ModeType)
            if avg_delta_time > 27. and avg_delta_time < 33.:
                Data_30sec_Instance.extend(Data_For_Instance)
            elif avg_delta_time > 17 and avg_delta_time < 23:
                Data_20sec_Instance.extend(Data_For_Instance)
            elif avg_delta_time > 7 and avg_delta_time < 13:
                Data_10sec_Instance.extend(Data_For_Instance)
            elif avg_delta_time > 0 and avg_delta_time < 3:
                Data_1sec_Instance.extend(Data_For_Instance)
                Data_30sec_Instance.extend(cut_seg_in_nsec(Data_For_Instance, 30))
                Data_20sec_Instance.extend(cut_seg_in_nsec(Data_For_Instance, 20))
                Data_10sec_Instance.extend(cut_seg_in_nsec(Data_For_Instance, 10))
                # 在这分别切成不同的时间间隔而不是下边再切，确保顺序
        if len(InstanceNumber) == 0:
            continue
    Total_30sec_Instance.append(np.array(Data_30sec_Instance, dtype=np.float64))
    Total_20sec_Instance.append(np.array(Data_20sec_Instance, dtype=np.float64))
    Total_10sec_Instance.append(np.array(Data_10sec_Instance, dtype=np.float64))
    Total_1sec_Instance.append(np.array(Data_1sec_Instance, dtype=np.float64))


filedir = r'C:\Users\Administrator\Desktop\zhangchi\0707,length=200,baseline,data'

with open(os.path.join(filedir, 'Data_30sec_For_Instance.pickle'), 'wb') as f:
    pickle.dump(Total_30sec_Instance, f)

with open(os.path.join(filedir, 'Data_20sec_For_Instance.pickle'), 'wb') as f:
    pickle.dump(Total_20sec_Instance, f)

with open(os.path.join(filedir, 'Data_10sec_For_Instance.pickle'), 'wb') as f:
    pickle.dump(Total_10sec_Instance, f)

with open(os.path.join(filedir, 'Data_1sec_For_Instance.pickle'), 'wb') as f:
    pickle.dump(Total_1sec_Instance, f)
