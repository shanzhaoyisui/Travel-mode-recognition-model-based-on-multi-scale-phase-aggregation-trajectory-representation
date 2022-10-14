import numpy as np
import pickle
# from My_Tools import local_feature_extra, global_feature_extra
import os
from tqdm import *
import math
# from scipy.signal import savgol_filter
# import math

# filedir = r'C:\Users\Administrator\Desktop\zhangchi\My TMI Project\operating console'
filedir = r'C:\Users\Administrator\Desktop\zhangchi\0707,length=200,baseline,data'
filename = '10s_Revised_InstanceCreation+NoJerkOutlier+Smoothing.pickle'
# 以下每个变量包含多个列表，每个列表属于一个用户
with open(os.path.join(filedir, filename), 'rb') as f:
    Total_RelativeDistance, Total_Speed, Total_Acceleration, Total_Jerk, Total_BearingRate, Total_Label,\
    Total_InstanceNumber, Total_Instance_InSequence, Total_Delta_Time, Total_Velocity_Change, Total_Day_of_Week, Total_Start_Time = pickle.load(f, encoding='latin1')


def alineof_seg_feature(segment, delta_segment):
    length = len(segment)
    # delta_segment = []

    # for i in range(length - 1):
    #     delta_segment.append(segment[i + 1] - segment[i])

    abs_delta_segment = [math.fabs(l) for l in delta_segment]
    segment = np.array(segment)
    delta_segment = np.array(delta_segment, dtype=np.float64)
    abs_delta_segment = np.array(abs_delta_segment, dtype=np.float64)
    # if len(segment) <= 1:
        # delta_segment = np.array([0])
        # abs_delta_segment = np.array([0])

    out = np.array([
        np.mean(segment),
        np.median(segment),
        np.max(segment),
        np.mean(delta_segment),
        np.median(delta_segment),
        np.max(delta_segment),
        np.mean(abs_delta_segment),
        np.median(abs_delta_segment),
        np.max(abs_delta_segment)
    ], dtype=np.float64)

    return out


def global_vectory(segment):
    segment_sorted = []
    segment_sorted = [l for l in segment if l not in segment_sorted or l != 0]
    segment_sorted = np.array(segment_sorted)
    segment_sorted = np.sort(segment_sorted)
    if len(segment) == 0:
        print('len(segment) == 0')  # 用完可以删掉
    out = np.array([
        np.mean(segment_sorted),
        np.median(segment_sorted),
        np.max(segment_sorted),
        np.std(segment_sorted),
        segment_sorted[math.floor(len(segment_sorted) * 0.10)],
        segment_sorted[math.floor(len(segment_sorted) * 0.25)],
        segment_sorted[math.floor(len(segment_sorted) * 0.50)],
        segment_sorted[math.floor(len(segment_sorted) * 0.75)],
        segment_sorted[math.floor(len(segment_sorted) * 0.90)]
    ])

    return out


def global_feature_extra(RD, SP, AC, J, BR, sum_time, day_of_week, start_time, length):
    out = np.zeros([4, 1, 9])
    out[0, 0] = global_vectory(SP)
    out[1, 0] = global_vectory(AC)
    # total_distance = np.sum(RD)
    abs_delta_bearing = [math.fabs(BR[i]) for i in range(len(BR))]
    out[2, 0] = global_vectory(abs_delta_bearing)
    out[3, 0, 0: 4] = np.array([
        np.sum(RD),
        sum_time,
        start_time,
        day_of_week
    ], dtype=np.float64)

    # abs_delta_bearing = [math.fabs(i) for i in ]


    return out


def local_feature_extra(RD, SP, AC, J, BR, length, seg_num):  # 用来提取特征向量转化成对应的9特征的向量
    # 2022年6月23日：这里的是最好用的
    out = np.zeros([4, seg_num, 9], dtype=np.float64)
    delta_SP = [SP[i + 1] - SP[i] for i in range(len(SP) - 1)]
    delta_SP.append(delta_SP[-1])
    delta_AC = [AC[i + 1] - AC[i] for i in range(len(AC) - 1)]
    delta_AC.append(delta_AC[-1])
    delta_J = [J[i + 1] - J[i] for i in range(len(J) - 1)]
    delta_J.append(delta_J[-1])
    delta_BR = [BR[i + 1] - BR[i] for i in range(len(BR) - 1)]
    delta_BR.append(delta_BR[-1])
    for i in range(seg_num):
        if (i + 1) * (length // seg_num) < length and i < seg_num - 1:
            end = (i + 1) * (length // seg_num)
        else:
            end = length
        speed = SP[i * (length // seg_num): end]
        delta_speed = delta_SP[i * (length // seg_num): end]
        acc = AC[i * (length // seg_num): end]
        delta_acc = delta_AC[i * (length // seg_num): end]
        jerk = J[i * (length // seg_num): end]
        delta_jerk = delta_J[i * (length // seg_num): end]
        bearing = BR[i * (length // seg_num): end]
        delta_bearing = delta_BR[i * (length // seg_num): end]
        out[0, i] = alineof_seg_feature(speed, delta_speed)
        out[1, i] = alineof_seg_feature(acc, delta_acc)
        out[2, i] = alineof_seg_feature(bearing, delta_bearing)
        out[3, i] = alineof_seg_feature(jerk, delta_jerk)

    return out


data = []
outline = []

for k in range(len(Total_InstanceNumber)):  # 两个人
    # 为每个用户创建4个通道的形状
    # 有4个通道(顺序为：Single_Globel_RelativeDistance、Single_GLOBAL_SPEED、Acceleration、BearingRate)
    # 取出的通道之后进行局部和全局特征提取
    RD = Total_RelativeDistance[k]  # 一个人全部轨迹[轨迹数, T]
    SP = Total_Speed[k]
    AC = Total_Acceleration[k]
    J = Total_Jerk[k]
    BR = Total_BearingRate[k]
    LA = Total_Label[k]  # 第k个轨迹段
    Day_of_Week = Total_Day_of_Week[k]
    Delta_Time = Total_Delta_Time[k]
    Start_Time = Total_Start_Time[k]
    # seg_length: the instances and number of GPS points in each instance for each user k
    seg_length = Total_InstanceNumber[k]  # 轨迹长度
    seg_num = len(seg_length)

    for i in tqdm(range(len(seg_length))):  # 选择其中一个轨迹
        if seg_length[i] > 25 and LA[i] !=5 and LA[i] != 6:
            one_segment = [[], [], []]  # 里边装的是同一个轨迹段提取出来的每一个特征值(顺序为：RelativeDistance、Speed、Acceleration、BearingRate)
            length = seg_length[i]
            day_of_week = Day_of_Week[i]
            start_time = Start_Time[i]
            sum_time = np.sum(np.array(Delta_Time[i]))
            if length == 0 or sum(RD[i]) == 0:
                continue
            cut_in_8_seg = local_feature_extra(RD[i], SP[i], AC[i], J[i], BR[i], length, 8)  # [4个特征, 8_seg, 9]
            cut_in_6_seg = local_feature_extra(RD[i], SP[i], AC[i], J[i], BR[i], length, 6)
            cut_in_4_seg = local_feature_extra(RD[i], SP[i], AC[i], J[i], BR[i], length, 4)
            one_segment[0].append(cut_in_8_seg)
            one_segment[0].append(cut_in_6_seg)
            one_segment[0].append(cut_in_4_seg)
            global_feature = global_feature_extra(RD[i], SP[i], AC[i], J[i], BR[i], sum_time, day_of_week, start_time, length)
            one_segment[1].append(global_feature)
            one_segment[2].append(LA[i])
            data.append(one_segment)
        else:
            outline_in = [k, i]
            outline.append(outline_in)

# filedir = r'C:\Users\Administrator\Desktop\zhangchi\0707,length=200,864,data'

with open(os.path.join(filedir, 'My_Pytorch_Data_My_Model_10s.pickle'), 'wb') as f:
    pickle.dump(data, f)

with open(os.path.join(filedir, 'outline_10s.pickle'), 'wb') as f:
    pickle.dump(outline, f)
