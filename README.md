# 基于多尺度相位聚合轨迹表示的出行方式识别模型
## 论文摘要
现有出行方式识别模型常依赖在数据采集设备中插入额外传感器元件或提高设备抽样频率的方法提升识别的准确率。但现实中提高设备采样率或增加传感器的做法会增加采样设备的耗能。并且，采样设备的采集频率设置难以统一的现实情况也同样影响出行方式检测模型的准确率。针对这样的问题，提出多尺度相位聚合-深层神经决策森林模型（Multi-scale Phase Aggregation-Deep Neural Decision Forests Model，MPA-NDF）。提取轨迹数据中的多尺度局部和全局特征令牌，采用卷积神经网络和相位检测令牌混合算法得到多尺度相位聚合的轨迹表示，利用深层神经决策森林算法得到出行方式分类结果。上述模型在低频混合抽样轨迹数据上能够完成更有效的出行方式分类工作。实验结果表明，与次优的随机森林模型对比MPA-NDF模型在三种低频重采样数据集上均有更高的分类准确率，分别提升了4.799、3.331和0.048个百分点，且平均准确率提升了2.726个百分点，具有更高的出行方式识别准确性。
## 代码文件
### 数据预处理过程
1. '/Data'本文存储路径
2. '/Data/Combined Trajectory_Label_Geolife'原始轨迹文件存储路径
3. 'LabelMatrix-TimeDays-TrajectrotyMatrix.py'添加标签
4. 'My_baselinedata_Trajectory_Split.py'将原数据集重采样低频重采样数据集
5. 'My_Instance_Creation.py'提取统计学特征
6. 'My_Pytorch_Data_Creation.py'生成全局和局部向量
7. 'My_Train_SubTrain_Data_Split.py'切分训练集出一个子集，便于测试训练集准确率
8. 'My_Train_Test_Data_Split.py'将数据拆分成训练集和测试集
9. 'Pytorch_Data_Mix.py'混合三种低频数据集
### 模型训练与测试
1. 'My_Model.py'模型文件
2. 'My_Opt.py'参数配置文件
3. 'ndf.py'ndf层代码
4. 'My_Test_Code.py'模型测试代码
5. 'My_Train_Code.py'模型训练代码
6. 'My_Tools.py'工具性函数
7. 'My_Model_Weightdict.mdl'最佳模型参数
### 引用声明
* 数据预处理过程中的部分代码改编自[这里](https://github.com/sinadabiri/Transport-Mode-GPS-CNN.git)
* 模型训练与测试中PATM算法来自[这里](https://github.com/huawei-noah/CV-Backbones/tree/master/wavemlp_pytorch)
* 模型训练与测试中ndf.py引用自论文
**KONTSCHIEDER P, FITERAU M, CRIMINISI A, et al. Deep neural decision forests[C]//Proceedings of the IEEE international conference on computer vision. 2015: 1467-1475.**
