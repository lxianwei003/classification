# classification
数值型特征样本数据集，利用神经网络进行二分类


classicaton.py 是基于三层一般神经网络的分类，准确率0.933

classication_cnn.py是基于cnn进行的分类，不稳定（原因还有待分析），好的时候是0.85左右，

train.csv是训练集，test.csv是测试集，两个是通过Excel处理成（data，labels）标签后按照3:1的比例从train_x.csv中得到的 

运行环境：
tensorflow 1.0.1
numpy 1.12.1+mkl
pandas 0.19.2
python3.5

