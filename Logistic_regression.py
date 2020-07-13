#!/usr/bin/env python
# coding: utf-8

# # Logistic_regression

# ## 1.加载数据

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


pimaForm = np.loadtxt('Pima.csv', dtype=np.float64, delimiter=',')   # 加载整个表格
data = pimaForm[:,:8].T   # 数据部分
label = pimaForm[:,8:].astype(np.int16)   # 标签部分


# In[3]:


print(data.shape, label.shape)


# ## 2.数据预处理

# ### 2.1 缺失值处理
# 这里缺失值处理采用填充均值

# In[4]:


dataMean = data.mean(axis=1)    # 计算均值，用于缺失值填充
dataStd = data.std(axis=1)      # 计算标准差，用于后续数据处理


# In[5]:


for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i][j] == 0:    # 出现缺失值就用均值填充
            data[i][j] = dataMean[i]


# ### 2.2 输入归一化
# 认为输入数据为正态分布

# In[6]:


data = ((data.T - dataMean)/dataStd).T  # 使用z-score标准化方法


# ### 2.3 数据集划分
# 训练集和测试集比例按照5：1

# In[7]:


testData = data[:,:128]
testLabel = label[:128]
trainData = data[:,128:]
trainLabel = label[128:]
print(testData.shape, testLabel.shape, trainData.shape, trainLabel.shape)


# ## 3.初始化参数

# In[8]:


def initParameter(feature_num):
    w = np.random.rand(1, feature_num)   # w:1xfeature_num, 用0-1随机数初始化
    b = 0    # b初始化为0
    return w,b


# In[9]:


test_w, test_b = initParameter(8)   # 测试一下
print(test_w.shape, test_b)


# ## 4.向前传播

# In[10]:


def forward(data, w, b):
    z = np.dot(w, data)   # 计算线性部分
    a = 1/(1+np.exp(-z))  # 计算预测结果
    return a.T


# In[11]:


a = forward(trainData, test_w, test_b)
print(a.shape)


# ## 5.梯度下降

# In[12]:


def gradDescent(w, b, x , y, a, learning_rate):
    w = w - learning_rate * np.dot(x, a-y).T / a.shape[0]   # 更新参数w
    b = b - learning_rate * (a-y).sum(axis=0)[0]   # 更新参数b
    return w,b


# In[13]:


test_w2, test_b2 = gradDescent(test_w, test_b, trainData, trainLabel, a, 0.01)


# In[14]:


print(test_w2.shape, test_b2)


# ## 6.开始训练

# In[15]:


w, b = initParameter(8)   # 测试一下
cost = []
for i in range(200):
    a = forward(trainData, w, b)
    w, b = gradDescent(w, b, trainData, trainLabel, a, 0.4)
    cost.append((-(trainLabel*np.log(a)+(1-trainLabel)*np.log(1-a))).sum(axis = 0)[0]/a.shape[0])


# In[16]:


plt.plot(cost)


# In[17]:


testPred = (forward(testData, w, b)>0.5).astype(np.int16)


# In[18]:


print("测试集正确率："+str((testPred==testLabel).astype(np.int16).reshape(-1).sum()/testPred.shape[0]))


# In[ ]:




