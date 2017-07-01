# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 21:27:41 2017

@author: user
"""

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import *
from pybrain.tools.xml import NetworkWriter,NetworkReader
import jqdata
import numpy as np
import matplotlib.pyplot as plt 
#DATA
stocks =['300072.XSHE','000413.XSHE','000503.XSHE','002310.XSHE','000839.XSHE','600547.XSHG', '600549.XSHG','000839.XSHE','600547.XSHG','600549.XSHG']
start_date='2015-07-01'
end_date='2015-08-31'
start_date_test='2015-09-01'
end_date_test='2015-09-30'
def pric(stock):
    #start_date='2015-07-01'
    #end_date='2015-8-31'
    return get_price(stock, start_date, end_date, frequency='daily')['open']
def pric_test(stock):
    #start_date='2015-09-01'
    #end_date='2015-9-31'
    return get_price(stock, start_date_test, end_date_test, frequency='daily')['open']
#建立樣本標籤
def cla_stock(df_stock):
    l=len(df_stock)
    a=np.zeros((l-4,6))
    for i in range(4,l-1):
        c=(df_stock[i+1]-df_stock[i])/df_stock[i]
        if (c>0.05):
            a[i-4,:]=np.array([1,0,0,0,0,0])
            print(i,'ok')
        elif(0.05>c>0.04):
            a[i-4,:]=np.array([0,1,0,0,0,0])
        elif(0.04>c>0.03):
            a[i-4,:]=np.array([0,0,1,0,0,0])
        elif(0.03>c>0.02):
            a[i-4,:]=np.array([0,0,0,1,0,0])
        elif(0.02>c>0.00):
            a[i-4,:]=np.array([0,0,0,0,1,0])
        else:
            a[i-4,:]=np.array([0,0,0,0,0,1])
    return a
def normal_data(df_stock):
    l=len(df_stock)
    a=np.zeros((l-1,1))
    for i  in range(l-1):
        a[i]=(df_stock[i+1]-df_stock[i])/df_stock[i]
    return a

stock2 = '601398.XSHG'

df_stock2 = get_price(stock2, start_date, end_date, frequency='daily')['open']
x1=normal_data(pric(stocks[0]))
x2=normal_data(pric(stocks[1]))
x3=normal_data(pric(stocks[2]))
x4=normal_data(pric(stocks[3]))
x5=normal_data(pric(stocks[4]))
x6=normal_data(pric(stocks[5]))
x7=normal_data(pric(stocks[6]))
x8=normal_data(pric(stocks[7]))
x9=normal_data(pric(stocks[8]))
x10=normal_data(pric(stocks[9]))
# 建立神經網路fnn
fnn =  RecurrentNetwork()

# 设立五層，第一層為輸入層，第二到四層為隱藏層，第五層為softmax輸出層
fnn.addInputModule(LinearLayer(30, name='inLayer'))
fnn.addModule(SigmoidLayer(50, name='hiddenLayer0'))
fnn.addModule(SigmoidLayer(40, name='hiddenLayer1'))
fnn.addModule(SigmoidLayer(30, name='hiddenLayer2'))
fnn.addOutputModule(SoftmaxLayer(6, name='outLayer'))

fnn.addConnection(FullConnection(fnn['inLayer'], fnn['hiddenLayer0'], name='c1'))
fnn.addConnection(FullConnection(fnn['hiddenLayer0'], fnn['hiddenLayer1'], name='c2'))
fnn.addConnection(FullConnection(fnn['hiddenLayer1'], fnn['hiddenLayer2'], name='c3'))
fnn.addConnection(FullConnection(fnn['hiddenLayer2'], fnn['outLayer'], name='c4'))
#建立隱藏層遞迴關係
fnn.addRecurrentConnection(FullConnection(fnn['hiddenLayer0'], fnn['hiddenLayer0'], name='c5'))
fnn.addRecurrentConnection(FullConnection(fnn['hiddenLayer1'], fnn['hiddenLayer1'], name='c6'))
fnn.addRecurrentConnection(FullConnection(fnn['hiddenLayer2'], fnn['hiddenLayer2'], name='c7'))

fnn.sortModules()

re=cla_stock(df_stock2)
DS = SupervisedDataSet(30,6)

for i in range(len(re[:,1])):
        DS.addSample([x1[i],x1[i+1],x1[i+2],x2[i],x2[i+1],x2[i+2],x3[i],x3[i+1],x3[i+2],x4[i],x4[i+1],x4[i+2],x5[i],x5[i+1],x5[i+2],\
                      x6[i],x6[i+1],x6[i+2],x7[i],x7[i+1],x7[i+2],x8[i],x8[i+1],x8[i+2],x9[i],x9[i+1],x9[i+2],x10[i],x10[i+1],x10[i+2]]\
                     ,[re[i,0],re[i,1],re[i,2],re[i,3],re[i,4],re[i,5]])
#倒傳遞訓練神經網路
trainer = BackpropTrainer(fnn, DS, momentum = 0.1,verbose=True, learningrate=0.01)
trainer.trainUntilConvergence(maxEpochs=3000)

#已非訓練樣本測試神經網路預測正確度
df_stock2_test = get_price(stock2, start_date_test, end_date_test, frequency='daily')['open']
re_test=cla_stock(df_stock2_test)
x1_t=normal_data(pric_test(stocks[0]))
x2_t=normal_data(pric_test(stocks[1]))
x3_t=normal_data(pric_test(stocks[2]))
x4_t=normal_data(pric_test(stocks[3]))
x5_t=normal_data(pric_test(stocks[4]))
x6_t=normal_data(pric_test(stocks[5]))
x7_t=normal_data(pric_test(stocks[6]))
x8_t=normal_data(pric_test(stocks[7]))
x9_t=normal_data(pric_test(stocks[8]))
x10_t=normal_data(pric_test(stocks[9]))
l2=len(re_test[:,1])
pre=np.zeros((l2,6))
err=np.zeros((l2,1))
succ=0
for i in range(l2):
    # 预测的X2的输出值
    xx2=[x1_t[i],x1_t[i+1],x1_t[i+2],x2_t[i],x2_t[i+1],x2_t[i+2],x3_t[i],x3_t[i+1],x3_t[i+2],x4_t[i],\
                       x4_t[i+1],x4_t[i+2],x5_t[i],x5_t[i+1],x5_t[i+2],x6_t[i],x6_t[i+1],x6_t[i+2],x7_t[i],x7_t[i+1],\
                       x7_t[i+2],x8_t[i],x8_t[i+1],x8_t[i+2],x9_t[i],x9_t[i+1],x9_t[i+2],x10_t[i],x10_t[i+1],x10_t[i+2]]                      
    pre[i,:] = fnn.activate(xx2)
    print(pre[i,:])
    ma=0
    ma_n=pre[i,0]
    for j in range(5):
        if (pre[i,j+1]>ma_n):
            ma=j+1
            ma_n=pre[i,j+1]
    pre[i,:]=np.array([0,0,0,0,0,0])
    pre[i,ma]=1
    print(pre[i,:])
    print(re_test[i,:])
    err[i]=sum(np.dot((pre[i,:]-re_test[i,:]),(pre[i,:]-re_test[i,:])))
    #print(int(err[i]))
    if (int(err[i])==0):
        succ=succ+1
        print(succ)
plt.plot(err,'r-')
print('rate:',succ/l2)
#write the network using xml file     
NetworkWriter.writeToFile(fnn,'myneuralnet.xml')
