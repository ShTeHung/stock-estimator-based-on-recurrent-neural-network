# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 21:29:04 2017

@author: user
"""

# 导入函数库
import jqdata
import numpy as np
import matplotlib.pyplot as plt

stocklist = get_index_stocks('000016.XSHG')
start_date='2016-01-01'
end_date='2016-12-31'
stocks = get_index_stocks('000300.XSHG')
stock2 = '601398.XSHG'
#df_stock1 = get_price(stock1, start_date, end_date, frequency='daily')['open'] 
df_stock2 = get_price(stock2, start_date, end_date, frequency='daily')['open']

def crosscorrelation(st1,st2):
    n=len(st1)
    #print(st2[:10])
    #result=np.zeros((int(n/2),1))
    result=0
    st1_ave=sum(st1)/n
    st2_ave=sum(st2)/n
    #print('2')
    for i in range(10):
        a=0
        c=0
        for j in range(n-i):
            a=a+(st1[j]-st1_ave)*(st2[j+i]-st2_ave)
            c=c+1
        result=result+a/(c*st1_ave*st2_ave)
    return result
#sq,r=crosscorrelation(df_stock1,df_stock2)
re=np.zeros((len(stocks),1))
num=0
for s in stocks:
    df_stock1 = get_price(s, start_date, end_date, frequency='daily')['open']
    re[num]=crosscorrelation(df_stock1,df_stock2)
    num=num+1
nn=len(re)
li=[]
for i in range(nn):
    li.append((re[i],stocks[i]))
ss=sorted(li, key=lambda x : x[0],reverse=True)
s=ss[:30]
print(s)
re_x=np.zeros((30,1))
re_y=[]
for i in range(30):
    re_x[i]=s[i][0]
    re_y.append(s[i][1])
plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(30)
ax.barh(y_pos, re_x, align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(re_y)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')


plt.show()