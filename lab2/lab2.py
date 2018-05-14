# -*- coding:utf-8 -*-
import csv
import numpy as np
import math
import pylab as pl
with open('iris.csv')as csvfile:
    reader=csv.reader(csvfile, delimiter=',')
    # dates0-9是普通的list
    dates1 = [];dates2 = [];dates3 = [];dates4 = []
    for row in reader:
        # 第一次实习第一部分等使用
        date1 = float(row[0]);date2 = float(row[1]);date3 = float(row[2]);date4 = float(row[3])
        dates1.append(date1);dates2.append(date2);dates3.append(date3);dates4.append(date4)

datess1=np.array(dates1);datess2=np.array(dates2);datess3=np.array(dates3);datess4=np.array(dates4)
b=[];d=[]
i=0
while(i<150):
    b.append(datess1[i]); b.append(datess2[i]); b.append(datess3[i]); b.append(datess4[i])
    d.append(b)
    i=i+1
    b=[]
m=0.0001
h=input("请输入edge length--h:")
n=input("请输入minimum density threshold:")
#print(d);print(h);print(n);print(m)
g=np.array(d)
#print(g);print(h);print(n);print(m)
q=input("请输入densityconnected--dbscan算法的e:")
w=input("请输入densityconnected--bdscan算法的minupts:")


#d为维数4
#FINDATTRACTOR函数中计算xt+1的公式的上面的计算公式
def KXIA(x,h,D,d):
   f=0
   for y in D:
        f=f+(1/math.pow(2*math.pi,d/2))*math.exp(-1/(2*h*h)*((x-y)*(x-y)).sum())
   return f
#FINDATTRACTOR函数中计算xt+1的公式的下面的计算公式
def KSHANG(x,h,D,d):
    c=0
    for y in D:
        c=c+(1/math.pow(2*math.pi,d/2))*math.exp(-1/(2*h*h)*((x-y)*(x-y)).sum())*y
    return c
#定义密度估计函数f
def f(x,D,d,h):
    a=1/(150*math.pow(h,d))*KXIA(x,h,D,d)
    return a
#定义denclue的总函数
def DENCLUE(D,h,n,m):
    A = []
    B = []
    for i in D:
        x=FINDATTRACTOR(i,D,h,m)
        if(f(x,D,4,h)>=n):
            A.append(x)
            B.append(i)
    #print(A)
    #print(B)
    return(A,B)
#定义寻找吸引子的FINDATTRACTOR函数
def FINDATTRACTOR(i,D,h,m):
    t=0
    T=[]
    T.append(i)
    a=KSHANG(i,h,D,4)/KXIA(i,h,D,4)
    T.append(a)
    t=t+1
    TT=np.array(T)
    while(math.sqrt(((TT[t]-TT[t-1])*(TT[t]-TT[t-1])).sum())>m):
        n=KSHANG(TT[t],h,D,4)/KXIA(TT[t],h,D,4)
        T.append(n)
        TT=np.array(T)
        t=t+1
    #print(TT[t])
    return TT[t]

#C1 D1为密度吸引子的集合、和密度吸引子相关的原数据集合
C1=DENCLUE(g,h,n,m)[0]
D1=DENCLUE(g,h,n,m)[1]
C2=np.array(C1)
D2=np.array(D1)
#print(C2)
#print(D2)

dataset1=[(float(C2[i][0]),float(C2[i][1]),float(C2[i][2]),float(C2[i][3])) for i in range(0,149,1)]
dataset2=[(float(D2[i][0]),float(D2[i][1]),float(D2[i][2]),float(D2[i][3])) for i in range(0,149,1)]
#print(dataset1)
#print(dataset2)

#计算欧几里得距离,a,b分别为两个元组
def dist(a, b):
    return math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2))

#算法模型
#初始化核心对象集合T为空，遍历一遍样本集D中所有的样本，计算每个样本点的ε-邻域中包含样本的个数，如果个数大于等于MinPts，则将该样本点加入到核心对象集合中。初始化聚类簇数k = 0， 初始化未访问样本集和为P = D。
#当T集合中存在样本时执行如下步骤：
#记录当前未访问集合P_old = P
#从T中随机选一个核心对象o,初始化一个队列Q = [o]
#P = P-o(从T中删除o)
#当Q中存在样本时执行：
#取出队列中的首个样本q
#计算q的ε-邻域中包含样本的个数，如果大于等于MinPts，则令S为q的ε-邻域与P的交集，
#Q = Q+S, P = P-S
#k = k + 1,生成聚类簇为Ck = P_old - P
#T = T - Ck
#划分为C= {C1, C2, ……, Ck}
def DBSCAN(D, e, Minpts):
    #初始化核心对象集合T,聚类个数k,聚类集合C, 未访问集合P,
    T = set(); k = 0; C = []; P = set(D)
    for d in D:
        #计算所有点中的core点
        J=[ i for i in D if dist(d, i) <= e]
        if len(J) >= Minpts:
            T.add(d)
    N=[]
    #开始聚类
    while len(T):
        P_old = P
        o = list(T)[np.random.randint(0, len(T))]
        P = P - set(o)
        Q = []; Q.append(o)
        M = []
        while len(Q):
            q = Q[0]
            Nq=[]
            for i in D:
                if dist(q,i)<=e:
                    Nq.append(i)
                    M.append(D.index(i))
            if len(Nq) >= Minpts:
                S = P & set(Nq)
                Q += (list(S))
                P = P - S
            Q.remove(q)
        N.append(M)
        k += 1
        Ck = list(P_old - P)
        T = T - set(Ck)
        C.append(Ck)
    return C,N
#画图
def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    markervalue=['x','x','x','x','x','x','x','+','+','+','+','+','+']
    for i in range(len(C)):
        coo_X = []    #x坐标列表
        coo_Y = []    #y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker=markervalue[i], color=colValue[i%len(colValue)], label=i)

    pl.legend(loc='upper right')
    pl.show()

#C,S = DBSCAN(dataset1, 0.12, 5)
C,S=DBSCAN(dataset1,q,w)
draw(C)
#for j in C:
#    print(j)
#print("|||||||||")
#for j in S:
 #   m=list(set(j))
 #   m.sort()
  #  print(m)
  #  l = []
  #  for k in m:
  #      l.append(dataset2[k])
  #  print(l)

print("分类的类簇数目和每个类簇的尺寸")
h=0
for j in C:
    print(h)
    print(len(j))
    h=h+1

print("按照每个类其对应的密度吸引子数组")
g=0
for j in C:
    print(g)
    print(j)
    g=g+1

print("每个类簇的纯度")
x=0
for j in C:
    print(x)
    print((len(j)/50.0))
    x=x+1