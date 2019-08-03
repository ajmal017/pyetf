import numpy as np
def prn_list(l):
    for t in range(len(l)):
        print(t, l[t])
        
# 时间序列循环
p = []
for t in range(100):
    p.append(t+1)
# 历史hln：每一段hln做一次操作
# 以[0:t]的数据，预测t+1的值，然后存到新list的t位置
# 例如，t=[0:14]的数据，预测t=15的值，然后存到新list的t=14位置
hln = 15
p1 = []
for t in range(hln, len(p)+1):
    tmp = p[t-hln:t]
    p1.append(tmp)
#len(p1)=len(p)-hln+1
p2 = [np.mean(p[t-hln:t]) for t in range(hln, len(p)+1)]
#fill historical value with zero or nan
p3 = [np.nan for t in range(0, hln-1)] + p2

# 未来ftn：每一段ftn做一次操作
# 以[t+1:t+ftn]的数据，计算t+1的值，然后存到新list的t位置
# 例如，t=[1:15]的数据，计算t=1的值，然后存到新list的t=0位置
ftn = 25
f1 = []
for t in range(1, len(p)-ftn+1):
    tmp = p[t:t+ftn]
    f1.append(tmp)
#len(f1)=len(p)-ftn+1
f2 = [np.mean(p[t:t+ftn]) for t in range(1, len(p)-ftn+1)]
f3 = f2 + [np.nan for t in range(0, ftn)]

# 历史hln与未来ftn：每段操作
# 以[0:t]的数据，预测t+1的值，然后存到新list的t位置
# 同时以[t+1:t+ftn]的数据，计算t+1的值，然后也存到新list的t位置
# [0:14]和[15:39]的数据，计算t=15的值，存到t=14的位置
hln = 15
ftn = 25
pf1 = []
for t in range(hln, len(p)-ftn+1):
    tmp = p[t-hln:t+ftn]
    pf1.append(tmp)
#len(pf1)=len(p)-hln-ftn+1
pf2 = [np.mean(p[t-hln:t+ftn]) for t in range(hln, len(p)-ftn+1)]
pf3 = [np.nan for t in range(0, hln-1)] + \
      pf2 + \
      [np.nan for t in range(0, ftn)]

# find and remove nan from list
pf4 = []
for t in range(len(pf3)):
    if ~np.isnan(pf3[t]): 
    # if np.isnan(pf3[t]) is False:
        pf4.append(pf3[t])
pf5 = [tmp for tmp in pf3 if ~np.isnan(tmp)]
