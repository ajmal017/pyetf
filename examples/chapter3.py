# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 18:27:45 2019

@author: w
"""
# 第三章
list1 = [3,6,1,8,10]
list2 = []
ln = len(list1)
for t in range(0, ln):
    list2.append(max(list1))
    list1.remove(max(list1))
print(list2)
i = 7
for t in range(0, ln):
    if i>list2[t]:
        list2.insert(t,i)
        break
print(list2)