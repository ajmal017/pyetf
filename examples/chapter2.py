# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 09:22:00 2019

@author: w
"""

first_number = input("第一个数：")
operator = input("运算符：")
second_number = input("第二个数：")

if ("." in first_number) or ("." in second_number):
    n1 = float(first_number)
    n2=float(second_number)
else:
    n1 = int(first_number)
    n2=int(second_number)

if operator == "+":
    r = n1+n2    

r = eval(first_number+operator+second_number)
  
print(f"运算结果是：{r}")