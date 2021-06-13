import os
import random
t_SBD=[]
v_SBD=[]
t=[]
v=[]
with open("train_SBD.txt", 'r') as f:
    for line in f.readlines():
        t_SBD.append(line.strip())
with open("val_SBD.txt", 'r') as f:
    for line in f.readlines():
        v_SBD.append(line.strip())
with open("train.txt", 'r') as f:
    for line in f.readlines():
        t.append(line.strip())
with open("val.txt", 'r') as f:
    for line in f.readlines():
        v.append(line.strip())

print(len(t_SBD))
print(len(v_SBD))
print(len(t))
print(len(v))

t_SBD_e = sorted(list(set(t_SBD)-set(t)-set(v)))
v_SBD_e = sorted(list(set(v_SBD)-set(t)-set(v)))
f_SBD_e = sorted(t_SBD_e + v_SBD_e)
print(len(f_SBD_e))
with open('train_exclude.txt', 'w') as f:
    for line in f_SBD_e:
        f.writelines(line+'\n')