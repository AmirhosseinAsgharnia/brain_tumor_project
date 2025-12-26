import numpy as np

A = [1,2,3,-5,4,-10,2,1]

A = sorted(set(i for i in A if i>=1))
print(A)