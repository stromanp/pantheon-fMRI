import numpy as np
import matplotlib.pyplot as plt


DB = [[0, 1, 0, 1, 0],[-0.5, 0, -1, 0, 1], [0, 1.0, 0, 0, 0], [0,0,0,1,0],[0,0,0,0,1]]
Min = [[0, 1, 0, 1, 0],[1, 0, 1, 0, 1], [0, 1, 0, 0, 0], [0,0,0,1,0],[0,0,0,0,1]]
e,v = np.linalg.eig(DB)

L1a = np.real(v[:,3])
L2a = np.real(v[:,4])

L1s = 0.5*L1/L1[3]
L2s = -0.5*L2/L2[4]
Lt = L1s + L2s

L1s = 0.1*L1/L1[3]
L2s = 0.5*L2/L2[4]
Lt = L1s + L2s