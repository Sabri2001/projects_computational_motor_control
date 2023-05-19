import numpy as np

def fun():
	return 3,4


a = np.linalg.norm(np.array(fun()))
print('a: ', a)
