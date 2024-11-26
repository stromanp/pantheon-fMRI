# function to estimate a matrix inverse using gradient descent
import numpy as np
import copy

def sortof_inverse(A, order = 'A invA'):
	n1,n2 = np.shape(A)
	I = np.eye(n1)

	dval = 0.01
	itermax = 100000
	alpha = 1.0e-2
	alphamin = 1.0e-8
	Lweight = 1.0e-4

	b = (1.0/np.mean(A))*np.random.randn(n1,n2)
	b = np.zeros((n1,n2))
	bestb = copy.deepcopy(b)
	iter = 0
	cost = costfunction(b,A,Lweight, order)
	while iter < itermax and alpha > alphamin:
		iter += 1
		bgrad = calc_bgrad(b,A,dval,Lweight, order)
		b -= alpha*bgrad
		newcost = costfunction(b,A,Lweight, order)
		if newcost < cost:
			if iter % 1000 == 0:
				print('est. matrix inverse:  iter {} alpha {:.3e}  cost {:.3f}'.format(iter,alpha,newcost))
			bestb = copy.deepcopy(b)
			cost = copy.deepcopy(newcost)
		else:
			b = copy.deepcopy(bestb)
			alpha *= 0.5
			print('est. matrix inverse:  iter {} alpha {:.3e}  cost {:.3f}'.format(iter,alpha,cost))

	print('finished .... cost = {:.3f}'.format(cost))
	return b

def calc_bgrad(b,A,dval,Lweight, order):
	cost = costfunction(b,A,Lweight, order)
	bgrad = np.zeros(np.shape(b))
	n1,n2 = np.shape(b)
	for a1 in range(n1):
		for a2 in range(n2):
			bcheck = copy.deepcopy(b)
			bcheck[a1,a2] += dval
			cost2 = costfunction(bcheck,A,Lweight, order)
			bgrad[a1,a2] = (cost2-cost)/dval
	return bgrad

def costfunction(b,A,Lweight, order):
	n1,n2 = np.shape(b)
	E = np.eye(n1)
	if order[:3] == 'inv':
		check = b@A
	else:
		check = A@b
	cost = np.sum((check-E)**2) + Lweight*np.sum(np.abs(b))
	return cost

