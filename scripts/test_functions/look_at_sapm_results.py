import matplotlib
import matplotlib.pyplot as plt
import numpy as np


resultsname1 = r'E:/FM2021data/HCstim_2230224124_V5b_results_corr.npy'
paramsname1 = r'E:/FM2021data/HCstim_2230224124_V5b_params.npy'

resultsname2 = r'E:/FM2021data/HCrest_2230224124_V5b_results_corr.npy'
paramsname2 = r'E:/FM2021data/HCrest_2230224124_V5b_params.npy'

connection = 'LC-Thalamus'
windownum = 30

results1 = np.load(resultsname1, allow_pickle=True)
params1 = np.load(paramsname1, allow_pickle=True).flat[0]

results2 = np.load(resultsname2, allow_pickle=True)
params2 = np.load(paramsname2, allow_pickle=True).flat[0]

betanamelist = params1['betanamelist']
beta_list = params1['beta_list']
rnamelist = params1['rnamelist']
nregions = len(rnamelist)
ncon = len(beta_list)
nr1,nr2 = np.shape(results1[0]['Mconn'])
connamelist = []
for nn in range(ncon):
	pair = beta_list[nn]['pair']
	snum, tnum = pair[0],pair[1]
	tname = rnamelist[tnum]
	if snum >= nregions:
		sname = 'int{}'.format(snum-nregions)
	else:
		sname = rnamelist[snum]
	connamelist += ['{}-{}'.format(sname,tname)]

NP1 = len(results1)
NP2 = len(results2)
betavals1 = np.zeros((ncon,NP1))
Mconn1 = np.zeros((nr1,nr2,NP1))
for nn in range(NP1):
	betavals1[:,nn] = results1[nn]['betavals']
	Mconn1[:,:,nn] = results1[nn]['Mconn']

betavals2 = np.zeros((ncon,NP2))
Mconn2 = np.zeros((nr1,nr2,NP2))
for nn in range(NP2):
	betavals2[:, nn] = results2[nn]['betavals']
	Mconn2[:,:,nn] = results2[nn]['Mconn']


# look at a particular connection
index = connamelist.index(connection)
pair = beta_list[index]['pair']
snum,tnum = pair[0],pair[1]

# bvals1 = betavals1[index,:]
# bvals2 = betavals2[index,:]
# mb1, sb1 = np.mean(bvals1), np.std(bvals1)/np.sqrt(NP1)
# t1 = mb1/sb1
# mb2, sb2 = np.mean(bvals2), np.std(bvals2)/np.sqrt(NP2)
# t2 = mb2/sb2
# print('group1 DB = {:.2f}{}{:.2f}  T = {:.1f}   group2 DB = {:.2f}{}{:.2f}  T = {:.1f} '.format(mb1,chr(177),sb1,t1, mb2,chr(177),sb2,t2))


DBvals1 = Mconn1[tnum,snum,:]
DBvals2 = Mconn2[tnum,snum,:]
mdb1, sdb1 = np.mean(DBvals1), np.std(DBvals1)/np.sqrt(NP1)
t1d = mdb1/sdb1
mdb2, sdb2 = np.mean(DBvals2), np.std(DBvals2)/np.sqrt(NP2)
t2d = mdb2/sdb2
print('group1 DB = {:.2f}{}{:.2f}  T = {:.1f}   group2 DB = {:.2f}{}{:.2f}  T = {:.1f} '.format(mdb1,chr(177),sdb1,t1d, mdb2,chr(177),sdb2,t2d))


plt.close(windownum)
plt.figure(windownum)
bvals1 = Mconn1[tnum,snum,:]
bvals2 = Mconn2[tnum,snum,:]
for nn in range(NP1):
	plt.plot([1,2],[bvals1[nn],bvals2[nn]],'-xr')