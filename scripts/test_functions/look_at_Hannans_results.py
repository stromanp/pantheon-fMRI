
import numpy as np
import matplotlib.pyplot as plt
import copy


resultsname1 = r'Y:\Hannan\neut_SAPM_0342423013_results_May2023_corr.npy'
resultsname2 = r'Y:\Hannan\neg_SAPM_0342423013_results_May2023_corr.npy'

results1 = np.load(resultsname1, allow_pickle=True)
results2 = np.load(resultsname2, allow_pickle=True)

neut_beta = np.zeros((19,45))
neg_beta = np.zeros((19,45))

for nn in range(19): neut_beta[nn,:] = results1[nn]['betavals']
for nn in range(19): neg_beta[nn,:] = results2[nn]['betavals']


paramsname = r'Y:\\Hannan\\neut_SAPM_0342423013_params_May2023.npy'
params = np.load(paramsname, allow_pickle=True).flat[0]


beta_list = params['beta_list']
rnamelist = params['rnamelist']
rnamelist_full = copy.deepcopy(rnamelist)
for nn in range(3): rnamelist_full += ['latent{}'.format(nn)]

connlist = []
for mm in range(45): print('{}  {}'.format(mm,connlist[mm]))


connum = 24
winnum = 50

conn_list = [17, 24, 36]

for dd in range(len(conn_list)):
	connum = conn_list[dd]
	winnum = 50 + dd

	plt.close(winnum)
	fig = plt.figure(winnum)
	markerlist = ['x','o','s','d','^']
	for nn in range(0,19):
		aa = nn % 5
		plt.plot([1,2],[neut_beta[nn,connum],neg_beta[nn,connum]],linestyle = '-',marker = markerlist[aa], color = cols[nn,:])
