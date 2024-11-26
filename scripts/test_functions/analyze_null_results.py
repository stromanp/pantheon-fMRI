import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import copy
import os
import pandas as pd

SAPMresultsname = r'E:\SAPMresults_Dec2022\null_results_L1_corr.npy'
SAPMparamsname = r'E:\SAPMresults_Dec2022\null_params_L1.npy'

statsrefname = r'E:\SAPMresults_Dec2022\network_model_June2023_SAPM_bstats_L1_1000samples.xlsx'

results = np.load(SAPMresultsname, allow_pickle=True)
params = np.load(SAPMparamsname, allow_pickle=True).flat[0]

latent_flag = params['latent_flag']
reciprocal_flag = params['reciprocal_flag']
ctarget = params['ctarget']
csource = params['csource']

latent_flag2 = copy.deepcopy(latent_flag)
latent_flag2[latent_flag2 > 0] = 1
regular_flag = 1-(latent_flag2 + reciprocal_flag)   # flag where connections are not latent or reciprocal

cr = np.where(regular_flag > 0)[0]
nbeta_selected = len(cr)
nbeta = len(results[0]['betavals'])

nsamples = len(results)
sample_list = list(range(nsamples))

# need to determine the probability of getting B values that are signficantly different than zero, across a group
# pick random groups

# load reference stats from null simulations
nr1,nr2 = np.shape(results[0]['Mconn'])
DBref_mean = np.zeros((nr1,nr2))
DBref_std = np.zeros((nr1,nr2))
rnamelist = copy.deepcopy(params['rnamelist'])
print('SAPMstatsfile = {}'.format(statsrefname))
if os.path.isfile(statsrefname):
	xls = pd.ExcelFile(statsrefname, engine='openpyxl')
	df1 = pd.read_excel(xls, 'B stats')
	stats_conname = df1.loc[:, 'name']
	stats_mean = df1.loc[:, 'mean']
	stats_std = df1.loc[:, 'std']

	for nn in range(len(stats_conname)):
		nregions = len(rnamelist)
		cname = stats_conname[nn]
		c = cname.index('-')
		sname = cname[:c]
		tname = cname[(c + 1):]
		tnum = rnamelist.index(tname)
		if 'latent' in sname:
			lnum = int(sname[6:])
			snum = nregions + lnum
		else:
			snum = rnamelist.index(sname)
		DBref_mean[tnum, snum] = stats_mean[nn]
		DBref_std[tnum, snum] = stats_std[nn]

	DBrefvals_mean = copy.deepcopy(DBref_mean[ctarget,csource])
	DBrefvals_std = copy.deepcopy(DBref_std[ctarget,csource])

NP = 20  # the group size
nrep = 5000  # the number of data sets to simulate

Tsample = np.zeros((nbeta,nrep))

for aa in range(nrep):
	# out of all the samples, pick NP results at random
	sample = np.random.choice(sample_list, NP, replace = False)
	beta_sample = np.zeros((NP,nbeta))
	for nn, ss in enumerate(sample):
		# print('{}  sample {}'.format(nn,ss))
		beta_sample[nn,:] = results[ss]['betavals']
	mb = np.mean(beta_sample, axis=0)
	ms = np.std(beta_sample, axis=0)/np.sqrt(NP)
	T = (mb-DBrefvals_mean)/(ms + 1.0e-20)
	Tsample[:,aa] = copy.deepcopy(T)

# now look at resulting distributions of B values, ignoring latents and reciprocals
Tmean = np.mean(Tsample, axis = 1)
Tsd = np.std(Tsample, axis = 1)
Tskew = stats.skew(Tsample, axis = 1)
Tkurt = stats.kurtosis(Tsample, axis = 1)

plt.close(71)
fig = plt.figure(71)
for nn in cr:
	count, bins = np.histogram(Tsample[nn,:],bins = 25)
	bb = (bins[1:] + bins[:-1]) / 2.0
	plt.plot(bb,count,'-', color = [0.7,0.7,0.7])


tdist = np.linspace(-6,6,200)
normdist = 1600. * stats.t.pdf(tdist,NP-1)
plt.plot(tdist,normdist,'-', color = [0,0,0], linewidth = 2)

SAPMresultsname = r'E:\SAPMresults_Dec2022\null_results_L1_corr.npy'
p,f = os.path.split(SAPMresultsname)
f,ext = os.path.splitext(f)
svgname = os.path.join(p,f + '_Tdist.svg')
plt.savefig(svgname, format='svg')