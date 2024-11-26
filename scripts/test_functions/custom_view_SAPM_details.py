# look at details of SAPM results
#


import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import HuberRegressor, LinearRegression
import pysem
import pysapm


ng = 3   # data group number
person = 15
window_number = 35


resultsdir = r'E:\SAPMresults2_Oct2022'
nametags = ['Sens', 'Low', 'RSnostim', 'Pain', 'High', 'RSstim']   # , 'Allpain'
refnames = ['nulldata_10000']
covtags = ['Sens', 'Low', 'RSnostim', 'Pain2', 'High', 'RSstim2']   # , 'allpain'
namesuffix = '_0310013210_all'

SAPMparamsnames = [os.path.join(resultsdir,nt+namesuffix+'_params.npy') for nt in nametags]
SAPMresultsnames = [os.path.join(resultsdir,nt+namesuffix+'_results.npy') for nt in nametags]
covnames = [os.path.join(resultsdir,nt+'_covariates.npy') for nt in covtags]
covtag = 'painrating'

ref_paramsnames = [os.path.join(resultsdir,nt+'_params.npy') for nt in refnames]
ref_resultsnames = [os.path.join(resultsdir,nt+'_results.npy') for nt in refnames]

SAPMparams = np.load(SAPMparamsnames[ng], allow_pickle=True).flat[0]
SAPMresults_load = np.load(SAPMresultsnames[ng], allow_pickle=True)
covariates = np.load(covnames[ng], allow_pickle=True).flat[0]

# create labels for each connection
rnamelist = SAPMparams['rnamelist']
Mconn = SAPMresults_load[0]['Mconn']
Minput = SAPMresults_load[0]['Minput']
betanamelist = SAPMparams['betanamelist']
beta_list = SAPMparams['beta_list']

nregions = len(rnamelist)
nr1, nr2 = np.shape(Mconn)
labeltext_record = []
Mconn_index_record = []
for n1 in range(nr1):
    tname = betanamelist[n1]
    tpair = beta_list[n1]['pair']
    if tpair[0] >= nregions:
        ts = 'int{}'.format(tpair[0] - nregions)
    else:
        ts = rnamelist[tpair[0]]
        if len(ts) > 4:  ts = ts[:4]
    tt = rnamelist[tpair[1]]
    if len(tt) > 4:  tt = tt[:4]
    for n2 in range(nr2):
        if np.abs(Mconn[n1,n2]) > 0:
            sname = betanamelist[n2]
            spair = beta_list[n2]['pair']
            if spair[0] >= nregions:
                ss = 'int{}'.format(spair[0] - nregions)
            else:
                ss = rnamelist[spair[0]]
                if len(ss) > 4:  ss = ss[:4]
            st = rnamelist[spair[1]]
            if len(st) > 4:  st = st[:4]

            labeltext = '{}-{}-{}'.format(ss, st, tt)
            labeltext_record += [labeltext]
            Mconn_index_record.append({'i':[n1,n2]})

# ------pick a connection to plot-----------------
connection_name = 'Hypo-LC-C6RD'
x = labeltext_record.index(connection_name)
n1, n2 = Mconn_index_record[x]['i']


# get details
# outputname = pysapm.display_SAPM_results(123, self.SRnametag, self.covariatesvalues, self.SRoptionvalue,
#                                          self.SRresultsdir, self.SRparamsname, self.SRresultsname,
#                                          self.SRgroup, self.SRtargetregion, self.SRpvalue, [], self.SRCanvas, True)

network = SAPMparams['network']
beta_list = SAPMparams['beta_list']
betanamelist = SAPMparams['betanamelist']
nruns_per_person = SAPMparams['nruns_per_person']
rnamelist = SAPMparams['rnamelist']
fintrinsic_count = SAPMparams['fintrinsic_count']
fintrinsic_region = SAPMparams['fintrinsic_region']
vintrinsic_count = SAPMparams['vintrinsic_count']
nclusterlist = SAPMparams['nclusterlist']
tplist_full = SAPMparams['tplist_full']
tcdata_centered = SAPMparams['tcdata_centered']
ctarget = SAPMparams['ctarget']
csource = SAPMparams['csource']
tsize = SAPMparams['tsize']
timepoint = SAPMparams['timepoint']
epoch = SAPMparams['epoch']
Nintrinsic = fintrinsic_count + vintrinsic_count
# end of reloading parameters-------------------------------------------------------

# load the SEM results

# for nperson in range(NP)
NP = len(SAPMresults_load)
resultscheck = np.zeros((NP, 4))
nbeta, tsize_full = np.shape(SAPMresults_load[0]['Sconn'])
ncon = nbeta - Nintrinsic
paradigm_centered = SAPMresults_load[0]['fintrinsic_base']  # model paradigm used for fixed pattern latent inputs

if epoch >= tsize:
    et1 = 0
    et2 = tsize
else:
    if np.floor(epoch / 2).astype(int) == np.ceil(epoch / 2).astype(int):  # even numbered epoch
        et1 = (timepoint - np.floor(epoch / 2)).astype(int)
        et2 = (timepoint + np.floor(epoch / 2)).astype(int)
    else:
        et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
        et2 = (timepoint + np.floor(epoch / 2)).astype(int)
if et1 < 0: et1 = 0
if et2 > tsize: et2 = tsize
epoch = et2 - et1
ftemp = paradigm_centered[et1:et2]


# get results for selected person
Sinput = SAPMresults_load[person]['Sinput']
Sconn = SAPMresults_load[person]['Sconn']
Minput = SAPMresults_load[person]['Minput']
Mconn = SAPMresults_load[person]['Mconn']
beta_int1 = SAPMresults_load[person]['beta_int1']
R2total = SAPMresults_load[person]['R2total']
Meigv = SAPMresults_load[person]['Meigv']
betavals = SAPMresults_load[person]['betavals']

nruns = nruns_per_person[person]
fintrinsic1 = np.array(list(ftemp) * nruns_per_person[person])

# ---------------------------------------------------
fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                         vintrinsic_count, beta_int1, fintrinsic1)
nr, tsize_total = np.shape(Sinput)
tsize = (tsize_total / nruns).astype(int)
nbeta, tsize2 = np.shape(Sconn)

tc = Sinput
tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
Sinput_total = tc1

tc = Sconn
tc1 = np.mean(np.reshape(tc, (nbeta, nruns, tsize)), axis=1)
Sconn_total = tc1

tc = fit
tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
fit_total = tc1

fit_check = Minput@Sconn_total

Mrecord = Mconn

# display input to target region------------------------------------------------------------
target = 'C6RD'
# target = 'LC'

rtarget = rnamelist.index(target)
m = Minput[rtarget, :]
sources = np.where(m == 1)[0]
rsources = [beta_list[ss]['pair'][0] for ss in sources]
nsources = len(sources)
nregions = len(rnamelist)
checkdims = np.shape(Sinput_total)
if np.ndim(Sinput_total) > 2:  nv = checkdims[2]
tsize = checkdims[1]

# get beta values from Mconn
textlist = []
for ss in sources:
    m = Mconn[:, ss]
    targets2ndlevel_list = np.where(m != 0.)[0]
    text = betanamelist[ss] + ': '
    beta = Mconn[targets2ndlevel_list, ss]
    for ss2 in range(len(beta)):
        valtext = '{:.2f} '.format(beta[ss2])
        text1 = '{}{}'.format(valtext, betanamelist[targets2ndlevel_list[ss2]])
        text += text1 + ', '
    textlist += [text[:-1]]

window = window_number
fig1 = plt.figure(window)  # for plotting in GUI, expect "window" to refer to a figure
plt.close(window)
fig1, axs = plt.subplots(nsources, 2, sharey=True, figsize=(12, 9), dpi=100, num=window)

x = list(range(tsize))
xx = x + x[::-1]
tc1 = Sinput_total[rtarget, :]
tc1f = fit_total[rtarget, :]

axs[0, 1].plot(x, tc1, '-ob', linewidth=1, markersize=4)
axs[0, 1].plot(x, tc1f, '-xr', linewidth=1, markersize=4)
axs[0, 1].set_title('target input {}'.format(rnamelist[rtarget]))

tc1 = fit_check[rtarget, :]
axs[1, 1].plot(x, tc1, '-ok')
axs[1, 1].set_title('total output')

for ss in range(nsources):
    tc1 = Sconn_total[sources[ss], :]
    axs[ss, 0].plot(x, tc1, '-xr')
    if rsources[ss] >= nregions:
        axs[ss, 0].set_title('source output {} {}'.format(betanamelist[sources[ss]], 'int'))
    else:
        axs[ss, 0].set_title('source output {} {}'.format(betanamelist[sources[ss]], rnamelist[rsources[ss]]))
    axs[ss, 0].annotate(textlist[ss], xy=(.025, .025), xycoords='axes  fraction',
                        horizontalalignment='left', verticalalignment='bottom', fontsize=6)





# display details of one input to target region------------------------------------------------------------
source = 'LC'
# source = 'C6RD'
rtarget = rnamelist.index(target)
rsource = rnamelist.index(source)
connection_name = '{}_{}'.format(rsource,rtarget)
rcon = betanamelist.index(connection_name)

fit_output = Mconn@Sconn_total

m = Minput[rsource, :]   # values for inputs to the "source"
sources2 = np.where(m == 1)[0]
rsources2 = [beta_list[ss]['pair'][0] for ss in sources2]
nsources2 = len(sources2)

# get beta values from Mconn
textlist = []
for ss in sources2:
    m = Mconn[:, ss]
    targets2ndlevel_list = np.where(m != 0.)[0]
    text = betanamelist[ss] + ': '
    beta = Mconn[targets2ndlevel_list, ss]
    for ss2 in range(len(beta)):
        valtext = '{:.2f} '.format(beta[ss2])
        text1 = '{}{}'.format(valtext, betanamelist[targets2ndlevel_list[ss2]])
        text += text1 + ', '
    textlist += [text[:-1]]

window = window_number+1
fig2 = plt.figure(window)  # for plotting in GUI, expect "window" to refer to a figure
plt.close(window)
fig2, axs2 = plt.subplots(nsources2, 2, sharey=True, figsize=(12, 9), dpi=100, num=window)

x = list(range(tsize))
tc1 = Sinput_total[rsource, :]
tc1f = fit_total[rsource, :]
tc1f2 = fit_check[rsource, :]

axs2[0, 1].plot(x, tc1, '-ob', linewidth=1, markersize=4)
axs2[0, 1].plot(x, tc1f, '-xr', linewidth=1, markersize=4)
# axs2[0, 1].plot(x, tc1f2, '-xk', linewidth=1, markersize=4)
axs2[0, 1].set_title('target input {}'.format(rnamelist[rsource]))

axs2[1, 1].plot(x, fit_output[rcon, :], '-ok')
axs2[1, 1].set_title('source output')

for ss in range(nsources2):
    tc1 = Sconn_total[sources2[ss], :]
    axs2[ss, 0].plot(x, tc1, '-xr')
    if rsources2[ss] >= nregions:
        axs2[ss, 0].set_title('source output {} {}'.format(betanamelist[sources2[ss]], 'int'))
    else:
        axs2[ss, 0].set_title('source output {} {}'.format(betanamelist[sources2[ss]], rnamelist[rsources2[ss]]))
    axs2[ss, 0].annotate(textlist[ss], xy=(.025, .025), xycoords='axes  fraction',
                        horizontalalignment='left', verticalalignment='bottom', fontsize=6)
