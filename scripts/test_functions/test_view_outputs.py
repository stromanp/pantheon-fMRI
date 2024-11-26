#
# checking on results
import numpy as np
import matplotlib.pyplot as plt
import pysapm
import pydatabase

# load lists
DBname = r'E:\graded_pain_database_May2022.xlsx'
# list_list = [r'E:\RS1_nostim_list.npy',
#              r'E:\RS1_control_list.npy',
#              r'E:\High_list.npy',
#              r'E:\Low_list.npy',
#              r'E:\Pain_list.npy',
#              r'E:\Sens_list.npy']
# name_list = ['RS1_nostim','RS1_control','High','Low','Pain','Sens']
# color_list = [[0,0,0],[1,0,0],[1,0,0],[0,0.5,0],[1,0,0],[0,0,0.5]]


list_list = [r'E:\RS1_nostim_list.npy',
             r'E:\Sens_list.npy',
             r'E:\Low_list.npy',
             r'E:\RS1_control_list.npy',
             r'E:\High_list.npy',
             r'E:\Pain_list.npy']
name_list = ['RS1_nostim','Sens','Low','RS1_control','High','Pain']
color_list = [[0,0,0],[0,0.5,0],[0,0,0.5],[0.3,0,0],[0.7,0,0],[1,0,0]]


prefix = 'xptc'
complete_list = list(range(557))
complete_filename_list, complete_dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, complete_list, prefix, mode='list')

group_indices = []
for listnumber in range(len(list_list)):
    listrecord = np.load(list_list[listnumber], allow_pickle=True).flat[0]
    dbnumlist = listrecord['dbnumlist']
    filename_list, dbnum_person_list, NP1 = pydatabase.get_datanames_by_person(DBname, dbnumlist, prefix, mode='list')

    g = [complete_dbnum_person_list.index(dbnumlist) for dbnumlist in dbnum_person_list]
    group_indices.append({'g':g})


SAPMparametersname = r'E:\\allconditionV6_params.npy'
SAPMresultsname = r'E:\\allconditionV6_results.npy'

# load SAPM parameters
SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
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
SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)

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

Mrecord = np.zeros((nbeta, nbeta, NP))
R2totalrecord = np.zeros(NP)
for nperson in range(NP):
    Sinput = SAPMresults_load[nperson]['Sinput']
    Sconn = SAPMresults_load[nperson]['Sconn']
    Minput = SAPMresults_load[nperson]['Minput']
    Mconn = SAPMresults_load[nperson]['Mconn']
    beta_int1 = SAPMresults_load[nperson]['beta_int1']
    R2total = SAPMresults_load[nperson]['R2total']
    Meigv = SAPMresults_load[nperson]['Meigv']
    betavals = SAPMresults_load[nperson]['betavals']

    nruns = nruns_per_person[nperson]
    fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])

    # ---------------------------------------------------
    fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                             vintrinsic_count, beta_int1, fintrinsic1)
    nr, tsize_total = np.shape(Sinput)
    tsize = (tsize_total / nruns).astype(int)
    nbeta, tsize2 = np.shape(Sconn)

    if nperson == 0:
        Sinput_total = np.zeros((nr, tsize, NP))
        Sconn_total = np.zeros((nbeta, tsize, NP))
        fit_total = np.zeros((nr, tsize, NP))

    tc = Sinput
    tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
    Sinput_total[:, :, nperson] = tc1

    tc = Sconn
    tc1 = np.mean(np.reshape(tc, (nbeta, nruns, tsize)), axis=1)
    Sconn_total[:, :, nperson] = tc1

    tc = fit
    tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
    fit_total[:, :, nperson] = tc1

    Mrecord[:, :, nperson] = Mconn
    R2totalrecord[nperson] = R2total


# show latent2 for each group
index = 30
plt.close(10)
fig = plt.figure(10)
g = []
# for listnumber in range(3):
#     g += group_indices[listnumber]['g']

g = group_indices[0]['g']
Sconn_group = Sconn_total[index,:,:]
Sconn_group = Sconn_group[:,g]
Sconn_avg = np.mean(Sconn_group,axis=1)
Sconn_sem = np.std(Sconn_group,axis=1)/np.sqrt(len(g))
plt.plot(list(range(tsize)),Sconn_avg, color = [0,0,0],linewidth = 2)
plt.plot(list(range(tsize)),Sconn_avg+Sconn_sem, color = [0.5,0.5,0.5],linewidth = 1)
plt.plot(list(range(tsize)),Sconn_avg-Sconn_sem, color = [0.5,0.5,0.5],linewidth = 1)

plt.close(11)
fig = plt.figure(11)
g = []
for listnumber in range(1,3):
    g += group_indices[listnumber]['g']

Sconn_group = Sconn_total[index,:,:]
Sconn_group = Sconn_group[:,g]
Sconn_avg = np.mean(Sconn_group,axis=1)
Sconn_sem = np.std(Sconn_group,axis=1)/np.sqrt(len(g))
plt.plot(list(range(tsize)),Sconn_avg, color = [0,1,0],linewidth = 2)
plt.plot(list(range(tsize)),Sconn_avg+Sconn_sem, color = [0,0.5,0],linewidth = 1)
plt.plot(list(range(tsize)),Sconn_avg-Sconn_sem, color = [0,0.5,0],linewidth = 1)

plt.close(12)
fig = plt.figure(12)
g = []
for listnumber in range(3,6):
    g += group_indices[listnumber]['g']

Sconn_group = Sconn_total[index,:,:]
Sconn_group = Sconn_group[:,g]
Sconn_avg = np.mean(Sconn_group,axis=1)
Sconn_sem = np.std(Sconn_group,axis=1)/np.sqrt(len(g))
plt.plot(list(range(tsize)),Sconn_avg, color = [1,0,0],linewidth = 2)
plt.plot(list(range(tsize)),Sconn_avg+Sconn_sem, color = [0.5,0,0],linewidth = 1)
plt.plot(list(range(tsize)),Sconn_avg-Sconn_sem, color = [0.5,0,0],linewidth = 1)


