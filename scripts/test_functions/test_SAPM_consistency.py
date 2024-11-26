# testing
import numpy as np
import load_templates
import os
import pandas as pd
import pysapm
import pysem
import time
import copy
import matplotlib.pyplot as plt

SAPMresultsdir = r'E:\SAPMresults2_Oct2022'
vintrinsic_count = 2
ncombos = 2**vintrinsic_count

rname1 = os.path.join(SAPMresultsdir, r'Pain_0310013210_all_results1.npy')
rname2 = os.path.join(SAPMresultsdir, r'Pain_0310013210_all_results2.npy')
rname3 = os.path.join(SAPMresultsdir, r'Pain_0310013210_all_results3.npy')

results1 = np.load(rname1,allow_pickle =True)
results2 = np.load(rname2,allow_pickle =True)
results3 = np.load(rname3,allow_pickle =True)

# network parameters
SAPMparamsname = os.path.join(SAPMresultsdir, r'Pain_0310013210_all_params.npy')
p = np.load(SAPMparamsname, allow_pickle=True).flat[0]
betanamelist = p['betanamelist']
beta_list = p['beta_list']
ctarget = p['ctarget']
csource = p['csource']
nregions = p['nregions']
rnamelist = p['rnamelist']
fintrinsic_count = p['fintrinsic_count']
vintrinsic_count = p['vintrinsic_count']
latent_flag = p['latent_flag']
reciprocal_flag = p['reciprocal_flag']

connection_names = []
for nn in range(len(csource)):
    source_pair = beta_list[csource[nn]]['pair']
    target_pair = beta_list[ctarget[nn]]['pair']
    if source_pair[0] >= nregions:
        sname = 'int{}'.format(source_pair[0]-nregions)
    else:
        sname = rnamelist[source_pair[0]]
    mname = rnamelist[target_pair[0]]
    tname = rnamelist[target_pair[1]]
    name = '{}-{}-{}'.format(sname[:4],mname[:4],tname[:4])
    connection_names += [name]
#------------------------------------------------------

person = 0

b1 = np.array([results1[person][x]['betavals'] for x in range(ncombos)])
b2 = np.array([results2[person][x]['betavals'] for x in range(ncombos)])
b3 = np.array([results3[person][x]['betavals'] for x in range(ncombos)])

R21 = np.array([results1[person][x]['R2total'] for x in range(ncombos)])
R22 = np.array([results2[person][x]['R2total'] for x in range(ncombos)])
R23 = np.array([results3[person][x]['R2total'] for x in range(ncombos)])

cgrid1 = np.array([ [np.corrcoef(b1[x,:],b2[y,:])[0,1] for x in range(ncombos)] for y in range(ncombos)])
cgrid2 = np.array([ [np.corrcoef(b1[x,:],b3[y,:])[0,1] for x in range(ncombos)] for y in range(ncombos)])
cgrid3 = np.array([ [np.corrcoef(b2[x,:],b3[y,:])[0,1] for x in range(ncombos)] for y in range(ncombos)])

M1 = np.array([results1[person][x]['Mintrinsic'] for x in range(ncombos)])
M2 = np.array([results2[person][x]['Mintrinsic'] for x in range(ncombos)])
M3 = np.array([results3[person][x]['Mintrinsic'] for x in range(ncombos)])

Mgrid0 = np.array([ [np.corrcoef(M1[x,0,:],M2[y,0,:])[0,1] for x in range(ncombos)] for y in range(ncombos)])
Mgrid1 = np.array([ [np.corrcoef(M1[x,1,:],M2[y,1,:])[0,1] for x in range(ncombos)] for y in range(ncombos)])

print('Mgrid0:\n  {}'.format(Mgrid0))
print('Mgrid1:\n  {}'.format(Mgrid1))

print('cgrid1:\n  {}'.format(cgrid1))
print('cgrid2:\n  {}'.format(cgrid2))
print('cgrid3:\n  {}'.format(cgrid3))


#-----group
NP = len(results1)
b1group = np.array([results1[np][0]['betavals'] for np in range(NP)])
b2group = np.array([results2[np][0]['betavals'] for np in range(NP)])
b3group = np.array([results3[np][0]['betavals'] for np in range(NP)])

R2group1 = np.array([results1[np][0]['R2total'] for np in range(NP)])

b1group_mean = np.mean(b1group,axis=0)
b1group_std = np.std(b1group,axis=0)
b2group_mean = np.mean(b2group,axis=0)
b2group_std = np.std(b2group,axis=0)
b3group_mean = np.mean(b3group,axis=0)
b3group_std = np.std(b3group,axis=0)
ncon = len(b1group_mean)

c = np.where(latent_flag)[0]
cnumvals = np.array(list(range(ncon)))
windownum = 51
plt.close(windownum)
fig = plt.figure(windownum)
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.errorbar(cnumvals,b1group_mean, b1group_std,linestyle = 'none', color = 'r', marker = 'o')
ax1.errorbar(cnumvals[c],b1group_mean[c], b1group_std[c],linestyle = 'none', color = 'k', marker = 'o')
ax2.errorbar(cnumvals,b2group_mean, b2group_std,linestyle = 'none', color = 'g', marker = 'o')
ax2.errorbar(cnumvals[c],b2group_mean[c], b2group_std[c],linestyle = 'none', color = 'k', marker = 'o')
ax3.errorbar(cnumvals,b3group_mean, b3group_std,linestyle = 'none', color = 'b', marker = 'o')
ax3.errorbar(cnumvals[c],b3group_mean[c], b3group_std[c],linestyle = 'none', color = 'k', marker = 'o')

r1 = 0
r2 = 0
r3 = 0
windownum = 21
plt.close(windownum)
fig = plt.figure(windownum)
plt.plot(b1[r1,:],b2[r2,:],'or')

plt.close(windownum+1)
fig = plt.figure(windownum+1)
plt.plot(b1[r1,:],b3[r3,:],'og')

plt.close(windownum+2)
fig = plt.figure(windownum+2)
plt.plot(b2[r2,:],b3[r3,:],'ob')

flag = np.zeros(len(b1[r1,:]))
mag = np.sqrt(b1[r1,:]**2 + b3[r3,:]**2)
for nn in range(len(b1[r1,:])):
    flag[nn] = b1[r1,nn]*b3[r3,nn] < 0

x = np.argsort(mag)
x = x[::-1]
for nn in range(len(b1[r1,:])):
    if flag[x[nn]]:
        print('{}  b1 = {:.3f}   b3 = {:.3f}'.format(connection_names[x[nn]],b1[r1,x[nn]],b3[r3,x[nn]]))


cgrid1 = np.array([[np.corrcoef(b0[x,c1],b1[y,c1])[0,1] for x in range(ncombos)] for y in range(ncombos)])
cgrid2 = np.array([[np.corrcoef(b0[x,c2],b1[y,c2])[0,1] for x in range(ncombos)] for y in range(ncombos)])
print('cgrid1 = \n{}'.format(cgrid1))
print('cgrid2 = \n{}'.format(cgrid2))

# figure out which groups are most consistent across people

# order the combos to match the order in the first person
results1r = copy.deepcopy(results1)
clist = []
for nn in range(vintrinsic_count):
    cc = np.where(latent_flag==(fintrinsic_count+1+nn))[0]
    clist.append({'c':cc})

# c1 = np.where(latent_flag==(fintrinsic_count+1))[0]
# c2 = np.where(latent_flag==(fintrinsic_count+2))[0]

# figure out how to reorder the 2nd run to match the 1st run
search_size = 2*np.ones(vintrinsic_count)
scalefactors = np.zeros((ncombos,vintrinsic_count))
for nn in range(ncombos):
    scalefactor = 1.0 - 2.0*pysapm.ind2sub_ndims(search_size, nn)
    scalefactors[nn,:] = scalefactor

person = 0
b0 = np.array([results1[person][x]['betavals'] for x in range(ncombos)])
for personindex in range(1,NP):
    b1 = np.array([results1[personindex][x]['betavals'] for x in range(ncombos)])

    intsign = np.zeros(vintrinsic_count)
    for nn in range(vintrinsic_count):
        c = clist[nn]['c']
        cc = np.corrcoef(b0[0,c],b1[0,c])[0,1]
        intsign[nn] = np.sign(cc)

    # cc1 = np.corrcoef(b0[0,c1],b1[0,c1])[0,1]
    # cc2 = np.corrcoef(b0[0,c2],b1[0,c2])[0,1]
    # intsign = np.array([np.sign(cc1), np.sign(cc2)])

    actualfactors = copy.deepcopy(scalefactors)
    for nn in range(ncombos):
        actualfactors[nn,:] *= intsign

    order = np.zeros(ncombos).astype(int)
    for nn in range(ncombos):
        ref = actualfactors[nn,:]
        check = np.zeros(ncombos)
        for nn2 in range(ncombos):
            check[nn2] = (ref == scalefactors[nn2,:]).all()
        x = np.where(check)[0]
        order[nn] = x

    for nn in range(ncombos):
        results1r[personindex][order[nn]] = copy.deepcopy(results1[personindex][nn])


# show the group average results now

optionnumber = 0
b1group = np.array([results1r[np][optionnumber]['betavals'] for np in range(NP)])
R2group1 = np.array([results1r[np][optionnumber]['R2total'] for np in range(NP)])

print('R2 range {:.3f} to {:.3f},  \naverage {:.3f} {} {:.3f}'.format(np.min(R2group1),np.max(R2group1),np.mean(R2group1),chr(177),np.std(R2group1)))

b1group_mean = np.mean(b1group,axis=0)
b1group_std = np.std(b1group,axis=0)
ncon = len(b1group_mean)

c = np.where(latent_flag)[0]
cnumvals = np.array(list(range(ncon)))
windownum = 100+optionnumber
plt.close(windownum)
fig = plt.figure(windownum)
plt.errorbar(cnumvals,b1group_mean, b1group_std,linestyle = 'none', color = 'r', marker = 'o')
plt.errorbar(cnumvals[c],b1group_mean[c], b1group_std[c],linestyle = 'none', color = 'k', marker = 'o')



# check effects of varying b values
X = x[0]   # pick the connection with the biggest inconsistency
r1 = 0
r3 = 0

betavals = copy.deepcopy(b3[r3,:])

R2total = results3[person][r3]['R2total']
Sinput = results3[person][r3]['Sinput']
Sconn = results3[person][r3]['Sconn']
Mconn = results3[person][r3]['Mconn']
Minput = results3[person][r3]['Minput']
beta_int1 = results3[person][r3]['beta_int1']
fintrinsic1 = results3[person][r3]['fintrinsic1']

db = np.array([-0.01,0.0,0.01])
Lweight = 1.0e-1
dval = 0.01
alpha = 1.0e-3

betavals2 = copy.deepcopy(betavals)
Mconn[ctarget, csource] = copy.deepcopy(betavals2)
dssq_db, ssqd, dssq_dbeta1 = pysapm.gradients_for_betavals(Sinput, Minput, Mconn, betavals2, ctarget, csource, dval,
                                                    fintrinsic_count, vintrinsic_count, beta_int1,
                                                    fintrinsic1, Lweight)
# apply the changes
# limit the betaval changes
dsmax = 0.1 / alpha
dssq_db[dssq_db < -dsmax] = -dsmax
dssq_db[dssq_db > dsmax] = dsmax

betavals2 -= alpha * dssq_db


for nn in range(len(db)):
    betavals2 = copy.deepcopy(betavals)
    betavals2[X] += db[nn]
    Mconn[ctarget,csource] = copy.deepcopy(betavals2)
    fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput,Mconn, fintrinsic_count,
                                                                           vintrinsic_count, beta_int1,fintrinsic1)
    print('{} err = {:.2f}'.format(db[nn],err))