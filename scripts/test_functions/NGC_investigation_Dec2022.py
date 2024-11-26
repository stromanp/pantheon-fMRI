# look at details of NGc results
# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pantheon\venv')
import numpy as np
import matplotlib.pyplot as plt
import pysapm
import os
import copy

latentnumber = 1   # latent input to the NGC

datadir = r'E:\SAPMresults2_Oct2022\SAPM_NGCinput_test'
nametag = '_0310013213'
resultsbase = ['RSnostim','Sens', 'Low', 'Pain','High', 'RSstim']
covnamebase = ['RSnostim','Sens', 'Low', 'Pain2','High', 'RSstim2']
nresults = len(resultsbase)
resultsnames = [resultsbase[x]+nametag+'_results.npy' for x in range(nresults)]
paramsnames = [resultsbase[x]+nametag+'_params.npy' for x in range(nresults)]
covnames = [covnamebase[x]+'_covariates.npy' for x in range(nresults)]


# load params
paramsnamefull = os.path.join(datadir,paramsnames[0])
p = np.load(paramsnamefull, allow_pickle=True).flat[0]
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
Nintrinsic = vintrinsic_count + fintrinsic_count

con_names = []
ncon = len(betanamelist)
for nn in range(ncon):
    pair = beta_list[nn]['pair']
    if pair[0] >= nregions:
        sname = 'int{}'.format(pair[0]-nregions)
    else:
        sname = rnamelist[pair[0]]
    tname = rnamelist[pair[1]]
    name = '{}-{}'.format(sname[:4],tname[:4])
    con_names += [name]

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


# resultsname1 = os.path.join(datadir,resultsnames[0])
# order = []
# order_record = []
# for nn in range(nresults):
#     resultsname2 = os.path.join(datadir,resultsnames[nn])
#     omean,ostd,olist = pysapm.compare_order_two_datasets(resultsname1, resultsname2, paramsnamefull)
#     order += [np.round(omean)]
#     order_record.append({'order':olist})
# order = np.array(order).astype(int)

# load data
studynumber = 5
resultsname = os.path.join(datadir,resultsnames[studynumber])
results = np.load(resultsname,allow_pickle=True)

paramsnamefull = os.path.join(datadir,paramsnames[studynumber])
p = np.load(paramsnamefull, allow_pickle=True).flat[0]
nruns_per_person = p['nruns_per_person']
tsize = p['tsize']

NP = len(results)
# NP,nv = np.shape(results)
# v = order[studynumber,0]

# load covariates
covnamefull = os.path.join(datadir,covnames[studynumber])
cov = np.load(covnamefull, allow_pickle=True).flat[0]
charlist = cov['GRPcharacteristicslist']
x = charlist.index('painrating')
covvals = cov['GRPcharacteristicsvalues'][x].astype(float)
G = np.concatenate((covvals[:,np.newaxis],np.ones((len(covvals),1))),axis=1)
G2 = np.concatenate((covvals[:,np.newaxis]**2,covvals[:,np.newaxis],np.ones((len(covvals),1))),axis=1)


for person in range(NP):
    # v = order_record[studynumber]['order'][person,0]
    R2total = results[person]['R2total']
    Sinput = results[person]['Sinput']
    Sconn = results[person]['Sconn']
    Mconn = results[person]['Mconn']
    Minput = results[person]['Minput']
    beta_int1 = results[person]['beta_int1']
    fintrinsic1 = results[person]['fintrinsic1']
    Meigv = results[person]['Meigv']
    betavals = results[person]['betavals']

    Mintrinsic = results[person]['Mintrinsic']
    Mintrinsic_mean = np.mean(np.reshape(Mintrinsic,(Nintrinsic,nruns_per_person[person],tsize)),axis=1)

    latent = copy.deepcopy(Mintrinsic[latentnumber,:])

    nin,tsizefull = np.shape(Sinput)
    ncon,tsizefull = np.shape(Sconn)

    fit, Mintrinsic2, Meigv2, err2 = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                             vintrinsic_count, beta_int1, fintrinsic1)

    if person == 0:
        latent_all = copy.deepcopy(latent)
        Sinput_all = copy.deepcopy(Sinput)
        Sconn_all = copy.deepcopy(Sconn)
        Mintrinsic_all = copy.deepcopy(Mintrinsic)
        Mintrinsic_mean_all = copy.deepcopy(Mintrinsic_mean[:,:,np.newaxis])
        Meigv_all = copy.deepcopy(Meigv[:,:,np.newaxis])
        betavals_all = copy.deepcopy(betavals[:,np.newaxis])
    else:
        latent_all = np.concatenate((latent_all,latent),axis = 0)
        Sinput_all = np.concatenate((Sinput_all,Sinput),axis = 1)
        Sconn_all = np.concatenate((Sconn_all,Sconn),axis = 1)
        Mintrinsic_all = np.concatenate((Mintrinsic_all,Mintrinsic),axis = 1)
        Mintrinsic_mean_all = np.concatenate((Mintrinsic_mean_all,Mintrinsic_mean[:,:,np.newaxis]),axis = 2)
        Meigv_all = np.concatenate((Meigv_all,Meigv[:,:,np.newaxis]),axis = 2)
        betavals_all = np.concatenate((betavals_all,betavals[:,np.newaxis]),axis = 1)

in_corr = np.zeros(nin)
for nn in range(nin):
    R = np.corrcoef(Sinput_all[nn,:],latent_all)
    in_corr[nn] = R[0,1]
    print('input to {}   R = {:.3f}'.format(rnamelist[nn],R[0,1]))
print('\n')

con_corr = np.zeros(ncon)
for nn in range(ncon):
    R = np.corrcoef(Sconn_all[nn,:],latent_all)
    con_corr[nn] = R[0,1]
    print('output {}   R = {:.3f}'.format(con_names[nn],R[0,1]))
print('\n')

# write out Meigv
meig = np.mean(Meigv_all,axis=2)
meig_sd = np.std(Meigv_all,axis=2)
latent_number = 0

xx = np.where(covvals >= 40.)[0]
for nn in range(ncon):
    # fit to covariates
    M = Meigv_all[nn,latent_number,:]
    b =  np.linalg.inv(G[xx,:].T @ G[xx,:]) @ G[xx,:].T @ M[xx]
    fit = G[xx,:] @ b
    err = M[xx] - fit
    errmean = np.mean(err)
    R2 = 1. - np.sum((err-errmean)**2)/np.sum((M - np.mean(M))**2 + 1.0e-20)
    if np.var(covvals) < 1e-6:  R2 = 0.0
    if np.var(M) < 1e-6:  R2 = 0.0
    print('{} {}  {:.3f} {} {:.3f}   R2 = {:.3f}'.format(nn,con_names[nn],meig[nn,latent_number],chr(177),meig_sd[nn,latent_number],R2))
print('\n')


# show one
nn = 17

M = Meigv_all[nn, latent_number, :]
b = np.linalg.inv(G[xx,:].T @ G[xx,:]) @ G[xx,:].T @ M[xx]
fit = G[xx,:] @ b
err = M[xx] - fit
errmean = np.mean(err)
R2 = 1. - np.sum((err - errmean) ** 2) / np.sum((M - np.mean(M)) ** 2 + 1.0e-20)

windownum = 20+nn
plt.close(windownum)
fig = plt.figure(windownum)
plt.plot(covvals,M,'ob')
covvals2 = covvals[xx]
x = np.argsort(covvals2)
plt.plot(covvals2[x],fit[x],'-k')


# plot Meigv
latentnum = 1
windownum = 30+latentnum
plt.close(windownum)
fig = plt.figure(windownum)
ncon,nl,NP = np.shape(Meigv_all)
for nn in range(NP):
    plt.plot(range(ncon),Meigv_all[:,latentnum,nn],linestyle='none',marker = 'o')


# check results symmetry---------------------------------------------
#--------------------------------------------------------------------
person = 3
latentnumber = 2
v = 0
R2total = results[person][v]['R2total']
Sinput = results[person][v]['Sinput']
Sconn = results[person][v]['Sconn']
Mconn = results[person][v]['Mconn']
Minput = results[person][v]['Minput']
beta_int1 = results[person][v]['beta_int1']
fintrinsic1 = results[person][v]['fintrinsic1']
Meigv = results[person][v]['Meigv']
betavals = results[person][v]['betavals']

Mintrinsic = results[person][v]['Mintrinsic']
Mintrinsic_mean = np.mean(np.reshape(Mintrinsic,(Nintrinsic,nruns_per_person[person],tsize)),axis=1)

latent = copy.deepcopy(Mintrinsic[latentnumber,:])
c = np.where(latent_flag == (latentnumber+1))[0]

betavals_working = copy.deepcopy(betavals)
betavals_working[c] *= -1.0
Mconn_working = copy.deepcopy(Mconn)
Mconn_working[ctarget,csource] = betavals_working

fit1, Mintrinsic1, Meigv1, err1 = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
                                                         beta_int1, fintrinsic1)
fit2, Mintrinsic2, Meigv2, err2 = pysapm.network_eigenvector_method(Sinput, Minput, Mconn_working, fintrinsic_count, vintrinsic_count,
                                                         beta_int1, fintrinsic1)

windownum = 45
plt.close(windownum)
fig = plt.figure(windownum)
nregions,tsizefull = np.shape(fit1)
for nn in range(nregions):
    ax = fig.add_subplot(3,4,nn+1)
    plt.plot(range(tsizefull),Sinput[nn,:],'-xr')
    plt.plot(range(tsizefull),fit1[nn,:],'-xg')
    plt.plot(range(tsizefull),fit2[nn,:],'-b')




# testing sorting methods-----------------------------------------
def sort_SAPM_results2_old(SAPMresults, vintrinsic_count, fintrinsic_count, latent_flag):
    NP,ncombos = np.shape(SAPMresults)

    for person in range(NP):
        for v in range(ncombos):
            Meigv = SAPMresults[person][v]['Meigv']
            if v == 0:
                Mset = copy.deepcopy(Meigv[:,:,np.newaxis])
            else:
                Mset = np.concatenate((Mset,Meigv[:,:,np.newaxis]),axis=2)
        if person == 0:
            Meigv_total = copy.deepcopy(Mset[:,:,:,np.newaxis])
        else:
            Meigv_total = np.concatenate((Meigv_total,Mset[:,:,:,np.newaxis]),axis=3)

    # result is ncon x nlatents x ncombos x NP
    # need to sort the combos to be the closest matches

    person0 = 0
    order_list = []
    for person1 in range(NP):
        Rgrid = np.zeros((ncombos,ncombos))
        order = np.zeros(ncombos)
        for nl1 in range(ncombos):
            for nl2 in range(ncombos):
                m1 = Meigv_total[:,:,nl1,person0]
                m2 = Meigv_total[:,:,nl2,person1]
                R = np.corrcoef(m1.flatten(),m2.flatten())[0,1]
                Rgrid[nl1,nl2] = R
            x = np.argmax(Rgrid[nl1,:])
            order[nl1] = x

        if person1 == 0:
            order_list = [order]
        else:
            order_list += [order]
    order_list = np.array(order_list).astype(int)

    # apply the sorting
    SAPMresultsr = copy.deepcopy(SAPMresults)
    for person1 in range(NP):
        for nl1 in range(ncombos):
            SAPMresultsr[person1][nl1] = copy.deepcopy(SAPMresults[person1][order_list[person1,nl1]])

    return SAPMresultsr, order_list



def sort_SAPM_results2(SAPMresults, vintrinsic_count, fintrinsic_count, latent_flag):
    NP,ncombos = np.shape(SAPMresults)

    Ncombo = 2 ** vintrinsic_count
    search_size = 2 * np.ones(vintrinsic_count)

    search_size = 2 * np.ones(vintrinsic_count)
    scalefactors = np.zeros((ncombos, vintrinsic_count))
    for nn in range(ncombos):
        scalefactor = 1.0 - 2.0 * pysapm.ind2sub_ndims(search_size, nn)
        scalefactors[nn, :] = scalefactor

    # reference
    person = 0
    relative_sign_record = np.zeros((vintrinsic_count,ncombos))
    ref_betavals = []
    for nlatent in range(vintrinsic_count):
        c = np.where(latent_flag == (nlatent + 1 + fintrinsic_count))[0]
        for v in range(ncombos):
            betavals = SAPMresults[person][v]['betavals'][c]
            if v == 0: ref_betavals.append({'refb':betavals})
            refb = SAPMresults[person][0]['betavals'][c]
            x = np.argmax([np.sum((betavals-refb)**2), np.sum((betavals+refb)**2)])
            xs = 2.*x-1.
            relative_sign_record[nlatent,v] = xs

    # compare
    person = 3
    sign_record = np.zeros((vintrinsic_count,ncombos))
    for nlatent in range(vintrinsic_count):
        c = np.where(latent_flag == (nlatent + 1 + fintrinsic_count))[0]
        for v in range(ncombos):
            betavals = SAPMresults[person][v]['betavals'][c]
            refb = ref_betavals[nlatent]['refb']
            x = np.argmax([np.sum((betavals-refb)**2), np.sum((betavals+refb)**2)])
            xs = 2.*x-1.
            sign_record[nlatent,v] = xs
    print(sign_record)




    c1 = np.where(latent_flag == 2)[0]
    c2 = np.where(latent_flag == 3)[0]

    person = 3
    windownum = 13
    plt.close(windownum)
    fig = plt.figure(windownum)
    b0 = SAPMresults[person][0]['betavals']
    b1 = SAPMresults[person][1]['betavals']
    plt.plot(range(len(b0)),b0,linestyle='none',marker = 'o')
    plt.plot(range(len(b1)),b1,linestyle='none',marker = 'o')
    plt.plot(c1,b0[c1],linestyle='none',marker = 'o',color='r')
    plt.plot(c2,b0[c2],linestyle='none',marker = 'o',color='g')
    plt.plot(c1,b1[c1],linestyle='none',marker = 'o',color='b')
    plt.plot(c2,b1[c2],linestyle='none',marker = 'o',color='k')


    c1 = np.where(latent_flag == 2)[0]
    c2 = np.where(latent_flag == 3)[0]
    person = 3
    windownum = 113
    plt.close(windownum)
    fig = plt.figure(windownum)
    for v in range(ncombos):
        b0 = SAPMresults[person][v]['betavals']
        plt.plot(range(len(b0)),b0,linestyle='none',marker = 'o')
        plt.plot(c1,b0[c1],linestyle='none',marker = 'x')
        plt.plot(c2,b0[c2],linestyle='none',marker = 'x')


    for person in range(NP):
        for nlatent in range(vintrinsic_count):
            c = np.where(latent_flag == (nlatent + 1 + fintrinsic_count))[0]

            for v in range(ncombos):
                betavals = SAPMresults[person][v]['betavals']



            if v == 0:
                Bset = copy.deepcopy(betavals[:,:,np.newaxis])
            else:
                Mset = np.concatenate((Mset,Meigv[:,:,np.newaxis]),axis=2)
        if person == 0:
            Meigv_total = copy.deepcopy(Mset[:,:,:,np.newaxis])
        else:
            Meigv_total = np.concatenate((Meigv_total,Mset[:,:,:,np.newaxis]),axis=3)

    # result is ncon x nlatents x ncombos x NP
    # need to sort the combos to be the closest matches

    person0 = 0
    order_list = []
    for person1 in range(NP):
        Rgrid = np.zeros((ncombos,ncombos))
        order = np.zeros(ncombos)
        for nl1 in range(ncombos):
            for nl2 in range(ncombos):
                m1 = Meigv_total[:,:,nl1,person0]
                m2 = Meigv_total[:,:,nl2,person1]
                R = np.corrcoef(m1.flatten(),m2.flatten())[0,1]
                Rgrid[nl1,nl2] = R
            x = np.argmax(Rgrid[nl1,:])
            order[nl1] = x

        if person1 == 0:
            order_list = np.array(order[:,np.newaxis])
        else:
            order_list = np.concatenate((order_list,order[:,np.newaxis]),axis=1)
    order_list = order_list.astype(int)

    # apply the sorting
    SAPMresultsr = copy.deepcopy(SAPMresults)
    for person1 in range(NP):
        for nl1 in range(ncombos):
            print(order_list[nl1,person1])
            SAPMresultsr[person1][nl1] = copy.deepcopy(SAPMresults[person1][order_list[nl1,person1]])

    return SAPMresultsr   #, order_list




def compare_order_two_datasets2(resultsname1, resultsname2, paramsname):
    results1 = np.load(resultsname1,allow_pickle=True)
    results2 = np.load(resultsname2,allow_pickle=True)

    SAPMparams = np.load(paramsname, allow_pickle=True).flat[0]
    fintrinsic_count = SAPMparams['fintrinsic_count']
    vintrinsic_count = SAPMparams['vintrinsic_count']
    ctarget = SAPMparams['ctarget']
    csource = SAPMparams['csource']
    latent_flag = SAPMparams['latent_flag']

    # compare the data sets -----------------------------------------
    Nintrinsics = vintrinsic_count + fintrinsic_count
    # ncombos = 2 ** Nintrinsics
    ncombos = 2 ** vintrinsic_count

    # organize the data---------------
    NP1,ncombos = np.shape(results1)
    for person in range(NP1):
        for v in range(ncombos):
            Meigv = results1[person][v]['Meigv']
            if v == 0:
                Mset = copy.deepcopy(Meigv[:,:,np.newaxis])
            else:
                Mset = np.concatenate((Mset,Meigv[:,:,np.newaxis]),axis=2)
        if person == 0:
            Meigv_total1 = copy.deepcopy(Mset[:,:,:,np.newaxis])
        else:
            Meigv_total1 = np.concatenate((Meigv_total1,Mset[:,:,:,np.newaxis]),axis=3)


    # organize the data---------------
    NP2,ncombos = np.shape(results2)
    for person in range(NP2):
        for v in range(ncombos):
            Meigv = results2[person][v]['Meigv']
            if v == 0:
                Mset = copy.deepcopy(Meigv[:,:,np.newaxis])
            else:
                Mset = np.concatenate((Mset,Meigv[:,:,np.newaxis]),axis=2)
        if person == 0:
            Meigv_total2 = copy.deepcopy(Mset[:,:,:,np.newaxis])
        else:
            Meigv_total2 = np.concatenate((Meigv_total2,Mset[:,:,:,np.newaxis]),axis=3)

    # need to sort the combos to be the closest matches

    big_order_list = []
    for person1 in range(NP1):
        order_list = []
        for person2 in range(NP2):
            Rgrid = np.zeros((ncombos,ncombos))
            order = np.zeros(ncombos)
            for nl1 in range(ncombos):
                for nl2 in range(ncombos):
                    m1 = Meigv_total1[:,:,nl1,person1]
                    m2 = Meigv_total2[:,:,nl2,person2]
                    R = np.corrcoef(m1.flatten(),m2.flatten())[0,1]
                    Rgrid[nl1,nl2] = R
                x = np.argmax(Rgrid[nl1,:])
                order[nl1] = x

            if person2 == 0:
                order_list = np.array(order[:,np.newaxis])
            else:
                order_list = np.concatenate((order_list,order[:,np.newaxis]),axis=1)
        order_list = np.array(order_list).astype(int)

        if person1 == 0:
            big_order_list = np.array(order_list[:,:, np.newaxis])
        else:
            big_order_list = np.concatenate((big_order_list, order_list[:,:, np.newaxis]), axis=2)

    orderm = [np.mean(big_order_list[x,:,:]) for x in range(ncombos)]
    ordersd = [np.std(big_order_list[x,:,:]) for x in range(ncombos)]

    final_order = np.argsort(orderm)

    # apply the sorting
    results2r = copy.deepcopy(results2)
    for person2 in range(NP):
        for nl2 in range(ncombos):
            results2r[person2][nl2] = copy.deepcopy(results2[person2][final_order[nl2]])

    return results2r, final_order