# look at details of NGc results
# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pantheon\venv')
import numpy as np
import matplotlib.pyplot as plt
import pysapm
import os
import copy

studynumber = 5

datadir = r'E:\SAPMresults2_Oct2022\SAPM_NGCinput_test'
nametag = '_0310013213'
resultsbase = ['RSnostim','Sens', 'Low', 'Pain','High', 'RSstim']
covnamebase = ['RSnostim','Sens', 'Low', 'Pain2','High', 'RSstim2']
nresults = len(resultsbase)
resultsnames = [resultsbase[x]+nametag+'_results.npy' for x in range(nresults)]
paramsnames = [resultsbase[x]+nametag+'_params.npy' for x in range(nresults)]
covnames = [covnamebase[x]+'_covariates.npy' for x in range(nresults)]


# load params
paramsnamefull = os.path.join(datadir,paramsnames[studynumber])
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
nruns_per_person = p['nruns_per_person']
tsize = p['tsize']

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
connection_names = np.array(connection_names)
#------------------------------------------------------

# load data
resultsname = os.path.join(datadir,resultsnames[studynumber])
results = np.load(resultsname,allow_pickle=True)
NP = len(results)

# load covariates
covnamefull = os.path.join(datadir,covnames[studynumber])
cov = np.load(covnamefull, allow_pickle=True).flat[0]
charlist = cov['GRPcharacteristicslist']
x = charlist.index('painrating')
covvals = cov['GRPcharacteristicsvalues'][x].astype(float)
G = np.concatenate((covvals[:,np.newaxis],np.ones((len(covvals),1))),axis=1)


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

    nin,tsizefull = np.shape(Sinput)
    ncon,tsizefull = np.shape(Sconn)

    if person == 0:
        Sinput_all = copy.deepcopy(Sinput)
        Sconn_all = copy.deepcopy(Sconn)
        Mintrinsic_all = copy.deepcopy(Mintrinsic)
        Mintrinsic_mean_all = copy.deepcopy(Mintrinsic_mean[:,:,np.newaxis])
        Meigv_all = copy.deepcopy(Meigv[:,:,np.newaxis])
        betavals_all = copy.deepcopy(betavals[:,np.newaxis])
    else:
        Sinput_all = np.concatenate((Sinput_all,Sinput),axis = 1)
        Sconn_all = np.concatenate((Sconn_all,Sconn),axis = 1)
        Mintrinsic_all = np.concatenate((Mintrinsic_all,Mintrinsic),axis = 1)
        Mintrinsic_mean_all = np.concatenate((Mintrinsic_mean_all,Mintrinsic_mean[:,:,np.newaxis]),axis = 2)
        Meigv_all = np.concatenate((Meigv_all,Meigv[:,:,np.newaxis]),axis = 2)
        betavals_all = np.concatenate((betavals_all,betavals[:,np.newaxis]),axis = 1)


beta_mean = np.mean(betavals_all,axis = 1)
beta_sd = np.std(betavals_all,axis = 1)
T = beta_mean/(beta_sd + 1.0e-20)
ncon = len(beta_mean)

corrR = np.zeros(ncon)
for nn in range(ncon):
    R = np.corrcoef(betavals_all[nn,:],covvals)
    corrR[nn] = R[0,1]

x = np.argsort(np.abs(T))
xT = np.array([np.where(x == s)[0] for s in range(len(x))]).flatten()
x = np.argsort(np.abs(corrR))
xR = np.array([np.where(x == s)[0] for s in range(len(x))]).flatten()
value_score = np.sqrt(xT**2 + xR**2)

x = np.argsort(value_score)
for nn in range(7):
    print('{}  b = {:.3f} {} {:.3f}  T = {:.3f}   R = {:.3f}'.format(connection_names[x[nn]],beta_mean[x[nn]],chr(177),beta_sd[x[nn]],T[x[nn]], corrR[x[nn]]))

# worst in list:
nn=0
print('{}  b = {:.3f} {} {:.3f}  T = {:.3f}   R = {:.3f}'.format(connection_names[x[nn]],beta_mean[x[nn]],chr(177),beta_sd[x[nn]],T[x[nn]], corrR[x[nn]]))

# find related connections and check them
cname = connection_names[x[nn]]
target = ctarget[x[nn]]
source = csource[x[nn]]
target_pair = beta_list[target]['pair']
source_pair = beta_list[source]['pair']

if source_pair[0] >= nregions:
    sname = 'int{}'.format(source_pair[0] - nregions)
else:
    sname = rnamelist[source_pair[0]]
mname = rnamelist[target_pair[0]]
tname = rnamelist[target_pair[1]]
name = '{}-{}-{}'.format(sname[:4], mname[:4], tname[:4])
name_part1 = '{}-{}'.format(sname[:4], mname[:4])
name_part2 = '{}-{}'.format(mname[:4], tname[:4])

# find other connections with the same source pair or target pair
source_match = np.zeros(len(ctarget))
target_match = np.zeros(len(ctarget))
for nn in range(len(ctarget)):
    source_match[nn] = source_pair == beta_list[csource[nn]]['pair']
    target_match[nn] = target_pair == beta_list[ctarget[nn]]['pair']

c = np.where(source_match)[0]
print('connections with the same source:  ')
for cc in c:
    print('{}  b = {:.3f} {} {:.3f}  T = {:.3f}   R = {:.3f}'.format(connection_names[cc], beta_mean[cc], chr(177),
                                                                   beta_sd[cc], T[cc], corrR[cc]))

c = np.where(target_match)[0]
print('connections with the same target:  ')
for cc in c:
    print('{}  b = {:.3f} {} {:.3f}  T = {:.3f}   R = {:.3f}'.format(connection_names[cc], beta_mean[cc], chr(177),
                                                                   beta_sd[cc], T[cc], corrR[cc]))
