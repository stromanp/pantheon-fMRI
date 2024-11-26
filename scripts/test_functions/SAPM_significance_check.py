# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pantheon\venv')
import numpy as np
import matplotlib.pyplot as plt
import pysapm
import os
import copy
import pydatabase
import random
import scipy.stats as stats
import pysem

datadir = r'E:\SAPMresults_Dec2022'
basename = 'null_test_3242423012'
clusterdataname = r'E:/SAPMresults_Dec2022\Pain_equalsize_cluster_def.npy'

covnames = 'allpain_covariates.npy'
covnamefull = os.path.join(datadir, covnames)
cov = np.load(covnamefull, allow_pickle=True).flat[0]
charlist = cov['GRPcharacteristicslist']
x = charlist.index('painrating')
covvals = cov['GRPcharacteristicsvalues'][x].astype(float)

NP = 10
nrep = 1
if NP < 55:
        covvals = covvals[:NP]
        cov2 = copy.deepcopy(cov)
        cov2['GRPcharacteristicsvalues'] = cov2['GRPcharacteristicsvalues'][:,:NP]
        newcovname = os.path.join(datadir, 'shortlist_covariates.npy')
        np.save(newcovname, cov2)

if NP > 55:
        nrep = (np.floor(NP/55) + 1)
        covvals = np.repeat(covvals,nrep)
        covvals = covvals[:NP]
        cov2 = copy.deepcopy(cov)
        cov2['GRPcharacteristicsvalues'] = np.repeat(cov2['GRPcharacteristicsvalues'],nrep,axis=1)[:,:NP]
        newcovname = os.path.join(datadir, 'shortlist_covariates.npy')
        np.save(newcovname, cov2)

# NP = len(covvals)
print('using null data for {} participants'.format(NP))

null_regiondataname = os.path.join(datadir, 'nulldata_High_regiondata2.npy')
null_regiondata = np.load(null_regiondataname, allow_pickle=True).flat[0]
null_region_properties = null_regiondata['region_properties']
nruns_per_person_total = null_region_properties[0]['nruns_per_person']
tsize = null_region_properties[0]['tsize']
NPtotal = len(null_region_properties[0]['nruns_per_person'])
NPlist = list(range(NPtotal))
nregions = len(null_region_properties)

resultsname = os.path.join(datadir, basename + '_results.npy')
paramsname = os.path.join(datadir, basename + '_params.npy')
sample_regiondataname = os.path.join(datadir, 'nullsample_regiondata.npy')

# networkfile = r'E:\SAPMresults_Dec2022\network_model_Jan2023_SAPM_V3.xlsx'
networkfile = r'E:\SAPMresults_Dec2022\network_model_Jan2023_SAPM.xlsx'
# networkfile = r'E:\SAPMresults_Dec2022\network_model_Jan2023_SAPM_balanced2.xlsx'

DBname = r'E:\graded_pain_database_May2022.xlsx'
timepoint = 'all'
epoch = 'all'
betascale=0.1

cnums = [3, 2, 4, 2, 4, 2, 3, 0, 1, 2]

# network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = pysapm.load_network_model_w_intrinsics(networkfile)
# ncluster_list = np.array([nclusterlist[x]['nclusters'] for x in range(len(nclusterlist))])
# full_rnum_base = [np.sum(ncluster_list[:x]) for x in range(len(ncluster_list))]
# clusterlist = np.array(cnums) + full_rnum_base

# load params

# for nn in range(nresults):
trial_results = []
ntrials = 20
for nn in range(ntrials):
        print('\n\n-------------------------------------------------------------')
        print('-----------trial {} of {}----------------------------------'.format(nn+1,ntrials))
        print('-------------------------------------------------------------\n\n')

        # pick NP data sets at random
        Nlist = random.sample(NPlist,NP)

        sample_region_properties = copy.deepcopy(null_region_properties)
        nruns_per_person = nruns_per_person_total[Nlist]
        nruns_total = np.sum(nruns_per_person)
        for rr in range(nregions):
                nclusters,tsizefull = np.shape(null_region_properties[rr]['tc'])
                sample_region_properties[rr]['tc'] = np.zeros((nclusters,nruns_total*tsize))
                sample_region_properties[rr]['tc_sem'] = np.zeros((nclusters,nruns_total*tsize))
                if nrep > 1:
                        sample_region_properties[rr]['DBnum'] = np.repeat(null_region_properties[rr]['DBnum'],nrep)[:NP]
                        sample_region_properties[rr]['nruns_per_person'] = np.repeat(nruns_per_person,nrep)[:NP]
                else:
                        sample_region_properties[rr]['DBnum'] = null_region_properties[rr]['DBnum'][:NP]
                        sample_region_properties[rr]['nruns_per_person'] = copy.deepcopy(nruns_per_person)

        for ss, ssfull in enumerate(Nlist):
                r1f = sum(nruns_per_person_total[:ssfull])
                r2f = sum(nruns_per_person_total[:(ssfull + 1)])

                r1 = sum(nruns_per_person[:ss])
                r2 = sum(nruns_per_person[:(ss + 1)])

                tpf = list(range(r1f*tsize, r2f*tsize))
                tp = list(range(r1*tsize, r2*tsize))

                for rr in range(nregions):
                        tc = copy.deepcopy(sample_region_properties[rr]['tc'])
                        tcsem = copy.deepcopy(sample_region_properties[rr]['tc_sem'])

                        tcfull = copy.deepcopy(null_region_properties[rr]['tc'])
                        tcsemfull = copy.deepcopy(null_region_properties[rr]['tc_sem'])

                        tc[:,tp] = copy.deepcopy(tcfull[:,tpf])
                        tcsem[:,tp] = copy.deepcopy(tcsemfull[:,tpf])

                        sample_region_properties[rr]['tc'] = copy.deepcopy(tc)
                        sample_region_properties[rr]['tc_sem'] = copy.deepcopy(tcsem)

        # save modified region properties
        region_data = copy.deepcopy(null_regiondata)
        region_data['region_properties'] = copy.deepcopy(sample_region_properties)
        region_data['DBnum'] = copy.deepcopy(sample_region_properties[0]['DBnum'])
        np.save(sample_regiondataname, region_data)

        resultsname = os.path.join(datadir, basename + '_results.npy')
        paramsname = os.path.join(datadir, basename + '_params.npy')

        tagname = '{}_trial{}'.format(basename,nn)
        sampleresultsname = os.path.join(datadir, tagname + '_results.npy')
        sampleparametersname = os.path.join(datadir, tagname + '_params.npy')
        # take data from null_regiondataname and extract a sample of data for each run
        pysapm.SAPMrun(cnums, sample_regiondataname, clusterdataname, sampleresultsname, sampleparametersname, networkfile, DBname, timepoint,
                epoch, betascale=betascale, reload_existing=False)

        # load the results and look for connections with B values that are:
        # 1) signfiicantly different than zero, 2) signficantly correlated with covariates
        SAPMresults_load = np.load(sampleresultsname, allow_pickle=True)
        params = np.load(sampleparametersname, allow_pickle=True).flat[0]
        beta_list = params['beta_list']
        ctarget = params['ctarget']
        csource = params['csource']
        rnamelist = params['rnamelist']
        network = params['network']

        # outputname = pysapm.display_SAPM_results(123, self.SRnametag, self.covariatesvalues, self.SRoptionvalue,
        #                     self.SRresultsdir, self.SRparamsname, self.SRresultsname, self.SRvariant,
        #                     self.SRgroup, self.SRtargetregion, self.SRpvalue, [], self.SRCanvas, True, multiple_output = multiple_output)

        nbeta,nbeta = np.shape(SAPMresults_load[0]['Mconn'])
        Nlatent = nbeta-nregions
        rnamelist_ext = copy.deepcopy(rnamelist)
        for ll in range(Nlatent):
                rnamelist_ext += ['int{}'.format(ll)]

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
                Mrecord[:, :, nperson] = Mconn
                R2totalrecord[nperson] = R2total

        # -------------------------------------------------------------------------------
        # -------------prep for regression with continuous covariate------------------------------
        p = covvals[np.newaxis, :]
        p -= np.mean(p)
        G = np.concatenate((np.ones((1, NP)), p), axis=0)  # put the intercept term first

        # -------------------------------------------------------------------------------------
        # significance of average Mconn values -----------------------------------------------
        # outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_SourceModel', 'DrawSAPMdiagram']
        Mconn_avg = np.mean(Mrecord, axis=2)
        Mconn_sd = np.std(Mrecord, axis=2)
        Mconn_sem = Mconn_sd / np.sqrt(NP)
        Tvals = Mconn_avg / (Mconn_sem + 1.0e-20)
        tol = 1e-10
        a,b = np.where(np.abs(Mconn_sem) < tol)
        Tvals[a,b] = 0.0

        pthresh = 0.05/(len(csource)-Nlatent)
        Tthresh = stats.t.ppf(1 - pthresh, NP - 1)

        # -------------------------------------------------------------------------------
        # -------------B-value regression with continuous covariate------------------------------
        Mregression = np.zeros((nbeta, nbeta, 3))
        for aa in range(nbeta):
                for bb in range(nbeta):
                        m = Mrecord[aa, bb, :]
                        if np.var(m) > 0:
                                b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis, :], G)
                                Mregression[aa, bb, :] = [b[0, 0], b[0, 1], R2]

        Rvals = Mregression[:,:,2]

        Zthresh = stats.norm.ppf(1 - pthresh)
        Rthresh = np.tanh(Zthresh / np.sqrt(NP - 1))
        R2thresh = Rthresh ** 2

        # betanamelist = [beta_list[a]['name'] for a in range(len(beta_list))]
        # for cc in range(len(network)):
        #         target = network[cc]['targetnum']
        #         sources = network[cc]['sourcenums']
        #         for mm in range(len(sources)):
        #                 source = sources[mm]
        #                 betaname = '{}_{}'.format(source, target)
        #                 x = betanamelist.index(betaname)
        #                 Minput[target, x] = 1

        trial_results.append({'Mconn_avg':Mconn_avg, 'Mconn_sem':Mconn_sem, 'Mconn_sd':Mconn_sd, 'Tvals':Tvals, 'Tthresh':Tthresh,
                              'Rvals':Rvals, 'R2thresh':R2thresh, 'beta_list':beta_list, 'ctarget':ctarget,
                              'csource':csource, 'rnamelist':rnamelist, 'nregions':nregions, 'rnamelist_ext':rnamelist_ext,
                              'Mrecord':Mrecord})


outputname = os.path.join(datadir, 'null_data_sapm_trials.npy')
np.save(outputname, trial_results)

view_results = True
if view_results:
        Mset = []
        Msdset = []
        Msemset = []
        plt.close(30)
        fig = plt.figure(30)
        ax = fig.add_subplot()
        outputname = os.path.join(datadir, 'null_data_sapm_trials.npy')
        trial_results = np.load(outputname, allow_pickle = True)

        for nn in range(ntrials):
                Mrecord = trial_results[nn]['Mrecord']
                Mconn_avg = trial_results[nn]['Mconn_avg']
                Mconn_sem = trial_results[nn]['Mconn_sem']
                Tvals = trial_results[nn]['Tvals']
                Tthresh = trial_results[nn]['Tthresh']
                Rvals = trial_results[nn]['Rvals']
                R2thresh = trial_results[nn]['R2thresh']
                beta_list = trial_results[nn]['beta_list']
                ctarget = trial_results[nn]['ctarget']
                csource = trial_results[nn]['csource']
                csource = trial_results[nn]['csource']
                rnamelist = trial_results[nn]['rnamelist']
                nregions = trial_results[nn]['nregions']
                rnamelist_ext = trial_results[nn]['rnamelist_ext']

                conn_name_list = ['{}-{}'.format(rnamelist_ext[csource[x]][:4],rnamelist_ext[ctarget[x]][:4]) for x in range(len(ctarget))]
                M = [Mconn_avg[ctarget[x],csource[x]] for x in range(len(ctarget))]
                Mset += M
                Msd = [Mconn_sd[ctarget[x],csource[x]] for x in range(len(ctarget))]
                Msdset += Msd
                Msem = [Mconn_sem[ctarget[x],csource[x]] for x in range(len(ctarget))]
                Msemset += Msem
                plt.errorbar(conn_name_list,M,yerr=Msd,marker='o', linestyle='none',markeredgecolor ='none')


                # for x in range(len(ctarget)):
                #         Mr = Mrecord[ctarget[x],csource[x],:]
                #         plt.plot([conn_name_list[x]]*len(Mr),Mr,'ob')

                # plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=7))
        ax.set_xticklabels(labels=conn_name_list, rotation='vertical', fontsize=7)

        plt.close(31)
        fig = plt.figure(31)
        ax = fig.add_subplot()
        bdata = []
        for nn in range(ntrials):
                Mrecord = trial_results[nn]['Mrecord']
                b = np.zeros((len(ctarget),NP))
                for x in range(len(ctarget)):
                        b[x,:] = Mrecord[ctarget[x], csource[x], :]
                if nn == 0:
                        bdata = copy.deepcopy(b)
                else:
                        bdata = np.concatenate((bdata,b),axis=1)


        plt.plot(conn_name_list,np.mean(bdata,axis=1),marker='o',linestyle='None',markeredgecolor ='none')
        plt.plot(conn_name_list,np.zeros(len(ctarget)),marker='None', linestyle='-',color ='k')
        plt.violinplot(bdata.T, list(range(len(ctarget))), showmeans=True, showmedians=True)
        ax.set_xticklabels(labels=conn_name_list, rotation='vertical', fontsize=7)

        Tset = np.array(Mset) / (np.array(Msemset) + 1.0e-20)
        Rset = []
        plt.close(32)
        fig = plt.figure(32)
        ax = fig.add_subplot()
        outputname = os.path.join(datadir, 'null_data_sapm_trials.npy')
        trial_results = np.load(outputname, allow_pickle = True)

        for nn in range(ntrials):
                Mconn_avg = trial_results[nn]['Mconn_avg']
                Mconn_sem = trial_results[nn]['Mconn_sem']
                Tvals = trial_results[nn]['Tvals']
                Tthresh = trial_results[nn]['Tthresh']
                Rvals = trial_results[nn]['Rvals']
                R2thresh = trial_results[nn]['R2thresh']
                beta_list = trial_results[nn]['beta_list']
                ctarget = trial_results[nn]['ctarget']
                csource = trial_results[nn]['csource']
                csource = trial_results[nn]['csource']
                rnamelist = trial_results[nn]['rnamelist']
                nregions = trial_results[nn]['nregions']
                rnamelist_ext = trial_results[nn]['rnamelist_ext']

                conn_name_list = ['{}-{}'.format(rnamelist_ext[csource[x]][:4],rnamelist_ext[ctarget[x]][:4]) for x in range(len(ctarget))]
                R = [Rvals[ctarget[x],csource[x]] for x in range(len(ctarget))]
                Rset += R
                plt.plot(conn_name_list,R,marker='o', linestyle='none',markeredgecolor ='none')
                # plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=7)

        ax.set_xticklabels(labels=conn_name_list, rotation='vertical', fontsize=7)

        windownum = 51
        plt.close(windownum)
        fig = plt.figure(windownum)
        Tset[np.abs(Tset) > 10] = 10.0
        plt.hist(Tset, bins=50)

        plt.close(windownum+1)
        fig = plt.figure(windownum+1)
        plt.hist(Rset, bins=50)


# check the check of the check
run_extra_check = False
if run_extra_check:
        nn = 1
        nperson = 1

        tagname = '{}_trial{}'.format(basename,nn)
        sampleresultsname = os.path.join(datadir, tagname + '_results.npy')
        sampleparametersname = os.path.join(datadir, tagname + '_params.npy')

        SAPMresults_load = np.load(sampleresultsname, allow_pickle=True)
        params = np.load(sampleparametersname, allow_pickle=True).flat[0]

        beta_list = params['beta_list']
        ctarget = params['ctarget']
        csource = params['csource']
        rnamelist = params['rnamelist']
        network = params['network']

        Sinput = SAPMresults_load[nperson]['Sinput']
        Sconn = SAPMresults_load[nperson]['Sconn']
        Minput = SAPMresults_load[nperson]['Minput']
        Mconn = SAPMresults_load[nperson]['Mconn']
        beta_int1 = SAPMresults_load[nperson]['beta_int1']
        R2total = SAPMresults_load[nperson]['R2total']
        Meigv = SAPMresults_load[nperson]['Meigv']
        betavals = SAPMresults_load[nperson]['betavals']
        Mintrinsic = SAPMresults_load[nperson]['Mintrinsic']
        Nintrinsic,tsize = np.shape(Mintrinsic)

        Sinput2 = Minput @ Sconn
        Sconn2 = Mconn @ Sconn

        Mconn2 = Mconn @ Mconn
        Mconn3 = Mconn2 @ Mconn
        Mconn4 = Mconn3 @ Mconn
        Mconn5 = Mconn4 @ Mconn

        Sconn3 = Mconn @ Mconn @ Sconn
        Sconn5 = Mconn5 @ Sconn

        np.max(np.abs(Sinput2-Sinput))
        np.max(np.abs(Sconn2-Sconn))

        e,v = np.linalg.eig(Mconn)
        Meig_full = v[:, -Nintrinsic:]