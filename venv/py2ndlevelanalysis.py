# set of programs for 2nd-level analysis of BOLD responses and SEM results, in
# relation to participant characteristics

# 1) need to indicate type of data to use:  a) BOLD responses, b) 1- and 2-source SEM, c) network SEM
# 2) need to indicate which personal characteristics to use, such as painratings, temperatures, age, etc. from DB
# 3) need to indicate type of test; significance or correlation
# 4) need to indicate statistical thresholds
# 5) need to save/display results

# need DBname and DBnum as inputs
# need database entry names for personal characteristics

import numpy as np
from scipy import stats

# 2-source SEM results
# save the results somehow
results = {'type': '2source', 'CCrecord': CCrecord, 'beta2': beta2, 'beta1': beta1, 'Zgrid2': Zgrid2,
           'Zgrid1_1': Zgrid1_1, 'Zgrid1_2': Zgrid1_2, 'DBname': self.DBname, 'DBnum': self.DBnum,
           'cluster_properties': cluster_properties}

resultsrecordname = os.path.join(self.SEMresultsdir, 'SEMresults_2source_record_' + self.SEMsavetag + '.npy')
np.save(resultsrecordname, results)


# network SEM results
# save the results somehow
results = {'type': 'network', 'resultsnames': outputnamelist, 'network': self.networkmodel,
           'regionname': self.SEMregionname, 'clustername': self.SEMclustername, 'DBname': self.DBname,
           'DBnum': self.DBnum}
resultsrecordname = os.path.join(self.SEMresultsdir, 'SEMresults_network_record_' + self.SEMsavetag + '.npy')
np.save(resultsrecordname, results)


# BOLD response data
region_data = np.load(self.SEMregionname, allow_pickle=True).flat[0]
region_properties = region_data['region_properties']
# dict_keys(['tc', 'tc_sem', 'nruns_per_person', 'tsize'])

cluster_data = np.load(self.SEMclustername, allow_pickle=True).flat[0]
cluster_properties = cluster_data['cluster_properties']
# dict_keys(['cx', 'cy', 'cz', 'IDX', 'nclusters', 'rname', 'regionindex', 'regionnum'])


def get_cluster_info(namelist, nclusterlist, number):
    regionend = np.cumsum(nclusterlist)
    cc = np.where(regionend > number)[0]
    regionnum = cc[0]
    if regionnum == 0:
        clusternum = number
    else:
        clusternum = (number - regionend[regionnum-1]).astype(int)
    regionname = namelist[regionnum]
    return regionname, clusternum, regionnum


def get_cluster_position_details(cluster_properties):
    nregions = len(cluster_properties)
    cluster_info = []
    rname_list = []
    ncluster_list = np.zeros(nregions)
    for ii in range(nregions):
        cx = cluster_properties[ii]['cx']
        cy = cluster_properties[ii]['cy']
        cz = cluster_properties[ii]['cz']
        IDX = cluster_properties[ii]['IDX']
        nclusters = cluster_properties[ii]['nclusters']
        ncluster_list[ii] = nclusters
        rname = cluster_properties[ii]['rname']
        rname_list.append(rname)
        regionnum = cluster_properties[ii]['regionnum']
        regionindex = cluster_properties[ii]['regionindex']
        regionlimits = [np.min(cx), np.max(cx), np.min(cy), np.max(cy), np.min(cz), np.max(cz)]
        cluster_coords = np.zeros((nclusters,3))
        for nn in range(nclusters):
            cc = np.where(IDX == nn)[0]
            x0 = np.mean(cx[cc])
            y0 = np.mean(cy[cc])
            z0 = np.mean(cz[cc])
            cluster_coords[nn,:] = x0,y0,z0
        entry = {'rname':rname, 'nclusters':nclusters, 'regionnum':regionnum, 'regionlimits':regionlimits, 'cluster_coords':cluster_coords}
        cluster_info.append(entry)
    return cluster_info, rname_list, ncluster_list


# look for significant group-average beta-value differences from zero
def group_average_significance(filename, pthreshold):
    data = np.load(filename, allow_pickle=True).flat[0]
    try:
        keylist = list(data.keys())
        if 'type' in keylist: datafiletype = 1   # SEM results
        if 'region_properties' in keylist: datafiletype = 2    # BOLD time-course data
    except:
        print('group_average_significance:  input file does not appear to have the correct data structure')
        return 0

    if datafiletype == 1:
        semtype = data['type']

        if semtype == '2source':
            # beta2 = np.zeros((nclusters, nclusters, nclusters, ntimepoints, NP, 2))
            # beta1 = np.zeros((nclusters, nclusters, nclusters, ntimepoints, NP, 2))

            cluster_properties = data['cluster_properties']
            cluster_info, rname_list, ncluster_list = get_cluster_position_details(cluster_properties)

            beta1 = data['beta1']
            beta2 = data['beta2']

            ntclusters, ns1sclusters, ns2clusters, ntimepoints, NP, nbeta = np.shape(beta1)

            mean_beta1 = np.mean(beta1,axis = 4)
            se_beta1 = np.std(beta1,axis = 4)/np.sqrt(NP)
            Tbeta1 = mean_beta1/(se_beta1 + 1.0e-10)

            mean_beta2 = np.mean(beta2,axis = 4)
            se_beta2 = np.std(beta2,axis = 4)/np.sqrt(NP)
            Tbeta2 = mean_beta2/(se_beta2 + 1.0e-10)

            Tthresh = stats.t.ppf(1-pthreshold,NP-1)

            beta1_sig = abs(Tbeta1) > Tthresh
            beta2_sig = abs(Tbeta2) > Tthresh

            # how to save/display the results?
            # check for redundant results?

            # write out significant results
            for tt in range(ntimepoints):
                results = []
                sig_temp = beta1_sig[:,:,:,tt,:]
                t,s1,s2,nb = np.where(sig_temp)    # significant connections during this time period

                Tvalue_list = np.zeros(len(t))
                for ii in range(len(t)):
                    Tvalue_list[ii] = Tbeta1[t[ii],s1[ii],s2[ii],tt,nb[ii]]
                    if nb[ii] == 0:
                        s = s1[ii]
                    else:
                        s = s2[ii]
                    # get region names, cluster numbers, etc.
                    targetname, targetcluster, targetnumber = get_cluster_info(rname_list, ncluster_list, t[ii])
                    sourcename, sourcecluster, sourcenumber = get_cluster_info(rname_list, ncluster_list, s)
                    targetcoords = cluster_info[targetnumber]['cluster_coords'][targetcluster,:]
                    targetlimits = cluster_info[targetnumber]['regionlimits']
                    sourcecoords = cluster_info[sourcenumber]['cluster_coords'][targetcluster,:]
                    sourcelimits = cluster_info[sourcenumber]['regionlimits']

                    entry = {'tname':targetname, 'tcluster':targetcluster, 'sname':sourcename, 'scluster':sourcecluster,
                             'tcoords':targetcoords, 'tlimits':targetlimits, 'scoords':sourcecoords, 'slimits':sourcelimits,
                             'Tvalue': Tvalue_list[ii]}
                    results.append(entry)
                # sort by significance
                dorder = np.argsort(np.abs(Tvalue_list))
                results = results[dorder]




        if semtype == 'network':
            # results = {'type': 'network', 'resultsnames': outputnamelist, 'network': self.networkmodel,
            #            'regionname': self.SEMregionname, 'clustername': self.SEMclustername, 'DBname': self.DBname,
            #            'DBnum': self.DBnum}
            resultsnames = data['resultsnames']
            network = data['network']
            regionname = data['regionname']

            for fname in resultsnames:
                results = np.load(fname, allow_pickle=True).flat[0]
                sem_one_target = results['sem_one_target_results']
                ntclusters = len(sem_one_target)

                for tt in range(ntclusters):
                    beta = sem_one_target[tt]['b']
                    ncombinations, ntimepoints, NP, nsources = np.shape(beta)

                    mean_beta = np.mean(beta, axis=2)
                    se_beta = np.std(beta, axis=2) / np.sqrt(NP)
                    Tbeta = mean_beta / (se_beta + 1.0e-10)

                    Tthresh = stats.t.ppf(1 - pthreshold, NP - 1)

                    beta_sig = abs(Tbeta) > Tthresh

                    # how to save/display the results?
                    # check for redundant results?


