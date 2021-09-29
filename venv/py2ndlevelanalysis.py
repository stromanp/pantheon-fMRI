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
import pyclustering
import pysem
import copy
import pydisplay
import os

# 2-source SEM results
# save the results somehow
# results = {'type': '2source', 'CCrecord': CCrecord, 'beta2': beta2, 'beta1': beta1, 'Zgrid2': Zgrid2,
#            'Zgrid1_1': Zgrid1_1, 'Zgrid1_2': Zgrid1_2, 'DBname': self.DBname, 'DBnum': self.DBnum,
#            'cluster_properties': cluster_properties}
#
# resultsrecordname = os.path.join(self.SEMresultsdir, 'SEMresults_2source_record_' + self.SEMsavetag + '.npy')
# np.save(resultsrecordname, results)
#
#
# # network SEM results
# # save the results somehow
# results = {'type': 'network', 'resultsnames': outputnamelist, 'network': self.networkmodel,
#            'regionname': self.SEMregionname, 'clustername': self.SEMclustername, 'DBname': self.DBname,
#            'DBnum': self.DBnum}
# resultsrecordname = os.path.join(self.SEMresultsdir, 'SEMresults_network_record_' + self.SEMsavetag + '.npy')
# np.save(resultsrecordname, results)
#
#
# # BOLD response data
# region_data = np.load(self.SEMregionname, allow_pickle=True).flat[0]
# region_properties = region_data['region_properties']
# # dict_keys(['tc', 'tc_sem', 'nruns_per_person', 'tsize'])
#
# cluster_data = np.load(self.SEMclustername, allow_pickle=True).flat[0]
# cluster_properties = cluster_data['cluster_properties']
# # dict_keys(['cx', 'cy', 'cz', 'IDX', 'nclusters', 'rname', 'regionindex', 'regionnum'])


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


def remove_reps_and_sort(id_list, value_list, data):
    # eliminate redundant values, for repeats keep the one with the largest value
    uid, indices = np.unique(id_list, return_index=True)
    keep_indices = []
    for cc in uid:
        indices2 = np.where(id_list == cc)[0]
        vindex = np.argmax(np.abs(value_list[indices2]))
        keep_indices.append(indices2[vindex])
    data2 = []
    value_list2 = []
    for cc in keep_indices:
        data2.append(data[cc])
        value_list2.append(value_list[cc])
    data = copy.deepcopy(data2)
    value_list = value_list2
    data2 = []
    value_list2 = []

    # sort by significance
    dorder = np.argsort(np.abs(value_list))
    data2 = copy.deepcopy(data)
    value_list2 = np.zeros(len(dorder))
    for ii, dd in enumerate(dorder):
        data2[ii] = copy.deepcopy(data[dd])
        value_list2[ii] = copy.deepcopy(value_list[dd])

    return data2, value_list2


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
            keys = ['tname', 'tcluster', 'sname', 'scluster', 'Tvalue', 'tx', 'ty', 'tz', 'tlimx1', 'tlimx2', 'tlimy1',
                    'tlimy2', 'tlimz1', 'tlimz2', 'sx', 'sy', 'sz', 'slimx1', 'slimx2', 'slimy1', 'slimy2', 'slimz1', 'slimz2']

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

            # write out significant results, based on beta1------------------------------
            for tt in range(ntimepoints):
                results = []
                sig_temp = beta1_sig[:,:,:,tt,:]
                t,s1,s2,nb = np.where(sig_temp)    # significant connections during this time period

                Tvalue_list = np.zeros(len(t))
                connid_list = np.zeros(len(t))   # identify connections - to be able to remove redundant ones later
                for ii in range(len(t)):
                    Tvalue_list[ii] = Tbeta1[t[ii],s1[ii],s2[ii],tt,nb[ii]]
                    if nb[ii] == 0:
                        s = s1[ii]
                    else:
                        s = s2[ii]
                    # get region names, cluster numbers, etc.
                    connid_list[ii] = t[ii]*1000 + s   # a unique identifier for the connection
                    targetname, targetcluster, targetnumber = get_cluster_info(rname_list, ncluster_list, t[ii])
                    sourcename, sourcecluster, sourcenumber = get_cluster_info(rname_list, ncluster_list, s)
                    targetcoords = cluster_info[targetnumber]['cluster_coords'][targetcluster,:]
                    targetlimits = cluster_info[targetnumber]['regionlimits']
                    sourcecoords = cluster_info[sourcenumber]['cluster_coords'][sourcecluster,:]
                    sourcelimits = cluster_info[sourcenumber]['regionlimits']

                    values = np.concatenate(([targetname, targetcluster, sourcename, sourcecluster, Tvalue_list[ii]],
                                             list(targetcoords),list(targetlimits), list(sourcecoords),list(sourcelimits)))
                    entry = dict(zip(keys, values))

                    #
                    #
                    # entry = {'tname':targetname, 'tcluster':targetcluster, 'sname':sourcename, 'scluster':sourcecluster,
                    #          'Tvalue': Tvalue_list[ii],'tx':targetcoords[0],'ty':targetcoords[1],'tz':targetcoords[2],
                    #          'tlimx1':targetlimits[0], 'tlimx2':targetlimits[1], 'tlimy1':targetlimits[2], 'tlimy2':targetlimits[3], 'tlimz1':targetlimits[4], 'tlimz2':targetlimits[5],
                    #          'sx':sourcecoords[0], 'sy':sourcecoords[1], 'sz':sourcecoords[2],
                    #          'slimx1':sourcelimits[0], 'slimx2':sourcelimits[1], 'slimy1':sourcelimits[2], 'slimy2':sourcelimits[3], 'slimz1':sourcelimits[4], 'slimz2':sourcelimits[5]}
                    results.append(entry)

                # eliminate redundant values, for repeats keep the one with the largest Tvalue
                results2, Tvalue_list2 = remove_reps_and_sort(connid_list, Tvalue_list, results)

                # # eliminate redundant values, for repeats keep the one with the largest Tvalue
                # uid, indices = np.unique(connid_list, return_index = True)
                # keep_indices = []
                # for cc in uid:
                #     indices2 = np.where(connid_list == cc)[0]
                #     Tindex = np.argmax(np.abs(Tvalue_list[indices2]))
                #     keep_indices.append(indices2[Tindex])
                # results2 = []
                # Tvalue_list2 = []
                # for cc in keep_indices:
                #     results2.append(results[cc])
                #     Tvalue_list2.append(Tvalue_list[cc])
                # results = copy.deepcopy(results2)
                # Tvalue_list = Tvalue_list2
                # results2 = []
                # Tvalue_list2 = []
                #
                # # sort by significance
                # dorder = np.argsort(np.abs(Tvalue_list))
                # results2 = copy.deepcopy(results)
                # for ii,dd in enumerate(dorder):
                #     results2[ii] = copy.deepcopy(results[dd])

                p,f = os.path.split(filename)
                f2,e = os.path.splitext(f)
                excelfilename = os.path.join(p,f2+'_2ndlevel.xlsx')
                excelsheetname = '2source beta1 ' + str(tt)
                pydisplay.pywriteexcel(results2, excelfilename, excelsheetname, 'append')

            results_beta1 = results2

            # now, write out significant results, based on beta2-------------------------
            for tt in range(ntimepoints):
                results = []
                sig_temp = beta2_sig[:,:,:,tt,:]
                t,s1,s2,nb = np.where(sig_temp)    # significant connections during this time period

                Tvalue_list = np.zeros(len(t))
                connid_list = np.zeros(len(t))   # identify connections - to be able to remove redundant ones later
                for ii in range(len(t)):
                    Tvalue_list[ii] = Tbeta2[t[ii],s1[ii],s2[ii],tt,nb[ii]]
                    if nb[ii] == 0:
                        s = s1[ii]
                    else:
                        s = s2[ii]
                    # get region names, cluster numbers, etc.
                    connid_list[ii] = t[ii]*1000 + s   # a unique identifier for the connection
                    targetname, targetcluster, targetnumber = get_cluster_info(rname_list, ncluster_list, t[ii])
                    sourcename, sourcecluster, sourcenumber = get_cluster_info(rname_list, ncluster_list, s)
                    targetcoords = cluster_info[targetnumber]['cluster_coords'][targetcluster,:]
                    targetlimits = cluster_info[targetnumber]['regionlimits']
                    sourcecoords = cluster_info[sourcenumber]['cluster_coords'][sourcecluster,:]
                    sourcelimits = cluster_info[sourcenumber]['regionlimits']

                    values = np.concatenate(([targetname, targetcluster, sourcename, sourcecluster, Tvalue_list[ii]],
                                             list(targetcoords),list(targetlimits), list(sourcecoords),list(sourcelimits)))
                    entry = dict(zip(keys, values))

                    results.append(entry)

                # eliminate redundant values, for repeats keep the one with the largest Tvalue
                results2, Tvalue_list2 = remove_reps_and_sort(connid_list, Tvalue_list, results)

                p,f = os.path.split(filename)
                f2,e = os.path.splitext(f)
                excelfilename = os.path.join(p,f2+'_2ndlevel.xlsx')
                excelsheetname = '2source beta2 ' + str(tt)
                pydisplay.pywriteexcel(results2, excelfilename, excelsheetname, 'append')

            results_beta2 = results2

            return results_beta1, results_beta2


        if semtype == 'network':
            # results = {'type': 'network', 'resultsnames': outputnamelist, 'network': self.networkmodel,
            #            'regionname': self.SEMregionname, 'clustername': self.SEMclustername, 'DBname': self.DBname,
            #            'DBnum': self.DBnum}
            keys = ['tname', 'tcluster', 'sname', 'scluster', 'Tvalue', 'tx', 'ty', 'tz', 'tlimx1', 'tlimx2', 'tlimy1',
                    'tlimy2', 'tlimz1', 'tlimz2', 'sx', 'sy', 'sz', 'slimx1', 'slimx2', 'slimy1', 'slimy2', 'slimz1', 'slimz2', 'timepoint']

            resultsnames = data['resultsnames']
            clustername = data['clustername']
            regionname = data['regionname']
            networkmodel = data['network']
            network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel)
            nclusterlist = np.array([ncluster_list[i]['nclusters'] for i in range(len(ncluster_list))])

            cluster_data = np.load(clustername, allow_pickle=True).flat[0]
            cluster_properties = cluster_data['cluster_properties']
            cluster_info, rname_list, ncluster_list = get_cluster_position_details(cluster_properties)

            results = []
            Tvalue_list = []
            connid_list = []   # identify connections - to be able to remove redundant ones later
            for networkcomponent, fname in enumerate(resultsnames):
                semresults = np.load(fname, allow_pickle=True).flat[0]
                sem_one_target = semresults['sem_one_target_results']
                ntclusters = len(sem_one_target)

                target = network[networkcomponent]['target']
                sources = network[networkcomponent]['sources']
                targetnum = network[networkcomponent]['targetnum']
                sourcenums = network[networkcomponent]['sourcenums']
                targetname = cluster_info[targetnum]['rname']
                targetlimits = cluster_info[targetnum]['regionlimits']

                for tt in range(ntclusters):
                    targetcoords = cluster_info[targetnum]['cluster_coords'][tt, :]
                    beta = sem_one_target[tt]['b']
                    ncombinations, ntimepoints, NP, nsources = np.shape(beta)

                    mean_beta = np.mean(beta, axis=2)
                    se_beta = np.std(beta, axis=2) / np.sqrt(NP)
                    Tbeta = mean_beta / (se_beta + 1.0e-10)
                    Tthresh = stats.t.ppf(1 - pthreshold, NP - 1)
                    beta_sig = abs(Tbeta) > Tthresh    # size is ncombinations x ntimepoints x nsources
                    # organize significant results
                    combo, nt, ss = np.where(beta_sig)  # significant connections during this time period

                    for ii in range(len(combo)):
                        # get region names, cluster numbers, etc.
                        Tvalue = Tbeta[combo[ii], nt[ii], ss[ii]]
                        timepoint = nt[ii]
                        sourcename = cluster_info[sourcenums[ss[ii]]]['rname']
                        mlist = pysem.ind2sub_ndims(nclusterlist[sourcenums],combo[ii]).astype(int)   # cluster number for each source
                        sourcecluster = mlist[ss[ii]]
                        sourcecoords = cluster_info[sourcenums[ss[ii]]]['cluster_coords'][sourcecluster, :]
                        sourcelimits = cluster_info[sourcenums[ss[ii]]]['regionlimits']

                        connid = nt[ii]*1e7 + targetnum*1e5 + tt*1e3 + sourcenums[ss[ii]]*10 + sourcecluster

                        values = np.concatenate(([targetname, tt, sourcename, sourcecluster, Tvalue],
                             list(targetcoords), list(targetlimits), list(sourcecoords), list(sourcelimits), [timepoint]))
                        entry = dict(zip(keys, values))

                        # entry = {'tname': targetname, 'tcluster': tt, 'sname': sourcename,
                        #          'scluster': sourcecluster, 'Tvalue': Tvalue_list[ii], 'tx': targetcoords[0], 'ty': targetcoords[1],
                        #          'tz': targetcoords[2],'tlimx1': targetlimits[0], 'tlimx2': targetlimits[1], 'tlimy1': targetlimits[2],
                        #          'tlimy2': targetlimits[3], 'tlimz1': targetlimits[4], 'tlimz2': targetlimits[5],
                        #          'sx': sourcecoords[0], 'sy': sourcecoords[1], 'sz': sourcecoords[2],
                        #          'slimx1': sourcelimits[0], 'slimx2': sourcelimits[1], 'slimy1': sourcelimits[2],
                        #          'slimy2': sourcelimits[3], 'slimz1': sourcelimits[4], 'slimz2': sourcelimits[5], 'timepoint':timepoint}

                        results.append(entry)
                        Tvalue_list.append(Tvalue)
                        connid_list.append(connid)

            # eliminate redundant values, for repeats keep the one with the largest Tvalue
            results2, Tvalue_list2 = remove_reps_and_sort(np.array(connid_list), np.array(Tvalue_list), results)

            # separate by timepoints
            timepoint_list = [results2[ii]['timepoint'] for ii in range(len(results2))]
            times = np.unique(timepoint_list)

            # timepoint_list = np.zeros(len(results2))
            # for ii in range(len(results2)):
            #     timepoint_list[ii] = results2[ii]['timepoint']

            # # sort by significance, and separate by timepoints
            # dorder = np.argsort(np.abs(Tvalue_list))
            # results2 = copy.deepcopy(results)
            # timepoints2 = []
            # for ii,dd in enumerate(dorder):
            #     results2[ii] = copy.deepcopy(results[dd])
            #     timepoints2[ii] = results[dd]['timepoint']

        # still need to split the data according to timepoints
        p,f = os.path.split(filename)
        f2,e = os.path.splitext(f)
        excelfilename = os.path.join(p,f2+'_2ndlevel.xlsx')
        for timepoint in times:
            indices = np.where(timepoint_list == timepoint)[0]
            results1 = []
            for ii in indices:
                results1.append(results2[ii])

            excelsheetname = 'network ' + str(timepoint)
            pydisplay.pywriteexcel(results1, excelfilename, excelsheetname, 'append')

        return results2

    if datafiletype == 2:
        # analyzing BOLD responses
        region_properties = data['region_properties']
        # regiondata_entry = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize,'rname': rname}

        p, f = os.path.split(filename)
        f2, e = os.path.splitext(f)
        excelfilename = os.path.join(p, f2 + '_2ndlevel.xlsx')

        nregions = len(region_properties)
        for rr in nregions:
            tc = region_properties[rr]['tc']          # nclusters x tsize_total
            tc_sem = region_properties[rr]['tc_sem']
            tsize = region_properties[rr]['tsize']
            nruns_per_person = region_properties[rr]['nruns_per_person']
            rname = region_properties[rr]['rname']
            NP = len(nruns_per_person)
            nclusters, tsize_total = np.shape(tc)

            # change shape of timecourse data array - prep data
            tc_per_person = np.zeros((nclusters,tsize,NP))
            tc_per_person_sem = np.zeros((nclusters,tsize,NP))
            for nn in NP:
                nruns = nruns_per_person[nn]
                t1 = np.sum(nruns_per_person[:nn])*tsize
                t2 = np.sum(nruns_per_person[:(nn+1)])*tsize
                tp = list(range(t1,t2))
                tc1 = np.mean(np.reshape(tc[:,tp],(nclusters,tsize,nruns)),axis = 2)
                tc1_sem = np.mean(np.reshape(tc_sem[:,tp],(nclusters,tsize,nruns)),axis = 2)
                tc_per_person[:,:,nn] = tc1
                tc_per_person_sem[:,:,nn] = tc1_sem

            # test significance of each timepoint
            mean_tc = np.mean(tc_per_person, axis = 2)
            sem_tc = np.std(tc_per_person, axis = 2)/np.sqrt(NP)
            T = mean_tc/(sem_tc + 1.0e-10)

            # check significance (or just write results to excel?)
            Tthresh = stats.t.ppf(1 - pthreshold, NP - 1)
            sig = abs(T) > Tthresh

            # mean_tc  :  nclusters x tsize
            # sem_tc  :  nclusters x tsize
            # T   : nclusters x tsize
            # sig : nclusters x tsize

            keys = ['rname avg','rname sem','rname T','rname sig']
            outputdata = []
            for tt in tsize:
                values = [mean_tc[tt],sem_tc[tt],T[tt],sig[tt]]
                entry = dict(zip(keys,values))
                outputdata.append(entry)

            excelsheetname = rname
            pydisplay.pywriteexcel(outputdata, excelfilename, excelsheetname, 'append')


