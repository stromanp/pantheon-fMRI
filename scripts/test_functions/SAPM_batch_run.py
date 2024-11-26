# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pantheon\venv')
import numpy as np
import matplotlib.pyplot as plt
import pysapm
import os
import copy
import pydatabase

datadir = r'E:\SAPMresults_Dec2022'
resultsbase = ['RSnostim','Sens', 'Low', 'Pain','High', 'RSstim']   # , 'allpainconditions'
covnamebase = ['RSnostim','Sens', 'Low', 'Pain2','High', 'RSstim2']   # , 'allpain'

nresults = len(resultsbase)
cnums = [3, 2, 4, 2, 4, 2, 3, 0, 1, 2]
nametag = '_3242423012'

resultsnames = [resultsbase[x]+nametag+'_norm_results.npy' for x in range(nresults)]
paramsnames = [resultsbase[x]+nametag+'_norm_params.npy' for x in range(nresults)]
covnames = [covnamebase[x]+'_covariates.npy' for x in range(nresults)]
regiondatanames = [resultsbase[x]+'_regiondata2.npy' for x in range(nresults)]
clusterdataname = r'E:/SAPMresults_Dec2022\Pain_equalsize_cluster_def.npy'
networkfile = r'E:\SAPMresults_Dec2022\network_model_Jan2023_SAPM.xlsx'
DBname = r'E:\graded_pain_database_May2022.xlsx'
timepoint = 'all'
epoch = 'all'
betascale=0.1

create_null_data = False
if create_null_data:
        npeople = 1000
        regiondataname = r'E:\SAPMresults_Dec2022\High_regiondata2.npy'
        covariatesname = r'E:\SAPMresults_Dec2022\High_covariates.npy'
        outputname, outputcovname = pysapm.generate_null_data_set(regiondataname, covariatesname, npeople=npeople)


# load params
paramsnamefull = os.path.join(datadir,paramsnames[0])

run_analysis = True
if run_analysis:

        # for nn in range(nresults):
        nn = 3
        regiondataname = os.path.join(datadir,regiondatanames[nn])
        SAPMresultsname = os.path.join(datadir,resultsnames[nn])
        SAPMparametersname = os.path.join(datadir,paramsnames[nn])
        pysapm.SAPMrun_V2(cnums, regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkfile, DBname, timepoint,
                epoch, betascale=betascale, reload_existing=False)

generate_results = True
if generate_results:
        # now generate results figures----------------------------------------------------------------
        print('generating figures to save as svg files ...')
        # outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_SourceModel', 'DrawSAPMdiagram',
        #                  'DrawAnatomy_axial', 'DrawAnatomy_sagittal']


        SRresultsdir = copy.deepcopy(datadir)
        target_region_list = ['C6RD','DRt','NGC','NRM','PAG','LC','NTS','PBN','Hypothalamus','Thalamus']

        SRpvalue = 0.00156
        SRpvalue = 0.01
        versiontag = '_p01'

        ylimits = [1.2]
        for nn in range(nresults):
                SRnametag = resultsbase[nn] + nametag
                covnamefull = os.path.join(datadir,covnames[nn])
                cov = np.load(covnamefull, allow_pickle=True).flat[0]
                charlist = cov['GRPcharacteristicslist']
                x = charlist.index('painrating')
                covvals = cov['GRPcharacteristicsvalues'][x].astype(float)

                SRparamsname = os.path.join(datadir,paramsnames[nn])
                SRresultsname = os.path.join(datadir,resultsnames[nn])
                SRvariant = 0  # get this from the order information

                c = list(range(len(covvals)))
                SRgroup = c

                SRoptionvalue = 'Plot_SourceModel'
                for regionname in target_region_list:
                        SRtargetregion = copy.deepcopy(regionname)
                        outputname = pysapm.display_SAPM_results(124, SRnametag, covvals, SRoptionvalue,
                                                                 SRresultsdir, SRparamsname, SRresultsname, SRvariant,
                                                                 SRgroup, SRtargetregion, SRpvalue, ylimits, 'none', False)
                        print('created figure {}'.format(outputname))

                # draw SAPM diagram

                regionnames = 'regions'
                SRoptionvalue = 'B_Significance'
                sheetname = 'Sheet1'
                drawregionsfile = r'E:\SAPMresults_Dec2022\SAPM_network_plotting_definition.xlsx'
                regions = pysapm.define_drawing_regions_from_file(drawregionsfile)
                statname = 'B'
                figurenumber = 200
                scalefactor = 'auto'
                results_file = pysapm.display_SAPM_results(124, SRnametag+versiontag, covvals, SRoptionvalue,
                                                         SRresultsdir, SRparamsname, SRresultsname, SRvariant,
                                                         SRgroup, SRtargetregion, SRpvalue, ylimits, 'none', False)

                if os.path.isfile(results_file):
                        try:
                                pysapm.draw_sapm_plot_SO(results_file, sheetname, regionnames, regions, statname, figurenumber, scalefactor,
                                                  cnums, thresholdtext='abs>0', writefigure=True)
                        except:
                                print('could not draw plot_SO for {} {}'.format(SRoptionvalue, covnames[nn]))


        ylimits = [1.2]
        for nn in range(1,nresults):
                SRnametag = resultsbase[nn] + nametag
                covnamefull = os.path.join(datadir,covnames[nn])
                cov = np.load(covnamefull, allow_pickle=True).flat[0]
                charlist = cov['GRPcharacteristicslist']
                x = charlist.index('painrating')
                covvals = cov['GRPcharacteristicsvalues'][x].astype(float)

                SRparamsname = os.path.join(datadir,paramsnames[nn])
                SRresultsname = os.path.join(datadir,resultsnames[nn])
                SRvariant = 0  # get this from the order information

                c = list(range(len(covvals)))
                SRgroup = c

                regionnames = 'regions'
                SRoptionvalue = 'B_Regression'
                sheetname = 'Sheet1'
                drawregionsfile = r'E:\SAPMresults_Dec2022\SAPM_network_plotting_definition.xlsx'
                regions = pysapm.define_drawing_regions_from_file(drawregionsfile)
                statname = 'slope'
                figurenumber = 201
                scalefactor = 'auto'
                if np.var(covvals) > 0:
                        results_file = pysapm.display_SAPM_results(124, SRnametag+versiontag, covvals, SRoptionvalue,
                                                                 SRresultsdir, SRparamsname, SRresultsname, SRvariant,
                                                                 SRgroup, SRtargetregion, SRpvalue, ylimits, 'none', False)

                        if os.path.isfile(results_file):
                                try:
                                        pysapm.draw_sapm_plot_SO(results_file, sheetname, regionnames, regions, statname, figurenumber, scalefactor,
                                                          cnums, thresholdtext='abs>0', writefigure=True)
                                except:
                                        print('could not draw plot_SO for {} {}'.format(SRoptionvalue, covnames[nn]))


distribution_of_R2 = True
if distribution_of_R2:
        R2data = []
        for nn in range(nresults):
                SRnametag = resultsbase[nn] + nametag
                covnamefull = os.path.join(datadir,covnames[nn])
                cov = np.load(covnamefull, allow_pickle=True).flat[0]
                charlist = cov['GRPcharacteristicslist']
                x = charlist.index('painrating')
                covvals = cov['GRPcharacteristicsvalues'][x].astype(float)

                SRparamsname = os.path.join(datadir,paramsnames[nn])
                SRresultsname = os.path.join(datadir,resultsnames[nn])

                results = np.load(SRresultsname,allow_pickle=True)
                R2list = [results[x]['R2total'] for x in range(len(results))]

                R2data.append({'R2list':R2list, 'covdata':covvals, 'name':resultsnames[nn]})

        plt.close(101)
        fig = plt.figure(101)
        plt.plot(R2data[0]['R2list'][:-1],R2data[5]['R2list'],'or')

        plt.close(102)
        fig = plt.figure(102)
        c = list(range(len(R2data[1]['R2list'])))
        c.remove(5)
        d = np.array([R2data[1]['R2list'][x] for x in c])
        plt.plot(d,R2data[3]['R2list'],'or')

        plt.close(103)
        fig = plt.figure(103)
        plt.plot(R2data[2]['R2list'],R2data[4]['R2list'],'or')


        # plt.close(111)
        # fig = plt.figure(111)
        # plt.plot(R2data[5]['covdata'],R2data[0]['R2list'][:-1],'or')
        # plt.plot(R2data[5]['covdata'],R2data[5]['R2list'],'ob')
        #
        # plt.close(112)
        # fig = plt.figure(112)
        # plt.plot(R2data[1]['covdata'],R2data[1]['R2list'],'or')
        # plt.plot(R2data[3]['covdata'],R2data[3]['R2list'],'ob')
        #
        # plt.close(113)
        # fig = plt.figure(113)
        # plt.plot(R2data[2]['covdata'],R2data[2]['R2list'],'or')
        # plt.plot(R2data[4]['covdata'],R2data[4]['R2list'],'ob')


# special case
run_special_case = False
if run_special_case:
        # E:/SAPMresults_Dec2022\AllPainE_3000444141_results.npy

        rname = os.path.join(datadir,'AllPain_equal_regiondata.npy')
        regiondata = np.load(rname,allow_pickle=True).flat[0]
        region_properties = regiondata['region_properties']
        DBnum = regiondata['DBnum']

        resultsname = os.path.join(datadir,'AllPainE_1202023213_results.npy')
        results = np.load(resultsname,allow_pickle=True)
        R2list = [results[x]['R2avg'] for x in range(len(results))]
        R2list = np.array(R2list)

        prefix = 'xptc'

        filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list', separate_conditions = True)

        c1 = np.where(R2list > np.mean(R2list))[0]
        c2 = np.where(R2list <= np.mean(R2list))[0]

        highlist = []
        for x in c1: highlist += dbnum_person_list[x]
        lowlist = []
        for x in c2: lowlist += dbnum_person_list[x]

        DBnumlist_highR2 = {'dbnumlist':highlist}
        DBnumlist_lowR2 = {'dbnumlist':lowlist}

        np.save(r'E:\allpain_list_lowR2_v2.npy',DBnumlist_lowR2)
        np.save(r'E:\allpain_list_highR2_v2.npy',DBnumlist_highR2)

        filename_list1, dbnum_person_list1, NP1 = pydatabase.get_datanames_by_person(DBname, highlist, prefix, mode='list', separate_conditions = True)
        filename_list2, dbnum_person_list2, NP2 = pydatabase.get_datanames_by_person(DBname, lowlist, prefix, mode='list', separate_conditions = True)

        # compare with individual runs
        resultsname1 = os.path.join(datadir,'HighR2_1202023213_results.npy')
        paramsname1 = os.path.join(datadir,'HighR2_1202023213_params.npy')
        resultsname2 = os.path.join(datadir,'LowR2_1202023213_results.npy')
        paramsname2 = os.path.join(datadir,'LowR2_1202023213_params.npy')

        results1 = np.load(resultsname1,allow_pickle=True)
        R2list1 = [results1[x]['R2avg'] for x in range(len(results1))]
        R2list1 = np.array(R2list1)
        indexhigh = np.argmin(R2list1)
        dbnum_person_list1[indexhigh]
        indexall = c1[indexhigh]
        dbnum_person_list[indexall]

        results2 = np.load(resultsname2,allow_pickle=True)
        R2list2 = [results2[x]['R2avg'] for x in range(len(results2))]
        R2list2 = np.array(R2list2)

        Sinput = results[indexall]['Sinput']
        Sinput1 = results1[indexhigh]['Sinput']

        Mintrinsic = results[indexall]['Mintrinsic']
        Mintrinsic1 = results1[indexhigh]['Mintrinsic']

        betavals = results[indexall]['betavals']
        betavals1 = results1[indexhigh]['betavals']


        # allpain runs
        regiondataname = os.path.join(datadir,r'AllPain_equal_highR2_regiondata.npy')
        regiondata = np.load(regiondataname, allow_pickle=True).flat[0]
        region_properties = regiondata['region_properties']
        DBnum = regiondata['DBnum']

        addedtag = 'r3'

        paramsname = os.path.join(datadir,r'HighR2_1202023213'+addedtag+'_params.npy')
        params = np.load(paramsname, allow_pickle=True).flat[0]
        tcdata_centered = params['tcdata_centered']
        resultsname = os.path.join(datadir,r'HighR2_1202023213'+addedtag+'_results.npy')
        results = np.load(resultsname, allow_pickle=True)
        R2list_high = [results[x]['R2total'] for x in range(len(results))]

        tplist_full = params['tplist_full']
        nperson = 3
        tp = tplist_full[epochnum][nperson]['tp']
        tsize_total = len(tp)
        nruns = nruns_per_person[nperson]
        tc_allpain = tcdata_centered[:, tp]
        print('all pain runs R2 person {} is {:.4f}'.format(nperson,R2list_high[nperson]))

        filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list')
        print('dbnums for person {} are {}'.format(nperson, dbnum_person_list[nperson]))


        # oneperson3 runs
        regiondataname = os.path.join(datadir,r'oneperson3_regiondata.npy')
        regiondata = np.load(regiondataname, allow_pickle=True).flat[0]
        region_properties = regiondata['region_properties']
        DBnum = regiondata['DBnum']

        paramsname = os.path.join(datadir,r'one3_1202023213_params.npy')
        params = np.load(paramsname, allow_pickle=True).flat[0]
        tcdata_centered = params['tcdata_centered']
        resultsname = os.path.join(datadir,r'one3_1202023213_results.npy')
        results = np.load(resultsname, allow_pickle=True)
        R2list_one = [results[x]['R2total'] for x in range(len(results))]

        tplist_full = params['tplist_full']
        nperson = 0
        tp = tplist_full[epochnum][nperson]['tp']
        tsize_total = len(tp)
        nruns = nruns_per_person[nperson]
        tc_one3 = tcdata_centered[:, tp]
        print('one person runs R2 person {} is {:.4f}'.format(nperson,R2list_one[nperson]))

        filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list')
        print('dbnums for person {} are {}'.format(nperson, dbnum_person_list[nperson]))