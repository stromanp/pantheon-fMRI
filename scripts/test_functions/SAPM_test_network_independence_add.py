# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pantheon\venv')
# check to see if a defined network produces a full-rank connectivity matrix
# or has linearly dependent elements
import numpy as np
import pysapm
import copy


networkfile = r'E:/SAPMresults2_Oct2022/network_model_5cluster_Nov2022_w_3intrinsics.xlsx'
network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = pysapm.load_network_model_w_intrinsics(networkfile)
ncluster_list = np.array([nclusterlist[x]['nclusters'] for x in range(len(nclusterlist))])
cluster_name = [nclusterlist[x]['name'] for x in range(len(nclusterlist))]
not_latent = [x for x in range(len(cluster_name)) if 'intrinsic' not in cluster_name[x]]
ncluster_list = ncluster_list[not_latent]
full_rnum_base = [np.sum(ncluster_list[:x]) for x in range(len(ncluster_list))]

namelist = [cluster_name[x] for x in not_latent]
regionlist = copy.deepcopy(namelist)
namelist += ['Rtotal']
namelist += ['R ' + cluster_name[x] for x in not_latent]
nregions = len(regionlist)


# test the full network, and then take out one element of the network at a time
original_network = copy.deepcopy(network)
for TT in range(len(network)+1):
    if TT == len(network):
        sourcenums = []
        test_n_sources = 1
    else:
        sourcenums = network[TT]['sourcenums']
        test_n_sources = nregions-len(sourcenums)-1  # add missing regions
    for SS in range(test_n_sources):
        if TT == len(network):
            network = copy.deepcopy(original_network)
        else:
            # c = [x for x in range(test_n_sources) if x != SS]
            network = copy.deepcopy(original_network)
            # sources = [network[TT]['sources'][x] for x in c]
            # sourcenums = [network[TT]['sourcenums'][x] for x in c]

            missing_sourcenums = [x for x in range(nregions) if (x not in network[TT]['sourcenums']) & (x != TT) ]
            sourcenums = network[TT]['sourcenums'] + [missing_sourcenums[SS]]
            sources = network[TT]['sources'] + [regionlist[missing_sourcenums[SS]]]
            network[TT]['sourcenums'] = sourcenums
            network[TT]['sources'] = sources
            added_sourcenum = missing_sourcenums[SS]
            added_source = regionlist[missing_sourcenums[SS]]

        # now construct the network matrices and test Mconn for independence
        beta_list = []
        nbeta = 0
        targetnumlist = []
        beta_id = []
        sourcelist = []
        for nn in range(len(network)):
            target = network[nn]['targetnum']
            sources = network[nn]['sourcenums']
            targetnumlist += [target]
            for mm in range(len(sources)):
                source = sources[mm]
                sourcelist += [source]
                betaname = '{}_{}'.format(source, target)
                entry = {'name': betaname, 'number': nbeta, 'pair': [source, target]}
                beta_list.append(entry)
                beta_id += [1000 * source + target]
                nbeta += 1

        Nintrinsic = fintrinsic_count + vintrinsic_count
        nregions = len(ncluster_list)
        ncon = nbeta - Nintrinsic

        # reorder to put intrinsic inputs at the end-------------
        beta_list2 = []
        beta_id2 = []
        x = np.where(np.array(sourcelist) < nregions)[0]
        for xx in x:
            beta_list2.append(beta_list[xx])
            beta_id2 += [beta_id[xx]]
        for sn in range(nregions, nregions + Nintrinsic):
            x = np.where(np.array(sourcelist) == sn)[0]
            for xx in x:
                beta_list2.append(beta_list[xx])
                beta_id2 += [beta_id[xx]]

        for nn in range(len(beta_list2)):
            beta_list2[nn]['number'] = nn

        beta_list = beta_list2
        beta_id = beta_id2

        beta_pair = []
        Mconn = np.zeros((nbeta, nbeta))
        count = 0
        for nn in range(len(network)):
            target = network[nn]['targetnum']
            sources = network[nn]['sourcenums']
            for mm in range(len(sources)):
                source = sources[mm]
                conn1 = beta_id.index(source * 1000 + target)
                if source >= nregions:  # intrinsic input
                    conn2 = conn1
                    Mconn[conn1, conn2] = 1  # set the intrinsic beta values
                else:
                    x = targetnumlist.index(source)
                    source_sources = network[x]['sourcenums']
                    for nn in range(len(source_sources)):
                        ss1 = source_sources[nn]
                        conn2 = beta_id.index(ss1 * 1000 + source)
                        beta_pair.append([conn1, conn2])
                        count += 1
                        Mconn[conn1, conn2] = count

        # prep to index Mconn for updating beta values
        beta_pair = np.array(beta_pair)
        ctarget = beta_pair[:, 0]
        csource = beta_pair[:, 1]

        latent_flag = np.zeros(len(ctarget))
        found_latent_list = []
        for nn in range(len(ctarget)):
            if csource[nn] >= ncon:
                if not csource[nn] in found_latent_list:
                    latent_flag[nn] = 1
                    found_latent_list += [csource[nn]]


        # setup Minput matrix--------------------------------------------------------------
        # Sconn = Mconn @ Sconn    # propagate the intrinsic inputs through the network
        # Sinput = Minput @ Mconn
        Minput = np.zeros((nregions, nbeta))  # mixing of connections to model the inputs to each region
        betanamelist = [beta_list[a]['name'] for a in range(nbeta)]
        for nn in range(len(network)):
            target = network[nn]['targetnum']
            sources = network[nn]['sourcenums']
            for mm in range(len(sources)):
                source = sources[mm]
                betaname = '{}_{}'.format(source, target)
                x = betanamelist.index(betaname)
                Minput[target, x] = 1

        # initialize Mconn values
        betascale = 1.0
        beta_initial = betascale*np.random.randn(len(csource))

        Mconn[ctarget,csource] = beta_initial

        det = np.linalg.det(Mconn)
        w, v = np.linalg.eig(Mconn)

        rank = np.linalg.matrix_rank(Mconn)

        if TT == len(network):
            print('full network (size {}):   rank = {}'.format(np.shape(Mconn), rank))
        else:
            print('Target {}   added Source {}:   rank = {}'.format(TT,added_sourcenum,rank))