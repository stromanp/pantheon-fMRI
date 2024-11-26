import numpy as np
from sklearn.decomposition import PCA

def nperms(n, input_list):
	# find all possible combinations of n values selected out of the input list
	# without duplicates

	nvals = len(input_list)
	permutations = []
	dimspec = [nvals for xx in range(n)]
	grid = np.ones(dimspec)

	coords = np.zeros((n,nvals**n))
	for nn in range(n):
		predims = nn
		postdims = n-nn-1
		d1 = np.tile(np.array(range(nvals)),nvals**postdims)
		d1 = np.repeat(d1,nvals**predims)
		coords[nn,:] = d1

	dcoords = coords[1:,:]-coords[:-1,:]
	keeplist = [xx for xx in range(nvals**n) if np.min(dcoords[:,xx]) > 0]
	perms = np.array(coords[:,keeplist]).astype(int)

	nperms = np.shape(perms)[1]
	output_list = []
	for xx in range(nperms):
		sample = [input_list[c] for c in perms[:,xx]]
		output_list += [sample]

	return np.array(output_list), perms



# cx = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
# cy = [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]
# cz = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]


def cluster_search_pca(regiondataname, vintrinsic_count, fintrinsic_count, fintrinsic_tc = []):
	regiondataname = r'E:\FM2021data\HCstim_region_data_Jan2024.npy'
	vintrinsic_count = 2
	fintrinsic_count = 1

	regiondata = np.load(regiondataname, allow_pickle=True).flat[0]
	region_properties = regiondata['region_properties']
	nregions = len(region_properties)
	rnamelist = [region_properties[xx]['rname'] for xx in range(nregions)]
	nruns_per_person = region_properties[0]['nruns_per_person']
	tsize = region_properties[0]['tsize']

	tc0 = region_properties[0]['tc']
	nclusterlist = [np.shape(region_properties[xx]['tc'])[0] for xx in range(nregions)]
	nclusters_total = np.sum(nclusterlist)
	NP = len(nruns_per_person)
	nruns_total = np.sum(nruns_per_person)

	paramsfile = r'E:\FM2021data\HCstim_1013302243_V2_params.npy'
	params = np.load(paramsfile, allow_pickle=True).flat[0]
	fintrinsic_base = copy.deepcopy(params['fintrinsic_base'])
	fintrinsic_base -= np.mean(fintrinsic_base)
	fintrinsic_tc = copy.deepcopy(fintrinsic_base)

	vdata = []
	for nn in range(NP):
		# data per person
		Sinput = np.zeros((nclusters_total,tsize*nruns_per_person[nn]))
		t1 = np.sum(nruns_per_person[:nn])*tsize
		t2 = np.sum(nruns_per_person[:(nn+1)])*tsize
		for rr in range(nregions):
			c1 = np.sum(nclusterlist[:rr]).astype(int)
			c2 = np.sum(nclusterlist[:(rr+1)])
			Sinput[c1:c2,:] = region_properties[rr]['tc'][:,t1:t2]

		if fintrinsic_count > 0:
			#  Sinput = f_b @ flatent_tc    - fit fixed latent component if there is one
			flatent_tc = np.repeat(fintrinsic_tc, nruns_per_person[nn], axis=1)
			f_b = Sinput @ flatent_tc.T @ np.linalg.inv(flatent_tc @ flatent_tc.T)
			f_fit = f_b @ flatent_tc
			Sinput_res = Sinput - f_fit   # take out fixed latent component
			var_flatent = np.var(Sinput,axis=1) - np.var(Sinput_res,axis=1)
		else:
			Sinput_res = copy.deepcopy(Sinput)
			var_flatent = np.zeros(nclusters_total)

		# get principal components and weights for timecourse data in Sinput
		# nregions, tsizefull = np.shape(Sinput)
		# Sin_std = np.repeat(np.std(Sinput, axis=1)[:, np.newaxis], tsizefull, axis=1)
		# Sinput_norm = Sinput / Sin_std
		pca = PCA(n_components=nclusters_total)
		pca.fit(Sinput_res)
		# S_pca_ = pca.fit(Sinput).transform(Sinput)

		components = pca.components_
		evr = pca.explained_variance_ratio_
		ev = pca.explained_variance_
		# get loadings
		mu = np.mean(Sinput_res, axis=0)   # the average component is separate and messes up everything because
											# it might not be linearly independent of other components
		vm = np.var(mu, ddof=1)
		mu = np.repeat(mu[np.newaxis, :], nclusters_total, axis=0)

		loadings = pca.transform(Sinput_res)
		fit_check = (loadings @ components) + mu

		# rr=10
		# f = np.sum(np.repeat(loadings[rr, :][:, np.newaxis], tsize * nruns_per_person[nn], axis=1) * components, axis=0) + mu[rr, :]
		# vf = np.var(f)
		#
		# f1 = np.sum(np.repeat(loadings[rr, :1][:, np.newaxis], tsize * nruns_per_person[nn], axis=1) * components[:1,:], axis=0) + mu[rr, :]
		# vf1 = np.var(f1)
		# fc1 = (loadings[rr,:1] @ components[:1,:]) + mu[rr,:]
		# vfc1 = np.var(fc1)

		# now find which set of clusters can be best explained by only vintrinsic_count PCA components
		vc = np.var(components,axis=1, ddof = 1)
		v_by_component = (loadings**2) * np.repeat(vc[:,np.newaxis],nclusters_total,axis=1)
		vS = np.var(Sinput_res - mu, axis=1, ddof = 1)
		v_ratio_by_component = v_by_component / np.repeat(vS[:,np.newaxis],nclusters_total,axis=1)

		if fintrinsic_count > 0:
			v_ratio_flatent = var_flatent / vS
			v_ratio_by_component = np.concatenate((v_ratio_by_component,v_ratio_flatent[:,np.newaxis]),axis=1)

		# v_ratio_by_component is [cluster_number x component]
		# find which combination of cluster numbers gives the highest total for some set of compoents across all people
		vdata.append({'vratio':v_ratio_by_component})
		if nn == 0:
			vdata_group = copy.deepcopy(v_ratio_by_component[:,:,np.newaxis])
		else:
			vdata_group = np.concatenate((vdata_group,v_ratio_by_component[:,:,np.newaxis]),axis=2)

	# find one cluster per region that gives the best overall fit for all people
	# permutations of first Nsearch components
	Nsearch = 8
	cnumset = list(range(Nsearch))
	component_list, perms = nperms(vintrinsic_count, cnumset)   # combinations of PCA terms to check
	ncombinations = np.shape(component_list)[0]

	multistep_results = []
	nrepeats = 10
	for rrr in range(nrepeats):
		# gradient-descent type search---------------------------------------
		# initial cluster guess
		offset = np.array([0] + list(np.cumsum(nclusterlist))[:-1])
		cluster_numbers = np.zeros(len(nclusterlist))
		for nn in range(len(nclusterlist)):
			cnum = np.random.choice(range(nclusterlist[nn]))
			cluster_numbers[nn] = copy.deepcopy(cnum)
		cnumlist = (cluster_numbers + offset).astype(int)

		fixed_clusters = []
		lastavg = 0
		maxiter = 1000
		iter = 0
		converging = True
		while converging and (iter < maxiter):
			iter += 1
			nbetterclusters = 0
			random_region_order = list(range(nregions))
			np.random.shuffle(random_region_order)
			for nnn in random_region_order:
				avg_var_list = np.zeros(nclusterlist[nnn])
				print('testing region {}'.format(nnn))
				if nnn in fixed_clusters:
					print('cluster for region {} is fixed at {}'.format(nnn, cluster_numbers[nnn]))
				else:
					for ccc in range(nclusterlist[nnn]):
						test_clusters = copy.deepcopy(cluster_numbers)
						if test_clusters[nnn] == ccc:  # no change in cluster number from last run
							avg_var_list[ccc] = lastavg
							print('  using cluster {}  total of avg. variance explained for the group is {:.3f} - current cluster'.format(ccc, avg_var_list[ccc]))
						else:
							test_clusters[nnn] = ccc
							cnumlist = (test_clusters + offset).astype(int)

							# find best combination in each person
							best_var = np.zeros(NP)
							for pp in range(NP):
								var_check_list = np.zeros(ncombinations)
								for nn in range(ncombinations):
									complist = component_list[nn, :]
									if fintrinsic_count > 0:
										complist = np.concatenate((complist, [nclusters_total]))

									vdata_subset = vdata_group[cnumlist, :, pp]
									# check = np.mean(np.sum(vdata_group[:,cc,:],axis=1),axis=1)
									var_check_list[nn] = np.sum(vdata_subset[:,complist])  # how much variance is accounted for by these components, for each cluster, in each person
								best_var[pp] = np.max(var_check_list)
							avg_var_list[ccc] = np.mean(best_var)

							# entry = {'R2list': R2list, 'R2list2': R2list2, 'region': nnn, 'cluster': ccc}
							# results_record.append(entry)
							print('  using cluster {}  total of avg. variance explained for the group is {:.3f}'.format(ccc, avg_var_list[ccc]))

					x = np.argmax(avg_var_list)
					this_avg = avg_var_list[x]
					delta_avg = this_avg - lastavg
					if this_avg > lastavg:
						cluster_numbers[nnn] = x
						nbetterclusters += 1
						lastavg = copy.deepcopy(this_avg)
					else:
						print('no improvement in clusters found ... region {}'.format(nnn))

					print('iter {} region {} new avg. variance = {:.3f}  previous avg. variance  = {:.3f}  delta variance = {:.3e} {}'.format(
							iter, nnn, this_avg, lastavg, delta_avg, time.ctime()))

			if nbetterclusters == 0:
				converging = False
				print('no improvement in clusters found in any region ...')

		print('\nbest cluster set so far is : {}'.format(cluster_numbers.astype(int)))
		multistep_results.append({'cluster_numbers':cluster_numbers, 'lastavg':lastavg})

	# find best of nrepeats
	lastavg_record = [multistep_results[xx]['lastavg'] for xx in range(nrepeats)]
	dd = np.argmax(lastavg_record)
	best_cluster_numbers = multistep_results[dd]['cluster_numbers'].astype(int)

	print('\nbest overall cluster set is : {}'.format(best_cluster_numbers))
	print('\n    overall variance estimate accounted for is {:.3f}'.format(np.max(lastavg_record)))


	#----------------end of gradient descent type search--------------------------


	search_results = []
	for pp in range(NP):
		vvals_mean = np.zeros(ncombinations)
		cnums = -1*np.ones((len(nclusterlist),ncombinations))
		for nn in range(ncombinations):
			cc = component_list[nn, :]
			if fintrinsic_count > 0:
				cc = np.concatenate((cc,[nclusters_total]))

			# check = np.mean(np.sum(vdata_group[:,cc,:],axis=1),axis=1)
			check = np.sum(vdata_group[:,cc,pp],axis=1)   # how much variance is accounted for by these components, for each cluster, in each person

			vvals = np.zeros(len(nclusterlist))
			for xx in range(len(nclusterlist)):
				c1 = np.sum(nclusterlist[:xx]).astype(int)
				c2 = np.sum(nclusterlist[:(xx+1)]).astype(int)
				set = check[c1:c2]
				cnums[xx,nn] = np.argmax(set)
				vvals[xx] = np.max(set)
			vvals_mean[nn] = np.mean(vvals)

		dd = np.argsort(vvals_mean)
		vvals_mean = vvals_mean[dd[::-1]]
		cnums_sorted = cnums[:,dd[::-1]]

		search_results.append({'cnums':cnums_sorted.astype(int), 'vvals':vvals_mean})

	bn = np.argmax(vvals_mean)
	best_cnums = search_results[bn]['cnums']
	best_vvals = search_results[bn]['vvals']

	return best_cnums, best_vvals