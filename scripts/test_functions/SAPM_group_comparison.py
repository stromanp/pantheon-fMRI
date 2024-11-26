import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

def run_sapm_group_comparison():
	pthreshold = 0.05

	resultsname1 = r'E:\FM2021data\FMstim_2341313432_results_corr.npy'
	paramsname1 = r'E:\FM2021data\FMstim_2341313432_params.npy'

	resultsname2 = r'E:\FM2021data\HCstim_2341313432_results_corr.npy'
	paramsname2 = r'E:\FM2021data\HCstim_2341313432_params.npy'

	results_type = 'input'
	outputfignametag = 'FMstim_vs_HCstim_input'

	results1 = np.load(resultsname1, allow_pickle=True)
	results2 = np.load(resultsname2, allow_pickle=True)
	DBname1 = results1[0]['DBname']
	DBname2 = results2[0]['DBname']
	DBnum1 = results1[0]['DBnum']
	DBnum2 = results2[0]['DBnum']
	if DBname1 == DBname2:
		print('The two results files use the same database file ...')
	else:
		print('The two results files use different database files ...')

	commondb1 = [DBnum1[xx] for xx in range(len(DBnum1)) if DBnum1[xx] in DBnum2]
	commondb2 = [DBnum2[xx] for xx in range(len(DBnum2)) if DBnum2[xx] in DBnum1]
	ncommon1 = len(commondb1)
	ncommon2 = len(commondb2)
	print('{} values of {} in DBnum1 are also in DBnum2'.format(ncommon1, len(DBnum1)))
	print('{} values of {} in DBnum2 are also in DBnum1'.format(ncommon2, len(DBnum2)))

	mark_times = np.array([[60., 70.], [120., 150.]])
	TR = 6.75

	layout = 'overlap'
	# layout = 'side_by_side'
	yrange = [-0.9, 0.9]

	sapm_group_comparison(resultsname1, paramsname1, resultsname2, paramsname2, pthreshold, outputfignametag, results_type, layout, yrange, mark_times, TR)



def sapm_group_comparison(resultsname1, paramsname1, resultsname2, paramsname2, pthreshold, outputfignametag, results_type = 'input', layout = 'overlap', yrange = [], mark_times = [], TR = 0.):

	if results_type not in ['input', 'output', 'fit', 'input+fit']:
		print('run_sapm_group_comparison.py ...')
		print('   acceptable inputs are: ''input'', ''output'', ''fit'', ''input+fit''')
	else:
		paradigm_centered1, Sinput_total1, Sconn_total1, fit_total1, Mintrinsic_total1, rnamelist = load_sapm_group_results(resultsname1, paramsname1)
		paradigm_centered2, Sinput_total2, Sconn_total2, fit_total2, Mintrinsic_total2, rnamelist = load_sapm_group_results(resultsname2, paramsname2)

		if results_type == 'input':
			nregions, tsize, NP1 = np.shape(Sinput_total1)
			Tvalues = temporal_ttest_comparison(Sinput_total1, Sinput_total2)
			pdata1 = copy.deepcopy(Sinput_total1)
			pdata2 = copy.deepcopy(Sinput_total2)

		if results_type == 'input+fit':
			nregions, tsize, NP1 = np.shape(Sinput_total1)
			Tvalues = temporal_ttest_comparison(Sinput_total1, Sinput_total2)
			pdata1 = copy.deepcopy(Sinput_total1)
			pdata2 = copy.deepcopy(Sinput_total2)

			pdata1b = copy.deepcopy(fit_total1)
			pdata2b = copy.deepcopy(fit_total2)

		if results_type == 'output':
			nregions_input, tsize, NP1 = np.shape(Sinput_total1)
			nregions, tsize, NP1 = np.shape(Sconn_total1)
			nlatent = nregions - nregions_input
			for mm in range(nlatent):
				rnamelist += ['int{}'.format(mm)]
			Tvalues = temporal_ttest_comparison(Sconn_total1, Sconn_total2)
			pdata1 = copy.deepcopy(Sconn_total1)
			pdata2 = copy.deepcopy(Sconn_total2)

		if results_type == 'fit':
			nregions, tsize, NP1 = np.shape(fit_total1)
			Tvalues = temporal_ttest_comparison(fit_total1, fit_total2)
			pdata1 = copy.deepcopy(fit_total1)
			pdata2 = copy.deepcopy(fit_total2)


		# plot results
		nregions, tsize, NP1 = np.shape(pdata1)
		nregions, tsize, NP2 = np.shape(pdata2)

		# pthreshold = 0.05
		Tthresh = stats.t.ppf(1-pthreshold,NP1+NP2-1)

		m1 = np.mean(pdata1, axis=2)
		sem1 = np.std(pdata1, axis=2) / np.sqrt(NP1)
		m2 = np.mean(pdata2, axis=2)
		sem2 = np.std(pdata2, axis=2) / np.sqrt(NP2)

		p,f1 = os.path.split(resultsname1)
		f,e = os.path.splitext(f1)

		for nn in range(nregions):
			windownum = nn+100
			title = copy.deepcopy(rnamelist[nn])
			sigflag = np.abs(Tvalues[nn,:]) > Tthresh
			plot_timecourse_data(title, windownum, m1[nn,:], sem1[nn,:], m2[nn,:], sem2[nn,:], layout, yrange, mark_times, TR, sigflag)

			figname = os.path.join(p,outputfignametag + '_{}.svg'.format(title) )
			plt.savefig(figname)


def temporal_ttest_comparison(tcdata1, tcdata2):
	nregions, tsize, NP1 = np.shape(tcdata1)
	nregions, tsize, NP2 = np.shape(tcdata2)

	# unpaired t-test on each time point
	mean1 = np.mean(tcdata1, axis=2)
	var1 = np.var(tcdata1, axis=2)
	mean2= np.mean(tcdata2, axis=2)
	var2 = np.var(tcdata2, axis=2)

	# pooled standard deviation:
	sp = np.sqrt(((NP1 - 1) * var1 + (NP2 - 1) * var2) / (NP1 + NP2 - 2))
	Tvalues = (mean1 - mean2) / (sp * np.sqrt(1 / NP1 + 1 / NP2) + 1.0e-20)

	return Tvalues



def load_sapm_group_results(resultsname, paramsname):

	SAPMresults_load = np.load(resultsname, allow_pickle=True)
	SAPMparams = np.load(paramsname, allow_pickle=True).flat[0]
	NP = len(SAPMresults_load)

	# load SAPM parameters
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

	nbeta, tsize_full = np.shape(SAPMresults_load[0]['Sconn'])
	ncon = nbeta - Nintrinsic
	if fintrinsic_count > 0:
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
	if fintrinsic_count > 0:
		ftemp = paradigm_centered[0, et1:et2]

	DBrecord = np.zeros((nbeta, nbeta, NP))
	Brecord = np.zeros((nbeta, nbeta, NP))
	Drecord = np.zeros((nbeta, nbeta, NP))
	R2totalrecord = np.zeros(NP)
	for nperson in range(NP):
		Sinput_original = SAPMresults_load[nperson]['Sinput_original']
		Sinput = SAPMresults_load[nperson]['Sinput']
		Sconn = SAPMresults_load[nperson]['Sconn']
		Minput = SAPMresults_load[nperson]['Minput']
		Mconn = SAPMresults_load[nperson]['Mconn']
		Mintrinsic = SAPMresults_load[nperson]['Mintrinsic']
		beta_int1 = SAPMresults_load[nperson]['beta_int1']
		R2total = SAPMresults_load[nperson]['R2total']
		Meigv = SAPMresults_load[nperson]['Meigv']
		betavals = SAPMresults_load[nperson]['betavals']

		nruns = nruns_per_person[nperson]
		if fintrinsic_count > 0:
			fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])

		fit = Minput @ Sconn

		nr, tsize_total = np.shape(Sinput_original)
		tsize = (tsize_total / nruns).astype(int)
		nbeta, tsize2 = np.shape(Sconn)

		if nperson == 0:
			Sinput_total = np.zeros((nr, tsize, NP))
			Sconn_total = np.zeros((nbeta, tsize, NP))
			fit_total = np.zeros((nr, tsize, NP))
			Mintrinsic_total = np.zeros((Nintrinsic, tsize, NP))

		tc = Sinput_original
		tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
		Sinput_total[:, :, nperson] = tc1

		tc = Sconn
		tc1 = np.mean(np.reshape(tc, (nbeta, nruns, tsize)), axis=1)
		Sconn_total[:, :, nperson] = tc1

		tc = fit
		tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
		fit_total[:, :, nperson] = tc1

		tc = Mintrinsic
		tc1 = np.mean(np.reshape(tc, (Nintrinsic, nruns, tsize)), axis=1)
		Mintrinsic_total[:, :, nperson] = tc1

		DBrecord[:, :, nperson] = Mconn
		Drecord[:ncon, :, nperson] = Minput
		Brecord[:ncon, :, nperson] = Mconn[:ncon, :] / (Minput + 1.0e-3)
		# Brecord[ktarget,ksource,nperson] = Mconn[ktarget,ksource]
		R2totalrecord[nperson] = R2total

	return paradigm_centered, Sinput_total, Sconn_total, fit_total, Mintrinsic_total, rnamelist



def plot_timecourse_data(title, windownum, tc1, tc1p, tc2 = [], tc2p = [], layout = [], yrange = [], mark_times = [], TR = 0., sigflag = []):

	if layout == 'overlap':
		plt.close(windownum)
		fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=100, num=windownum)

		tsize = len(tc1)
		x = list(range(tsize))
		xx = x + x[::-1]

		y1 = list(tc1 + tc1p)
		y2 = list(tc1 - tc1p)
		yy = y1 + y2[::-1]
		ymax = np.max(yy)
		ax.plot(x, tc1, '-or', linewidth=1, markersize=4)
		ax.fill(xx, yy, facecolor=(1, 0, 0), edgecolor='None', alpha=0.2)
		ax.set_title(title)

		if len(tc2) == tsize:
			y1 = list(tc2 + tc2p)
			y2 = list(tc2 - tc2p)
			yy = y1 + y2[::-1]
			ymax2 = np.max(yy)
			ymax = np.max([ymax,ymax2])
			ax.plot(x, tc2, '-ob', linewidth=1, markersize=4)
			ax.fill(xx, yy, facecolor=(0, 0, 1), edgecolor='None', alpha=0.2)

		if len(yrange) == 2:
			plt.ylim(yrange[0],yrange[1])

		if len(sigflag) == tsize:
			for ss in range(tsize):
				if sigflag[ss]:
					ax.plot([x[ss],x[ss]], [ymax, ymax],'X',markersize = 5, color = [0,0,0])


		if (len(mark_times) > 0) & (TR > 0.0):
			bottom, top = plt.ylim()
			nr,nt = np.shape(mark_times)
			# one zone
			t1 = mark_times[0,0]/TR - 0.5
			t2 = mark_times[0,1]/TR - 0.5
			xxf = [t1,t2,t2,t1]
			yyf = [bottom, bottom, top, top]
			ax.fill(xxf, yyf, facecolor=(1, 1, 0), edgecolor='None', alpha=0.2)

			if nr > 1:
				# second zone
				t1 = mark_times[1,0]/TR - 0.5
				t2 = mark_times[1,1]/TR - 0.5
				xxf = [t1,t2,t2,t1]
				yyf = [bottom, bottom, top, top]
				ax.fill(xxf, yyf, facecolor=(0, 1, 1), edgecolor='None', alpha=0.2)

	else:
		plt.close(windownum)
		fig, ax = plt.subplots(1, 2, figsize=(12, 9), sharey = True, dpi=100, num=windownum)

		tsize = len(tc1)
		x = list(range(tsize))
		xx = x + x[::-1]

		y1 = list(tc1 + tc1p)
		y2 = list(tc1 - tc1p)
		yy = y1 + y2[::-1]
		ymax = np.max(yy)
		ax[0].plot(x, tc1, '-or', linewidth=1, markersize=4)
		ax[0].fill(xx, yy, facecolor=(1, 0, 0), edgecolor='None', alpha=0.2)
		ax[0].set_title(title)

		if len(yrange) == 2:
			plt.ylim(yrange[0],yrange[1])


		if len(tc2) == tsize:
			y1 = list(tc2 + tc2p)
			y2 = list(tc2 - tc2p)
			yy = y1 + y2[::-1]
			ymax2 = np.max(yy)
			ymax = np.max([ymax,ymax2])
			ax[1].plot(x, tc2, '-ob', linewidth=1, markersize=4)
			ax[1].fill(xx, yy, facecolor=(0, 0, 1), edgecolor='None', alpha=0.2)

			if len(yrange) == 2:
				plt.ylim(yrange[0], yrange[1])


		if len(sigflag) == tsize:
			for ss in range(tsize):
				if sigflag[ss]:
					ax[0].plot([x[ss],x[ss]] ,[ymax, ymax],'X',markersize = 5, color = [0,0,0])
					ax[1].plot([x[ss],x[ss]], [ymax, ymax],'X',markersize = 5, color = [0,0,0])


		if (len(mark_times) > 0) & (TR > 0.0):
			bottom, top = plt.ylim()
			nr, nt = np.shape(mark_times)
			# one zone
			t1 = mark_times[0, 0] / TR - 0.5
			t2 = mark_times[0, 1] / TR - 0.5
			xxf = [t1, t2, t2, t1]
			yyf = [bottom, bottom, top, top]
			ax[0].fill(xxf, yyf, facecolor=(1, 1, 0), edgecolor='None', alpha=0.2)
			ax[1].fill(xxf, yyf, facecolor=(1, 1, 0), edgecolor='None', alpha=0.2)

			if nr > 1:
				# second zone
				t1 = mark_times[1, 0] / TR - 0.5
				t2 = mark_times[1, 1] / TR - 0.5
				xxf = [t1, t2, t2, t1]
				yyf = [bottom, bottom, top, top]
				ax[0].fill(xxf, yyf, facecolor=(0, 1, 1), edgecolor='None', alpha=0.2)
				ax[1].fill(xxf, yyf, facecolor=(0, 1, 1), edgecolor='None', alpha=0.2)



def show_one_set_of_results():
	fname = r'E:\FM2021data\FMstim_2341313432_results_corr.npy'
	# fname = r'E:\FM2021data\FMrest_2341313432_results_corr.npy'
	paramsname = r'E:\FM2021data\FMstim_2341313432_params.npy'
	covname = r'E:\FM2021data\FMstim_2341313432_results_covariates.npy'

	source = 'LC'
	target = 'Thalamus'

	results = np.load(fname, allow_pickle=True)
	params = np.load(paramsname, allow_pickle=True).flat[0]
	covs = np.load(covname, allow_pickle=True).flat[0]

	cc = covs['GRPcharacteristicslist'].index('firstpainrating')
	covvals = covs['GRPcharacteristicsvalues'][cc,:]

	r1 = params['rnamelist'].index(source)
	r2 = params['rnamelist'].index(target)
	name = '{}_{}'.format(r1,r2)
	check = [params['beta_list'][xx]['name'] == name for xx in range(len(params['beta_list']))]
	cnum = np.where(check)[0][0]

	NP = len(results)
	DB = np.array([results[xx]['Mconn'][r2,r1] for xx in range(NP)])

	np.corrcoef(DB,covvals)

	G = np.concatenate((covvals[:,np.newaxis], np.ones((NP,1))),axis=1)
	m = np.linalg.inv(G.T @ G) @ G.T @ DB

	mincov = np.min(covvals)
	maxcov = np.max(covvals)
	f1 = (np.array([mincov,1])[:,np.newaxis]).T @ m
	f1 = m[0]*mincov + m[1]
	f2 = m[0]*maxcov + m[1]

	windownum = 79
	plt.close(windownum)
	fig = plt.figure(windownum)
	plt.plot(covvals,DB,'ob')
	plt.plot([mincov,maxcov],[f1,f2],'-k')


def compare_two_data_sets():

	fname1 = r'E:\FM2021data\FMstim_2341313432_params.npy'
	fname2 = r'E:\FM2021data\FMstim_2341313432_params.npy'

	params1 = np.load(fname1, allow_pickle=True).flat[0]
	params2 = np.load(fname2, allow_pickle=True).flat[0]

	data1 = params1['tcdata_centered']
	data2 = params2['tcdata_centered']
