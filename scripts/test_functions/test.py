

for xx in range(len(cluster_properties)):
	IDX = cluster_properties[xx]['IDX']
	print('region {} {}:'.format(xx,cluster_properties[xx]['rname']))
	for cc in range(5):
		check = np.where(IDX == cc)[0]
		nvox = len(check)
		print('          cluster {} has {} voxels'.format(cc,nvox))
