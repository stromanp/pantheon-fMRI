
import image_operations_3d as i3d

fname = r'D:\Howie_FM2_Brain_Data\JUN27_2019C\Series12\cSeries12.nii'
input_data, new_affine = i3d.load_and_scale_nifti(fname)

windownum = 102
plt.close(windownum)
fig = plt.figure(windownum)
plt.plot(range(135),tc,'-xr')

xs,ys,zs,ts = np.shape(input_data)
t1 = 4
t2 = 20
TR = 2.0
tt = np.array(range(t1,t2))
result_img = np.zeros((xs,ys,zs))
for zz in range(zs):
	print('slice {} of {}'.format(zz+1,zs))
	for xx in range(xs):
		for yy in range(ys):
			tc = input_data[xx, yy, zz, :]
			vals = copy.deepcopy(tc[t1:t2])
			a = np.polyfit(tt,vals,1)
			result_img[xx,yy,zz] = a[0]



fig = plt.figure(windownum+1)
plt.imshow(result_img[:,:,37])

plt.close(windownum + 2)
fig = plt.figure(windownum+2)
plt.imshow(result_img[92,:,:])



plt.close(windownum + 3)
fig = plt.figure(windownum+3)
plt.imshow(result_img[:,62,:])