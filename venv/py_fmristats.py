# GRF corrected p-threshold
import numpy as np
import scipy.stats as stats
import scipy
import math

def create_test_data():
    ts = 30
    test = np.zeros((10,10,10,ts))
    for tt in range(ts):
        test[3:5,3:5,3:5,tt] = 1.0 + 0.1*math.cos(tt*np.pi/5)
    test = test + 0.1*np.random.rand(10,10,10,ts)
    return test


def py_GRFcorrected_pthreshold(p_corr, residual_data, search_mask, df=0):
     # return [p_unc, FWHM, R]

    # input residual data after doing GLM fitting
    e = residual_data
    ts = np.shape(e)[3]
    if df == 0:
        df = ts-1

    # convert the residual map to normalized residuals (Z-scores)
    sigmap = np.std(e,axis = 3)
    mean_e = np.mean(e, axis = 3)
    e2 = e
    for tt in range(ts):
        e2[:,:,:,tt] = (e2[:,:,:,tt]-mean_e)/(sigmap+1.0e-20)

    [resel_xyz, resel_img, RPV, RESEL, FWHM, R] = py_smoothness_est(e2, search_mask)
    p_unc = py_uc_RF(p_corr/2,df,R[3])   # divide by 2 to make it 2-sided T-distribution

    return p_unc, FWHM, R


def py_Bonferonni_corrected_pthreshold(p_corr, search_mask, voxel_volume = 1.0):
    # convert the residual map to normalized residuals (Z-scores)
    cx,cy,cz = np.where(search_mask > 0)
    nvox = len(cx)/voxel_volume
    print('py_Bonferonni_corrected_pthreshold:  estimating for {:06.1f} voxels'.format(nvox))
    p_unc = (p_corr/2)/nvox   # divide by 2 to make it 2-sided T-distribution

    return p_unc, nvox


def py_smoothness_est(e, mask):
    # return [resel_xyz, resel_img, RPV, RESEL, FWHM, R]
    # smoothness_est function
    xs,ys,zs,ts = np.shape(e)
    ey,ex,ez,et = np.gradient(e)

    Lxx = np.sum(ex*ex,axis = 3)/ts
    Lxy = np.sum(ex*ey,axis = 3)/ts
    Lxz = np.sum(ex*ez,axis = 3)/ts
    Lyy = np.sum(ey*ey,axis = 3)/ts
    Lyz = np.sum(ey*ez,axis = 3)/ts
    Lzz = np.sum(ez*ez,axis = 3)/ts
    Lyx = Lxy
    Lzx = Lxz
    Lzy = Lyz

    resel_img = Lxx*Lyy*Lzz + 2*Lxy*Lyz*Lxz - Lxx*Lyz*Lyz - Lxy*Lxy*Lzz - Lxz*Lxz*Lyy
    resel_img = np.where(resel_img < 0,0,resel_img)
    resel_img = np.sqrt(resel_img/(4*math.log(2))**3)

    # resel_img = QU_spatial_smooth(resel_img, fwhm_img));
    RPV = resel_img
    c = np.where(mask>0)
    resel_img = np.mean(resel_img[c])

    resel_x = np.sqrt(Lxx/(4*math.log(2)))
    resel_y = np.sqrt(Lyy/(4*math.log(2)))
    resel_z = np.sqrt(Lzz/(4*math.log(2)))
    resel_xyz = np.array([np.mean(resel_x), np.mean(resel_y), np.mean(resel_z)])

    RESEL = (resel_img**(1/3))*resel_xyz/((resel_xyz.prod())**(1/3))
    FWHM = (1./RESEL)

    R = py_sample_volume(mask, FWHM)  # takes into account the shape of the search volume

    return resel_xyz, resel_img, RPV, RESEL, FWHM, R


def py_sample_volume(mask, FWHM):
    # return R
    mask = np.where(np.abs(mask)>0,1,0)
    xs,ys,zs = np.shape(mask)
    c = np.where(mask > 0)
    P = np.size(c[0][:])

    xc = np.sum(mask,axis = 0);  c = np.where(xc > 0);  Ex = np.sum(xc[c]-1)
    yc = np.sum(mask,axis = 1);  c = np.where(yc > 0);  Ey = np.sum(yc[c]-1)
    zc = np.sum(mask,axis = 2);  c = np.where(zc > 0);  Ez = np.sum(zc[c]-1)

    # faces
    ss = np.copy(mask)
    ss[:xs-1,:,:] = ss[:xs-1,:,:] + ss[1:,:,:]
    ss[:,:ys-1,:] = ss[:,:ys-1,:] + ss[:,1:,:]
    Fxy = np.size(np.where(ss == 4)[0][:])

    ss = np.copy(mask)
    ss[:xs-1,:,:] = ss[:xs-1,:,:] + ss[1:,:,:]
    ss[:,:,:zs-1] = ss[:,:,:zs-1] + ss[:,:,1:]
    Fxz = np.size(np.where(ss == 4)[0][:])

    ss = np.copy(mask)
    ss[:,:ys-1,:] = ss[:,:ys-1,:] + ss[:,1:,:]
    ss[:,:,:zs-1] = ss[:,:,:zs-1] + ss[:,:,1:]
    Fyz = np.size(np.where(ss == 4)[0][:])

    # cubes
    ss = np.copy(mask)
    ss[:xs-1,:,:] = ss[:xs-1,:,:] + ss[1:,:,:]
    ss[:,:ys-1,:] = ss[:,:ys-1,:] + ss[:,1:,:]
    ss[:,:,:zs-1] = ss[:,:,:zs-1] + ss[:,:,1:]
    C = np.size(np.where(ss == 8)[0][:])

    rx = 1/FWHM[0];  ry = 1/FWHM[1];  rz = 1/FWHM[2]
    R = np.zeros(4)
    R[0] = P - (Ex + Ey + Ez) + (Fxy + Fxz + Fyz) - C
    R[1] = (Ex - Fxy - Fxz + C)*rx + (Ey - Fxy - Fyz + C)*ry + (Ez - Fxz - Fyz + C)*rz
    R[2] = (Fxy-C)*rx*ry + (Fxz-C)*rx*rz + (Fyz-C)*ry*rz
    R[3] = C*rx*ry*rz

    return R



def py_uc_RF(a,df,R):
    # return u

    #-Find approximate value
    p = a/np.max(R)
    u = stats.t.ppf(1 - p/2, df)

    #-Approximate estimate using E{m}
    d = 1
    du = 1e-6
    while np.abs(d) > 1e-6:
        # output of py_P_RF is P,p,Ec,Ek
        P,P,p,P = py_P_RF(1,0,u,df,R)
        P,P,q,P = py_P_RF(1,0,u+du,df,R)
        d = (a - p)/((q - p)/du)
        u = u + d
        if not np.isfinite(u):
            u=np.inf

    #-Refined estimate using 1 - exp(-E{m})
    d  = 1
    while np.abs(d) > 1e-6:
        p,P,P,P = py_P_RF(1,0,u,df,R)
        q,P,P,P = py_P_RF(1,0,u+du,df,R)
        d = (a - p)/((q - p)/du)
        u  = u + d
        if not np.isfinite(u):
            u=np.inf

    # u = QU_p_val(u,df)
    u = stats.t.sf(np.abs(u), df) * 2

    return u




def py_P_RF(c,k,T,df,R):
    # return [P,p,Ec,Ek]
    # get EC densities
    #--------------------------------------------------------------------------
    # print('R = ',R)
    D = np.where(R>0)[0][-1]   # the last occurence of where R is > 0
    # print('D = ',D)
    if np.size(R) > 1:
        R = R[:(D+1)]
    # print('R = ',R)
    G = [np.sqrt(np.pi)/math.gamma((dd+1)/2) for dd in range(D+1)]
    # print('line 166: G = ',G)
    if np.size(R) == 1:
        G =  np.sqrt(np.pi)/math.gamma(2)
    # print('line 172: G = ',G)
    # print('line 174: T = ',T)
    EC = py_ECdensity(T,df)
    # print('EC = ',EC)
    EC = np.where(np.isfinite(EC),EC,0)
    # print('EC = ',EC)
    EC  = np.max(np.append(EC[:D+1],1.0e-20))
    # print('EC = ',EC)

    # corrected p value
    # ECGprod = EC'.*G ?
    ECGprod = np.dot(EC,G)
    # print('size of EC = ',np.shape(EC))
    # print('size of G = ',np.shape(G))
    # print('ECGprod = ',ECGprod)
    P = np.triu(scipy.linalg.toeplitz(ECGprod))
    P = P[0,:]
    EM = (R/G)*P        # <maxima> over D dimensions
    Ec = np.sum(EM)          # <maxima>
    # print('size of R is ',np.shape(R))
    # print('size of R is ',np.size(R))
    # print('D = ',D)
    if np.size(R) == 1:
        EN = P[0]*R
    else:
        EN = P[0]*R[D]        # <resels>
    Ek = EN/EM[D]         # Ek = EN/EM(D);

    #-Get P{n > k}
    # assume a Gaussian form for P{n > k} ~ exp(-beta*k^(2/D))
    # Appropriate for high d.f. SPM{T}
    D = D - 1
    if np.size(R) == 1:
        D = 3
    beta = (math.gamma(D/2 + 1)/Ek)**(2/D)
    p = math.exp(-beta*(k**(2/D)))

    #-Poisson clumping heuristic {for multiple clusters}
    # print('py_P_RF: c = ',c)
    # print('py_P_RF: Ec = ',Ec)
    # print('py_P_RF: p = ',p)
    P = 1 - stats.poisson.cdf(c - 1,(Ec + 1.0e-20)*p)

    return P,p,Ec,Ek


# QU_ECdensity
def py_ECdensity(t,df):
    # return EC
    # t = t(:)'

    a = 4*math.log(2)
    b = math.exp(scipy.special.gammaln((df+1)/2) - scipy.special.gammaln(df/2))
    c = (1+(t**2)/df)**((1-df)/2)

    EC = np.zeros((4,np.size(t)))
    EC[0,:] = 1 - stats.t.cdf(t,df)
    EC[1,:] = (a**(1/2))/(2*np.pi)*c
    EC[2,:] = a/((2*np.pi)**(3/2))*c*t/((df/2)**(1/2))*b
    EC[3,:] = a**(3/2)/((2*np.pi)**2)*c*((df-1)*(t**2)/df - 1)

    return EC



# # QU_p_val
# # p = (1-tcdf(T,df));
# import scipy.stats as stats
# pval = stats.t.sf(np.abs(T), df)*2  # two-sided pvalue = Prob(abs(t)>tt)


# QU_T_val
# T = stats.t.ppf(1-p/2, df)    # p/2 to make it two-tailed t-value