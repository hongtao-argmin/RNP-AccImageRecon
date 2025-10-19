# Tao edit
# Jun/2024. Tao
# Oct. 18/2025. Tao 
# wavelet based CT reconstruction
# solve 0.5*\|AW^{-1}z-b\|_2^2+\|z\|_1 with ADMM and FISTA
# image: x = W^{-1}z -- W denotes the used wavelet // only implement the orthogonal wavelet case
# we assume the image size is square.
# Tao Hong, Zhaoyi Xu, Jason Hu, and Jeffrey A. Fessler, 
# ``Using Randomized Nyström Preconditioners to Accelerate Variational Image Reconstruction'', To appear in IEEE Transactions on Computational Imaging, arXiv:2411.08178, 2025.

import CTutilities as CTutl
import utilities as utl
import odl
import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import h5py
import os
import ptwt, pywt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-cuda_index', dest='cuda_index', help='Select the index of GPU',type=str,default = '1')
parser.add_argument('-MaxIter', dest='MaxIter',help='Maximal number of iterations', type=int,default = 150)
parser.add_argument('-MaxIter_Test', dest='MaxIter_Test', help='Maximal number of iterations for choosing the trade-off parameter',type=int,default = 100)
parser.add_argument('-CG_Iter', dest='CG_Iter', help='Maximal number of iterations in ADMM LS CG solver',type=int,default = 300)
parser.add_argument('-rho', dest='rho', help='rho parameter in ADMM',type=float,default = 1)
parser.add_argument('-noise_level', dest='noise_level',help='The squart root of the standard variance of adding Gaussian noise',type=float,default = 8e-3)
# search a trade-off parameter
parser.add_argument('-TROffBeg', dest='TROffBeg', help='The first value of the trade-off parameter',type=float,default = 1e-8)
parser.add_argument('-TROffEnd', dest='TROffEnd', help='The last value of the trade-off parameter',type=float,default = 1e-3)
parser.add_argument('-TROffNum', dest='TROffNum',help='The number of tested trade-off parameter between the first and last tested value', type=int,default = 300)
parser.add_argument('-sketch_size', dest='sketch_size', help='The size of sketch for building preconditioner: large-> better but more computational time',type=int,default = 100)
parser.add_argument('-mu', dest='mu', help='Parameter in sketch to avoid zero singular value',type=float,default = 1e-5)
parser.add_argument('-verbose', dest='verbose', type=bool,help='Show the running procudure of the algorithm',default = True)
parser.add_argument('-isSave', dest='isSave',type=bool, help='True: saved the reconstructed image at each iteration',default = False)
parser.add_argument('-iSmkdir', dest='iSmkdir',type=bool,help='True: create a folder for saving obtained results', default = True)
parser.add_argument('-isPlot', dest='isPlot', type=bool,help='True: plot the results',default = False)
parser.add_argument('-imPathFolder', dest='imPathFolder', type=str,help='Folder for storing the testing images',default = './CTData/')
parser.add_argument('-imFile', dest='imFile', type=str,help='Name of the tested data file',default = 'TestData.mat')
parser.add_argument('-imSavedPathFolder', dest='imSavedPathFolder', type=str,help='Folder to save the results, need to create before running',default = '/YourOwnLocalPath/')
parser.add_argument('-views', dest='views',help='The number of views', type=int,default = 100)
parser.add_argument('-detectors_size', dest='detectors_size',help='The size of the detector', type=int,default = 1024)
parser.add_argument('-im_index', dest='im_index',help='Slice index', type=int,default = 10)
parser.add_argument('-CT_type', dest='CT_type',help='Type of CT geometry, parbeam or fanbeam', type=str,default = 'fanbeam')#'parbeam'
parser.add_argument('-WavType', dest='WavType',type=str,help='The wavelet type',default = 'db4')#haar
parser.add_argument('-num_level', dest='num_level', type=int, help='The number of levels of the wavelet',default = 4)
parser.add_argument('-wave_bnd', dest='wave_bnd', type=str,help='The boundary condition of the wavelet', default ='zero')
args = parser.parse_args()
views = args.views
detectors_size = args.detectors_size
sketch_size = args.sketch_size
MaxIter = args.MaxIter
MaxIter_Test = args.MaxIter_Test
cuda_index = args.cuda_index
im_index = args.im_index
iSmkdir = args.iSmkdir
imName_only = os.path.splitext(args.imFile)[0]
im_set = h5py.File(args.imPathFolder+args.imFile, 'r')
im_set = im_set['TestData']
im = im_set[:,:,0,im_index]
im = np.transpose(im, (1, 0))
im_size = im.shape[0]
im_size_prod = im_size*im_size
device = torch.device('cuda:'+cuda_index)
im_original = torch.from_numpy(im).squeeze().unsqueeze(0).unsqueeze(0).to(device).to(torch.float32)

if args.CT_type=='parbeam':
    CT_Modul = CTutl.parbeam([im_size, im_size], views, detectors_size) # image size, number of views, number of sensor detectors
elif args.CT_type =='fanbeam':
    CT_Modul = CTutl.fanbeam([im_size, im_size], views, detectors_size) # image size, number of views, number of sensor detectors

Ax = lambda x: CT_Modul.Atimes(x)
ATx = lambda x: CT_Modul.ATtimes(x)

# get the Lipschitz constant
L_sch = utl.Power_Iter([im_size, im_size],A=Ax,AT=ATx,device=device)
print('The Lip. constant is: {}\n'.format(L_sch))
L_numpy = np.sqrt(L_sch.cpu().numpy())

Ax_scale = lambda x: CT_Modul.Atimes(x)/L_numpy
ATx_scale = lambda x: CT_Modul.ATtimes(x)/L_numpy

output = Ax_scale(im_original)
#torch.manual_seed(1) # for cpu
torch.cuda.manual_seed(1) # for GPU
output_noise = output+args.noise_level*torch.randn_like(output)#*torch.norm(output)
snr = 10*torch.log10(torch.norm(output)/torch.norm(output_noise-output))
print('The measurements SNR is {}'.format(snr.cpu().numpy()))

# for wavelet -- only implement for orthogonal wavelets.
#‘constant’, ‘zero’, ‘reflect’, ‘periodic’, ‘symmetric’
wavelet_type = pywt.Wavelet(args.WavType)
num_level = args.num_level
wave_bnd = args.wave_bnd

## only use orthogonal wavelet
Phix = lambda x: ptwt.wavedec2(x,wavelet_type,level=num_level, mode=wave_bnd)
PhiTx = lambda x: ptwt.waverec2(x, wavelet_type)
PhiTxBatch = lambda x: ptwt.waverec2(x, wavelet_type)

list_coeff_size = []
im_coeff_size = 0
temp_coeff = Phix(im_original)
list_coeff_size.append(temp_coeff[0].size(2))
im_coeff_size += list_coeff_size[0]
for iter in range(num_level):
    list_coeff_size.append(temp_coeff[iter+1][0].size(2))
    im_coeff_size += list_coeff_size[iter+1]

im_coeff_size_prod = im_coeff_size*im_coeff_size
start_time = time.perf_counter()
U,S,lambda_l = CTutl.Build_Sketch_Wav_Pred(Ax,ATx,Phix,PhiTxBatch,im_coeff_size,list_coeff_size,im_coeff_size_prod,sketch_size,num_level,device=device)
end_time = time.perf_counter()
sketch_time = end_time-start_time
print('The cost time of building the preconditioner {}\n'.format(sketch_time))
# build preconditioner
mu = args.mu
U_temp_inv = U*torch.sqrt(1-(lambda_l+mu)/(S+mu))
P_inv = lambda x: CTutl.P_invx_Simp(x,U_temp_inv,im_coeff_size)#lambda x: CTutl.P_invx(x,Lambda_inv,U,lambda_l+mu,im_size)

# Test on the TV Reco. with FISTA algorithm.
L_pre = utl.PowerIterWav([im_coeff_size,im_coeff_size],num_level,list_coeff_size,A=Ax_scale,AT=ATx_scale,Wx=Phix,WTx=PhiTx,P=P_inv,device=device)
print('The New Lip. Constant is {}\n'.format(L_pre.cpu().numpy()))
L_P_inv = utl.Power_Iter([im_coeff_size, im_coeff_size],P=P_inv,device=device)
L_P_inv = 1/L_P_inv # minimal eigenvalue of the preconditioner
U_temp = U*torch.sqrt((S+mu)/(lambda_l+mu)-1)

L = utl.PowerIterWav([im_coeff_size,im_coeff_size],num_level,list_coeff_size,A=Ax_scale,AT=ATx_scale,Wx=Phix,WTx=PhiTx,device=device)
b_noise = output_noise
verbose = True


beta_set = np.linspace(args.TROffBeg,args.TROffEnd, num=args.TROffNum,endpoint=True)
psnr_set = []
ssim_set = []

for iter in range(beta_set.shape[0]):
    _,_,psnr_set_FISTA_Wav,\
    ssim_set_FISTA_Wav,_ = \
utl.FISTA_Wav(MaxIter_Test,Ax_scale,ATx_scale,b_noise,L=L,Wx=Phix,WTx=PhiTx,im_size=im_coeff_size,num_level=num_level,list_coeff_size=list_coeff_size,TR_off = beta_set[iter],save=None,SaveIter=False,\
                             original=im,verbose=verbose,device=device)
    psnr_set.append(np.max(psnr_set_FISTA_Wav))
    ssim_set.append(np.max(ssim_set_FISTA_Wav))
    print('The maximal PSNR and SSIM are: {},{}'.format(np.max(psnr_set_FISTA_Wav),np.max(ssim_set_FISTA_Wav)))
index = np.argmax(psnr_set)
beta = beta_set[index] 
print('The used trade-off parameters is {}\n'.format(beta))


algName = imName_only +str(im_index)+ 'FISTA'+'views'+str(views)
loc = args.imSavedPathFolder +args.CT_type+algName
if iSmkdir:    
    if not os.path.exists(loc):
        os.mkdir(loc)
    # Save arguments to a file
    with open(loc+'/args.json', 'w') as f:
        json.dump(vars(args), f)
    Dict = {'im_GT':im,'beta':beta,'sketch_time':sketch_time,'Measurements':torch.squeeze(output).cpu().numpy(),\
            'NoisyMeasurements':torch.squeeze(output_noise).cpu().numpy()}
    scipy.io.savemat("%s/CTImRecoInfo.mat" % loc, Dict)
x_ord,lst_cost_ord,psnr_set_FISTA_TV_ord,\
    ssim_set_FISTA_TV_ord,lst_time_ord = \
utl.FISTA_Wav(MaxIter,Ax_scale,ATx_scale,b_noise,L=L,Wx=Phix,WTx=PhiTx,im_size=im_coeff_size,num_level=num_level,list_coeff_size=list_coeff_size,TR_off = beta,save=loc,SaveIter=args.isSave,\
                             original=im,verbose=verbose,device=device)


algName = imName_only + str(im_index)+'PreFISTA'+'sketchsize'+str(sketch_size)+'views'+str(views)
loc = args.imSavedPathFolder+args.CT_type+algName
if iSmkdir:    
    if not os.path.exists(loc):
        os.mkdir(loc)
L_acc = L_pre
x_pre,lst_cost_pre,psnr_set_FISTA_TV_pre,\
    ssim_set_FISTA_TV_pre,lst_time_pre = \
utl.FISTA_Wav(MaxIter,Ax_scale,ATx_scale,b_noise,L_acc,Wx=Phix,WTx=PhiTx,im_size=im_coeff_size,num_level=num_level,list_coeff_size=list_coeff_size,U=U_temp,isPre=True,P=P_inv,L_P_inv=L_P_inv,TR_off = beta,save=loc,SaveIter=args.isSave,\
                             original=im,verbose=verbose,device=device)
