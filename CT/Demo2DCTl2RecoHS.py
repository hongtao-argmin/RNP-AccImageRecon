# Tao edit
# May/2024
# 14th Jun/2024. Tao 
# Tao Hong, Zhaoyi Xu, Jason Hu, and Jeffrey A. Fessler, 
# ``Using Randomized Nyström Preconditioners to Accelerate Variational Image Reconstruction'', To appear in IEEE Transactions on Computational Imaging, arXiv:2411.08178, 2025.

import CTutilities as CTutl
import utilities as utl
import utilitiesHS as utlHS
import torch
import scipy.io
import numpy as np
import time
import json
import os
import h5py

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-cuda_index', dest='cuda_index', help='Select the index of GPU',type=str,default = '1')
parser.add_argument('-MaxIter', dest='MaxIter',help='Maximal number of iterations', type=int,default = 150)
parser.add_argument('-MaxIter_Test', dest='MaxIter_Test', help='Maximal number of iterations for choosing the trade-off parameter',type=int,default = 20)
parser.add_argument('-noise_level', dest='noise_level',help='The squart root of the standard variance of adding Gaussian noise',type=float,default = 0)
# search a trade-off parameter
parser.add_argument('-TROffBeg', dest='TROffBeg', help='The first value of the trade-off parameter',type=float,default = 5e-6)
parser.add_argument('-TROffEnd', dest='TROffEnd', help='The last value of the trade-off parameter',type=float,default = 1e-3)
parser.add_argument('-TROffNum', dest='TROffNum',help='The number of tested trade-off parameter between the first and last tested value', type=int,default = 200)
parser.add_argument('-sketch_size', dest='sketch_size', help='The size of sketch for building preconditioner: large-> better but more computational time',type=int,default = 200)
parser.add_argument('-mu', dest='mu', help='Parameter in sketch to avoid zero singular value',type=float,default = 1e-5)
parser.add_argument('-verbose', dest='verbose', type=bool,help='Show the running procudure of the algorithm',default = True)
parser.add_argument('-isSave', dest='isSave',type=bool, help='True: saved the reconstructed image at each iteration',default = False)
parser.add_argument('-iSmkdir', dest='iSmkdir',type=bool,help='True: create a folder for saving obtained results', default = True)
parser.add_argument('-isPlot', dest='isPlot', type=bool,help='True: plot the results',default = False)
parser.add_argument('-imPathFolder', dest='imPathFolder', type=str,help='Folder for storing the testing images',default = './CTData/')
parser.add_argument('-imFile', dest='imFile', type=str,help='Name of the tested data file',default = 'TestData.mat')
parser.add_argument('-imSavedPathFolder', dest='imSavedPathFolder', type=str,help='Folder to save the results, need to create before running',default = '/YourOwnLocalPath/')

parser.add_argument('-beta',dest='beta',type=float,help='If beta is not Inf then we use the given beta',default=float('inf'))
parser.add_argument('-views', dest='views',help='The number of views', type=int,default = 100)
parser.add_argument('-detectors_size', dest='detectors_size',help='The size of the detector', type=int,default = 1024) #512
parser.add_argument('-Inner_iter', dest='Inner_iter',help='Maximal number of inner iterations for TV', type=int,default = 200)
parser.add_argument('-im_index', dest='im_index',help='Slice index', type=int,default = 10)
parser.add_argument('-p', dest='p',help='p norm', type=float, default = 1e6)# 1,2,float('inf')
parser.add_argument('-CT_type', dest='CT_type',help='Type of CT geometry, parbeam or fanbeam', type=str,default = 'fanbeam')#'parbeam'

args = parser.parse_args()

views = args.views
detectors_size = args.detectors_size
sketch_size = args.sketch_size
Inner_iter = args.Inner_iter
MaxIter = args.MaxIter
MaxIter_Test = args.MaxIter_Test
p = args.p
if p>=1e6:
    p = float('inf')
cuda_index = args.cuda_index
im_index = args.im_index
iSmkdir = args.iSmkdir
imName_only = os.path.splitext(args.imFile)[0]
device = torch.device('cuda:'+cuda_index)

im_set = h5py.File(args.imPathFolder+args.imFile, 'r')
im_set = im_set['TestData']
im = im_set[:,:,0,im_index]
im = np.transpose(im, (1, 0))
im_size = im.shape[0]
im_size_prod = im_size*im_size
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

torch.cuda.manual_seed(1) # for GPU
output_noise = output+args.noise_level*torch.randn_like(output)#*torch.norm(output)
snr = 10*torch.log10(torch.norm(output)/torch.norm(output_noise-output))
print('The measurements SNR is {}'.format(snr.cpu().numpy()))


# build the sketch preconditioner
start_time = time.perf_counter()
U,S,lambda_l = CTutl.Build_Sketch_Pred(Ax,ATx,im_size,im_size_prod,sketch_size,device=device)
end_time = time.perf_counter()
sketch_time = end_time-start_time
print('The cost time of building the preconditioner {}\n'.format(sketch_time))
# build preconditioner
mu = args.mu

U_temp = U*torch.sqrt(1-(lambda_l+mu)/(S+mu))
P_inv = lambda x: CTutl.P_invx_Simp(x,U_temp,im_size)
# Test on the TV Reco. with FISTA algorithm.
L_pre = utl.Power_Iter([im_size, im_size],A=Ax_scale,AT=ATx_scale,P=P_inv,device=device)
print('The New Lip. Constant is {}\n'.format(L_pre.cpu().numpy()))
L_P_inv = utl.Power_Iter([im_size, im_size],P=P_inv,device=device)
L_P_inv = 1/L_P_inv 

b_noise = output_noise
verbose = True

# --------------------- search trade-off parameter --------------------- #
if args.beta == float('inf'):
    beta_set = np.linspace(args.TROffBeg,args.TROffEnd, num=args.TROffNum,endpoint=True)
    psnr_set = []
    ssim_set = []

    for iter in range(beta_set.shape[0]):
        _,_,psnr_set_FISTA_HS,\
        ssim_set_FISTA_HS,_ = \
    utlHS.FISTA_HS(MaxIter_Test,Ax_scale,ATx_scale,b_noise,p=p,L=L_pre,P =P_inv,L_P_inv=L_P_inv,TR_off = beta_set[iter],save=None,SaveIter=False,\
                                original=im,verbose=verbose,device=device)
        psnr_set.append(np.max(psnr_set_FISTA_HS))
        ssim_set.append(np.max(ssim_set_FISTA_HS))
        print('The maximal PSNR and SSIM are: {},{}'.format(np.max(psnr_set_FISTA_HS),np.max(ssim_set_FISTA_HS)))
    index = np.argmax(psnr_set)
    beta = beta_set[index] 
    print('The used trade-off parameters is {}\n'.format(beta))
else:
    beta = args.beta


algName = imName_only +str(im_index) + 'FISTA'+'views'+str(views)
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
x_ord,lst_cost_ord,psnr_set_FISTA_HS_ord,\
    ssim_set_FISTA_HS_ord,lst_time_ord = \
    utlHS.FISTA_HS(MaxIter,Ax_scale,ATx_scale,b_noise,1,p=p,Num_iter = Inner_iter,TR_off = beta,save=loc,SaveIter=args.isSave,\
                             original=im,verbose=verbose,device=device)


algName = imName_only +str(im_index) +'PreFISTA'+'sketchsize'+str(sketch_size)+'views'+str(views)
loc = args.imSavedPathFolder+args.CT_type+algName
if iSmkdir:    
    if not os.path.exists(loc):
        os.mkdir(loc)
L_acc = L_pre
x_pre,lst_cost_pre,psnr_set_FISTA_HS_pre,\
    ssim_set_FISTA_HS_pre,lst_time_pre = \
    utlHS.FISTA_HS(MaxIter,Ax_scale,ATx_scale,b_noise,L_acc,p=p,P=P_inv,Num_iter = Inner_iter,L_P_inv=L_P_inv,TR_off = beta,save=loc,SaveIter=args.isSave,\
                             original=im,verbose=verbose,device=device)

print('The maximal PSNR and SSIM of ord. are: {},{}\n'.format(np.max(psnr_set_FISTA_HS_ord),np.max(ssim_set_FISTA_HS_ord)))
print('The maximal PSNR and SSIM of pre. are: {},{}\n'.format(np.max(psnr_set_FISTA_HS_pre),np.max(ssim_set_FISTA_HS_pre)))
