# demo of the image deblurring/super-resolution reconstruction with impulsive noise.
# Authors: Tao Hong <tao.hong@austin.utexas.edu> and Zhaoyi Xu <zhaoyix@umich.edu>
# Date: 10/18/2025.
# Tao Hong, Zhaoyi Xu, Jason Hu, and Jeffrey A. Fessler, 
# ``Using Randomized Nyström Preconditioners to Accelerate Variational Image Reconstruction'', To appear in IEEE Transactions on Computational Imaging, arXiv:2411.08178, 2025.
# import necessary packages
import torch.nn.functional as F
import torch
import cv2
import numpy as np
import utilities as utl
import OptAlgLpLq as opt
import os
import json
import scipy.io
import argparse
# parameters to set the experiments
parser = argparse.ArgumentParser()
parser.add_argument('-cuda_index', dest='cuda_index', help='Select the index of GPU',type=str,default = '1')
parser.add_argument('-MaxIter', dest='MaxIter',help='Maximal number of iterations', type=int,default = 10)
parser.add_argument('-MaxIter_Test', dest='MaxIter_Test', help='Maximal number of iterations for choosing the trade-off parameter',type=int,default = 10)
parser.add_argument('-TROffBeg', dest='TROffBeg', help='The first value of the trade-off parameter',type=float,default = 1e-3)
parser.add_argument('-TROffEnd', dest='TROffEnd', help='The last value of the trade-off parameter',type=float,default = 1e-1)
parser.add_argument('-TROffNum', dest='TROffNum',help='The number of tested trade-off parameter between the first and last tested value', type=int,default = 100)
parser.add_argument('-beta', dest='beta', help='Trade-off parameter, If the default beta is larger than 1e6, we will search one',type=float,default = np.inf)#
parser.add_argument('-kernel_size', dest='kernel_size', help='Kernal size of the blur kernel',type=int,default = 9)
parser.add_argument('-sketch_size', dest='sketch_size', help='The size of sketch for building preconditioner: large-> better but more computational time',type=int,default = 100)
parser.add_argument('-CG_Tolerance', dest='CG_Tolerance',help='The tolerance for stopping running CG for solving the least square', type=float,default = 1e-4)
parser.add_argument('-im_task', dest='im_task', type=str,help='Image task: Deblur, SR, Denoise',default = 'SR')
parser.add_argument('-kernel_type', dest='kernel_type', type=str,help='Two options: Unoform or Gaussian',default = 'Gaussian') # 'Gaussian'
parser.add_argument('-sigma_gauss_kernel', dest='sigma_gauss_kernel', help='The variance of building the Gaussian kernel',type=float,default = 1.6)
parser.add_argument('-SR_Ratio', dest='SR_Ratio', type=int,help='Super-resolution downscale ratio',default = 2)
parser.add_argument('-isColor', dest='isColor', type=bool,help='Must set true if work on color images',default = True)
parser.add_argument('-verbose', dest='verbose', type=bool,help='Show the running procudure of the algorithm',default = True)
parser.add_argument('-isSave', dest='isSave',type=bool, help='True: saved the reconstructed image at each iteration',default = False)
parser.add_argument('-iSmkdir', dest='iSmkdir',type=bool,help='True: create a folder for saving obtained results', default = False)
parser.add_argument('-imPathFolder', dest='imPathFolder', type=str,help='Folder for storing the testing images',default = './test_images/')
parser.add_argument('-imName', dest='imName', type=str,help='Name of the tested image',default = 'bike.tif')#leaves cameraman
parser.add_argument('-imSavedPathFolder', dest='imSavedPathFolder', type=str,help='Folder to save the results, need to create before running',default = '/YourOwnLocalPath/')
parser.add_argument('-salt_prob', dest='salt_prob', help='The probability of of salt noise',type=float,default = 0.05)
parser.add_argument('-pepper_prob', dest='pepper_prob',help='The probability of of pepper noise', type=float,default = 0.05)
parser.add_argument('-p', dest='p',help='The p norm of the data-fidelity part', type=float,default = 1)
parser.add_argument('-q', dest='q',help='The q norm of the regularizer part', type=float, default = 1)
args = parser.parse_args()

# take out the parameter
cuda_index = args.cuda_index
MaxIter = args.MaxIter
MaxIter_Test = args.MaxIter_Test
salt_prob = args.salt_prob
pepper_prob = args.pepper_prob
p = args.p
q = args.q
kernel_size = args.kernel_size
channel_in = 1
channel_out = 1
sketch_size = args.sketch_size
CG_Tolerance = args.CG_Tolerance
im_task = args.im_task
kernel_type = args.kernel_type
sigma_gauss_kernel = args.sigma_gauss_kernel
SR_Ratio = args.SR_Ratio
isColor = args.isColor
verbose = args.verbose
imPathFolder = args.imPathFolder
imName = args.imName
imName_only = os.path.splitext(imName)[0]
imSavedPathFolder = args.imSavedPathFolder
isSave = args.isSave
iSmkdir = args.iSmkdir

image_path = imPathFolder+imName

device = torch.device('cuda:'+cuda_index)
im_original = cv2.imread(image_path)

if isColor:
    im_original = np.float32(cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB))/255.0
else:
    im_original = np.float32(cv2.cvtColor(im_original, cv2.COLOR_BGR2GRAY))/255.0
im = torch.from_numpy(im_original).unsqueeze(0).unsqueeze(0).to(device)
if im_task == 'SR':
    im = utl.crop_to_scale(im,SR_Ratio,isColor=isColor)
    im_original = utl.crop_to_scale_Numpy(im_original,SR_Ratio,isColor=isColor)
im_size = im_original.shape[0:2]

# build the convolutional kernel and its adjoint
if kernel_type == 'Uniform':
    filter = torch.ones(channel_in,channel_out,kernel_size,kernel_size).to(device)/kernel_size**2
elif kernel_type=='Gaussian':
    filter = utl.gaussian_kernel(kernel_size,sigma_gauss_kernel)
    filter = filter.to(device)
m = torch.nn.Conv2d(channel_in,channel_out, kernel_size, bias=None, padding=kernel_size//2, device=device)
for param in m.parameters():
    param.requires_grad = False
m.weight.data = filter.clone()
mT = torch.nn.ConvTranspose2d(channel_in,channel_out, kernel_size, bias=None, padding=kernel_size//2, device=device)
for param in mT.parameters():
    param.requires_grad = False
mT.weight.data  = filter.clone()

if im_task == 'Deblur':
    Ax = lambda x: m(x)
    ATx = lambda x: mT(x)
   
elif im_task == 'SR':
    # super-resolution forward model
    def SR_Ax(x,Ax,Ratio):
        y = Ax(x)
        y = y[:,:,::Ratio,::Ratio]
        return y
    def SR_ATx(y,ATx,Ratio,im_size,device='cpu'):
        nBatch,_,_,_ = y.size()
        x = torch.zeros(nBatch,1,im_size[0],im_size[1],device=device)
        x[:,:,::Ratio,::Ratio] = y
        x = ATx(x)
        return x
    Ax = lambda x: SR_Ax(x,m,SR_Ratio)
    ATx = lambda y: SR_ATx(y,mT,SR_Ratio,im_size,device=device)

elif im_task == 'Denoise':
    Ax = lambda x: x
    ATx = lambda x: x

# degradation procedure
if isColor:
    if im_task == 'Deblur' or im_task=='Denoise':
        b = torch.zeros_like(im)
        for num_channels in range(im.shape[4]):
            b[:,:,:,:,num_channels] = Ax(im[:,:,:,:,num_channels])
    elif im_task == 'SR':
        b_temp = torch.zeros_like(im)
        for num_channels in range(im.shape[4]):
            b_temp[:,:,:,:,num_channels] = m(im[:,:,:,:,num_channels])
        # downsample
        b = b_temp[:,:,::SR_Ratio,::SR_Ratio,:]
else:
    b = Ax(im)

# add impulse noise
b_noisy_filter,b_noisy = utl.add_salt_and_pepper_noise(b,salt_prob,pepper_prob,isColor=isColor)
b_noisy_temp = b_noisy_filter
# transfer RGB color to YCRCB
if isColor:
    b_noisy_input = utl.PrepareImage(b_noisy).to(device)
    im_ref = torch.squeeze(utl.PrepareImage(im)).numpy()
else:
    b_noisy_input = b_noisy
    im_ref = im_original

if im_task == 'Deblur' or im_task=='Denoise':
    x_ini = b_noisy_input
else:
    x_ini = F.interpolate(b_noisy_input, size=im.shape[2:4], mode='bicubic', align_corners=True)
    x_ini = x_ini[:, :, 0:im.shape[2], 0:im.shape[3]].clamp(min=0, max=1)
    if isColor:
        temp = F.interpolate(torch.permute(b_noisy,(0,1,4,2,3)).squeeze(0), size=im.shape[2:4], mode='bicubic', align_corners=False)
        b_noisy = torch.permute(temp,(0,2,3,1)).unsqueeze(0)


# trade-off parameters
# find ``best'' trade-off parameters
TV_bound ='Dirchlet'
Phix = lambda x: torch.cat((utl.GetGradSingle_Pytorch(x,TV_bound=TV_bound,Dir='x-axis',isAdjoint = False,device=device),utl.GetGradSingle_Pytorch(x,TV_bound=TV_bound,Dir='y-axis',isAdjoint = False,device=device)),0)
PhiTx = lambda x: utl.GetGradSingle_Pytorch(x[0,:,:,:].unsqueeze(0),TV_bound=TV_bound,Dir='x-axis',isAdjoint = True,device=device)+utl.GetGradSingle_Pytorch(x[1,:,:,:].unsqueeze(0),TV_bound=TV_bound,Dir='y-axis',isAdjoint = True,device=device)
PhiTxBatch = lambda x: utl.GetGradSingle_Pytorch(x[0:sketch_size,:,:,:],TV_bound=TV_bound,Dir='x-axis',isAdjoint = True,device=device)+utl.GetGradSingle_Pytorch(x[sketch_size:,:,:,:],TV_bound=TV_bound,Dir='y-axis',isAdjoint = True,device=device)
verbose = True
if args.beta>=1e6:
    beta_set = np.linspace(args.TROffBeg,args.TROffEnd, num=args.TROffNum,endpoint=True)
    psnr_set = []
    ssim_set = []

    for iter in range(beta_set.shape[0]):
        _,_,psnr_set_FISTA_TV,\
        ssim_set_FISTA_TV,_,_ = \
            opt.IRM_lplq(MaxIter_Test,Ax,ATx,Phix,PhiTx,b_noisy_input,p,q,sketch_size=sketch_size,isPre=True,PhiTxBatch=PhiTxBatch,TR_off = beta_set[iter],im_size=im_size,CG_Tolerance=CG_Tolerance,\
                x_ini=x_ini,isColor=isColor,b_noisy=b_noisy,save=None,SaveIter=False,\
                                original=im_ref,verbose=verbose,device=device)
        psnr_set.append(np.max(psnr_set_FISTA_TV))
        ssim_set.append(np.max(ssim_set_FISTA_TV))
        print('The maximal PSNR and SSIM are: {},{}'.format(np.max(psnr_set_FISTA_TV),np.max(ssim_set_FISTA_TV)))
    index = np.argmax(psnr_set)
    beta = beta_set[index] 
    print('The used trade-off parameters is {}\n'.format(beta))
else:
    beta = args.beta


algName = '/l' +str(p) + '_l' +str(q)+imName_only+'_salt'+str(salt_prob)+'pepp'+str(pepper_prob)
loc = imSavedPathFolder + im_task + kernel_type+ algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
        # Save arguments to a file
    with open(loc+'/args.json', 'w') as f:
        json.dump(vars(args), f)
    Dict = {'im_original':im_original,'beta':beta,'b': torch.squeeze(b).cpu().numpy(),\
            'b_noisy':torch.squeeze(b_noisy).cpu().numpy(),\
                'b_noisy_temp':torch.squeeze(b_noisy_temp).cpu().numpy()}
    scipy.io.savemat("%s/SavedInfo.mat" % loc,Dict)
x_ord,lst_cost_ord,psnr_set_FISTA_TV_ord,\
    ssim_set_FISTA_TV_ord,lst_time_ord,lst_iter_ord = \
        opt.IRM_lplq(MaxIter,Ax,ATx,Phix,PhiTx,b_noisy_input,p,q,TR_off = beta,im_size=im_size,CG_Tolerance=CG_Tolerance,\
             x_ini=x_ini,isColor=isColor,b_noisy=b_noisy,save=loc,SaveIter=isSave,\
                             original=im_ref,verbose=verbose,device=device)
if isColor:
    x_ord = utl.MergeChannels(b_noisy,x_ord)


algName = '/Pre_l' +str(p) + '_l' +str(q)+imName_only+'_salt'+str(salt_prob)+'pepp'+str(pepper_prob)
loc = imSavedPathFolder+im_task+ kernel_type+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_pre,lst_cost_pre,psnr_set_FISTA_TV_pre,\
    ssim_set_FISTA_TV_pre,lst_time_pre,lst_iter_pre = \
        opt.IRM_lplq(MaxIter,Ax,ATx,Phix,PhiTx,b_noisy_input,p,q,sketch_size=sketch_size,isPre=True,PhiTxBatch=PhiTxBatch,TR_off = beta,\
             im_size=im_size,CG_Tolerance=CG_Tolerance,x_ini=x_ini,isColor=isColor,b_noisy=b_noisy,save=loc,SaveIter=isSave,\
                             original=im_ref,verbose=verbose,device=device)
if isColor:
    x_pre = utl.MergeChannels(b_noisy,x_pre)