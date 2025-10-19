# demo for walnut reconstruction. 
# Tao Hong, Zhaoyi Xu, Jason Hu, and Jeffrey A. Fessler, ``Using Randomized Nyström Preconditioners to Accelerate Variational Image Reconstruction'', To appear in IEEE Transactions on Computational Imaging, arXiv:2411.08178, 2025.
# if you use the walnut, please also cite the walnut paper:
# K. H¨am¨al¨ainen, L. Harhanen, A. Kallonen, A. Kujanp¨a¨a, E. Niemi, and S. Siltanen, “Tomographic X-ray data of a walnut,” arXiv preprint arXiv:1502.04064, 2015.
import CTutilities as CTutl
import utilities as utl
import torch
import scipy.io
import numpy as np
import time
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-cuda_index', dest='cuda_index', help='Select the index of GPU',type=str,default = '1')
parser.add_argument('-MaxIter', dest='MaxIter',help='Maximal number of iterations', type=int,default = 150)
parser.add_argument('-beta', dest='beta', help='The trade-off parameter, if it is very large then we search it, otherwise we just take it.',type=float,default=1e-4)#float('inf'))
parser.add_argument('-MaxCG_Iter', dest='MaxCG_Iter',help='The maximal number of CG iterations for solving the LS in ADMM.', type=int,default = 80)
parser.add_argument('-CG_Tolerance', dest='CG_Tolerance',help='TThe tolerance for solving least square with CG methods',type=float,default = 1e-5)
parser.add_argument('-sketch_size', dest='sketch_size', help='The size of sketch for building preconditioner: large-> better but more computational time',type=int,default = 100)
parser.add_argument('-mu', dest='mu', help='Parameter in sketch to avoid zero singular value',type=float,default = 1e-5)
parser.add_argument('-verbose', dest='verbose', type=bool,help='Show the running procudure of the algorithm',default = True)
parser.add_argument('-isSave', dest='isSave',type=bool, help='True: saved the reconstructed image at each iteration',default = True)
parser.add_argument('-iSmkdir', dest='iSmkdir',type=bool,help='True: create a folder for saving obtained results', default = True)
parser.add_argument('-imFile', dest='imFile', type=str,help='Name of the tested data file',default = 'TestData.mat')
parser.add_argument('-imSavedPathFolder', dest='imSavedPathFolder', type=str,help='Folder to save the results, need to create before running',default = '/YourOwnLocalPath/')
parser.add_argument('-Inner_iter', dest='Inner_iter',help='Maximal number of inner iterations for TV', type=int,default = 80)
parser.add_argument('-TV_type', dest='TV_type',help='Type of total variation, iso or l1', type=str,default = 'iso')#'l1'
args = parser.parse_args()


MaxCG_Iter = args.MaxCG_Iter
sketch_size = args.sketch_size
Inner_iter = args.Inner_iter
MaxIter = args.MaxIter
TV_type = args.TV_type
cuda_index = args.cuda_index
iSmkdir = args.iSmkdir
device = torch.device('cuda:'+cuda_index)

# Open the .mat file
imFile = scipy.io.loadmat('./Walnutdata/Data328Proj40.mat')
imGTFile = scipy.io.loadmat('./Walnutdata/imGT328.mat')
im = (imGTFile['imGT']).astype(np.float64)
A = imFile['A']
A_coo = A.tocoo()
indices = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long)
values = torch.tensor(A_coo.data, dtype=torch.float32)
A_torch_sparse = torch.sparse_coo_tensor(indices, values, A_coo.shape).coalesce()
A_torch_dense = A_torch_sparse.to_dense().to(device)
#A_torch_dense = torch.from_numpy(A).to(device)
b = imFile['m']
im_max = 1#np.max(im)
im = im/im_max
A_torch_dense = A_torch_dense/im_max
b = b/im_max
b_m,b_n = b.shape
im_size = im.shape[0]
im_size_prod = im_size*im_size
im_original = torch.from_numpy(im).squeeze().unsqueeze(0).unsqueeze(0).to(device).to(torch.float32)
b = torch.from_numpy(b).squeeze().unsqueeze(0).unsqueeze(0).to(device).to(torch.float32)

Ax = lambda x: (torch.reshape(A_torch_dense@torch.reshape(torch.squeeze(x).t(),(im_size_prod,1)),(b_n,b_m))).t().unsqueeze(0).unsqueeze(0)
ATx = lambda x: (torch.reshape(A_torch_dense.t()@torch.reshape(torch.squeeze(x).t(),(b_m*b_n,1)),(im_size,im_size))).t().unsqueeze(0).unsqueeze(0)

# get the Lipschitz constant
L_sch = utl.Power_Iter([im_size, im_size],A=Ax,AT=ATx,device=device)
print('The Lip. constant is: {}\n'.format(L_sch))
L_numpy = np.sqrt(L_sch.cpu().numpy())


Ax_scale = lambda x: (torch.reshape(A_torch_dense@torch.reshape(torch.squeeze(x).t(),(im_size_prod,1)),(b_n,b_m))).t().unsqueeze(0).unsqueeze(0)/L_numpy
ATx_scale = lambda x: (torch.reshape(A_torch_dense.t()@torch.reshape(torch.squeeze(x).t(),(b_m*b_n,1)),(im_size,im_size))).t().unsqueeze(0).unsqueeze(0)/L_numpy
b = b/L_numpy

output = b
output_noise = output

# build the sketch preconditioner
start_time = time.perf_counter()
U,S,lambda_l = CTutl.Build_Sketch_Real_Pred(Ax,ATx,im_size,im_size_prod,sketch_size,isBatch=False,device=device)
end_time = time.perf_counter()
sketch_time = end_time-start_time
print('The cost time of building the preconditioner {}\n'.format(sketch_time))
# build preconditioner
mu = args.mu
U_temp = U*torch.sqrt(1-(lambda_l+mu)/(S+mu))
P_inv = lambda x: CTutl.P_invx_SimpReal(x,U_temp,im_size)
L_pre = utl.Power_Iter([im_size, im_size],A=Ax_scale,AT=ATx_scale,P=P_inv,device=device)
print('The New Lip. Constant is {}\n'.format(L_pre.cpu().numpy()))
L_P_inv = utl.Power_Iter([im_size, im_size],P=P_inv,device=device)
L_P_inv = 1/L_P_inv # minimal eigenvalue of the preconditioner
b_noise = output_noise
verbose = True

verbose = args.verbose
beta = args.beta

algName = 'RealFISTATV'+args.TV_type
loc = args.imSavedPathFolder + algName
if iSmkdir:    
    if not os.path.exists(loc):
        os.mkdir(loc)
x_ord,lst_cost_ord,psnr_set_FISTA_TV_ord,\
    ssim_set_FISTA_TV_ord,lst_time_ord,lst_mse_ord = \
        utl.FISTA_TV(MaxIter,Ax_scale,ATx_scale,b_noise,1,Num_iter = Inner_iter,TR_off = beta,TV_type = TV_type,save=loc,SaveIter=args.isSave,\
                             original=None,verbose=verbose,device=device)


algName = 'RealPreFISTATV'+args.TV_type+'sketchsize'+str(sketch_size)
loc = args.imSavedPathFolder+algName
if iSmkdir:    
    if not os.path.exists(loc):
        os.mkdir(loc)
L_acc = L_pre
x_pre,lst_cost_pre,psnr_set_FISTA_TV_pre,\
    ssim_set_FISTA_TV_pre,lst_time_pre,lst_mse_pre  = \
        utl.FISTA_TV(MaxIter,Ax_scale,ATx_scale,b_noise,L_acc,P=P_inv,Num_iter = Inner_iter,TV_type = TV_type,L_P_inv=L_P_inv,TR_off = beta,save=loc,SaveIter=args.isSave,\
                             original=None,verbose=verbose,device=device)
