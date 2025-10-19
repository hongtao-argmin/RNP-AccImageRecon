# Tao Hong, Zhaoyi Xu, Jason Hu, and Jeffrey A. Fessler, 
# ``Using Randomized Nyström Preconditioners to Accelerate Variational Image Reconstruction'', To appear in IEEE Transactions on Computational Imaging, arXiv:2411.08178, 2025.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
import utilities as utl
import scipy.io

def utl_PowerIter(im_size,A=lambda x: x,isComplex = False,dim=2,device='cpu',tol = 1e-6):
    ''' 
    Power iteration to estimate the maximal eigenvalue of AHA.
    '''
    if isComplex:
        if dim==2:
            b_k = torch.randn(im_size[0],im_size[1],dtype=torch.complex32).unsqueeze(0).unsqueeze(0).to(device)
        elif dim==3:
            b_k = torch.randn(im_size[0],im_size[1],im_size[2],dtype=torch.complex32).unsqueeze(0).unsqueeze(0).to(device)
    else:
        if dim==2:
            b_k = torch.randn(im_size[0],im_size[1]).unsqueeze(0).unsqueeze(0).to(device)
        elif dim == 3:
            b_k = torch.randn(im_size[0],im_size[1],im_size[2]).unsqueeze(0).unsqueeze(0).to(device)
   
    Ab_k = A(b_k)
    norm_b_k = torch.norm(Ab_k)
    while True:
        b_k = Ab_k/norm_b_k
        Ab_k = A(b_k)
        norm_b_k_1 = torch.norm(Ab_k)
        if torch.abs(norm_b_k_1-norm_b_k)<=tol:#/norm_b_k
            break
        else:
            norm_b_k = norm_b_k_1
    L = torch.vdot(b_k.flatten(),Ab_k.flatten()/torch.vdot(b_k.flatten(),b_k.flatten()))
    return torch.real(L)

def IRM_lplq(num_iters, Ax, ATx, Phix,PhiTx,b,p,q,TR_off,im_size,sketch_size=0,isPre=False,PhiTxBatch=None,x_ini=None,\
             CG_Tolerance=1e-8,isColor=False,b_noisy=None,save=None, verbose = True, original= None,SaveIter=True,device='cpu'):
  """
  Unconstrained Optimization.
  Solve the optimization problem using IRN:

    \min_x \frac{1}{p} \| A x - b \|_p^p + TR_off/q*\|\Phi x\|_q^q
  suppose square image that im_size represents the size of the dimension.
  Inputs:
    num_iters : Maximum number of outer iterations.
    Ax  : Forward model.
    ATx : adjoint of the forward model.
    Phix,PhiTx: forward and adj. operator for the reg. part.
    b (Array): Measurement.
    save/SaveIter (None or String): If specified, path to save iterations and timings.
    verbose (Bool): Print information.
  Returns:
    x (Array): Reconstruction.
    lst_cost,lst_psnr,lst_ssim,np.cumsum(lst_time),lst_iter: measured criteria
  """
  epsi = 1e-6
  im_size_prod = im_size[0]*im_size[1]
  AHb = ATx(b)
  if x_ini==None:
      x =  AHb
  else:
      x = x_ini
  lst_time  = []
  lst_cost = []
  lst_psnr = []
  lst_ssim = []
  lst_iter = []
  if verbose:
      if isPre:
          pbar = tqdm(total=num_iters, desc="Pre-lp_lq", \
                      leave=True)
      else:
          pbar = tqdm(total=num_iters, desc="lp_lq", \
                      leave=True)

  lst_cost.append((1/p*torch.sum(torch.abs(Ax(x)-b)**p)**(1.0/p)+TR_off/q*torch.sum(torch.abs(Phix(x))**q)**(1.0/q)).cpu().numpy())
  lst_time.append(0)
  if original is not None:
      lst_psnr.append(utl.PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
      lst_ssim.append(utl.SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
  for k in range(num_iters):
      start_time = time.perf_counter()
      W_F = ((Ax(x)-b)**2+epsi)**((p-2)/2)      
      W_R = ((Phix(x))**2+epsi)**((q-2)/2)
      
      Forward_A = lambda x: ATx(W_F*Ax(x))+TR_off*PhiTx(W_R*Phix(x))
      b_new = ATx(W_F*b)
      if isPre:
          # build preconditioner
          Forward_A_Batch_1 = lambda x: ATx(W_F*Ax(x))
          Forward_A_Batch_2 = lambda x: utl_forward_reg(x,Phix,PhiTxBatch,TR_off*W_R,sketch_size)
          U,S,lambda_l = utl_Build_Sketch_Pred(Forward_A_Batch_1,Forward_A_Batch_2,im_size,im_size_prod,sketch_size,device=device)
          U_temp = U*torch.sqrt(1-lambda_l/S)
          P = lambda x: (x.flatten()-U_temp@(U_temp.t()@x.flatten())).view([1, 1, im_size[0], im_size[1]])
      else:
          P = lambda x: x
      # run CG here
      r_k = b_new-Forward_A(x)
      z_k = P(r_k)
      count = 0
      while True:
        count = count+1
        p_k = r_k
        Ap_k = Forward_A(p_k)
        alpha_k = torch.vdot(r_k.flatten(),z_k.flatten())/torch.vdot(p_k.flatten(),Ap_k.flatten())
        x = x+alpha_k*p_k
        r_k_1 = r_k-alpha_k*Ap_k
        if torch.norm(r_k_1)<CG_Tolerance:
            break
        else:
            z_k_1 = P(r_k_1)
            beta_k = torch.vdot(r_k_1.flatten(),z_k_1.flatten())/torch.vdot(r_k.flatten(),z_k.flatten())
            p_k_1 = z_k_1+beta_k*p_k
            p_k = p_k_1
            r_k = r_k_1
            z_k = z_k_1
      end_time = time.perf_counter()
      lst_time.append(end_time - start_time)
      lst_iter.append(count)
      if original is not None:
          lst_psnr.append(utl.PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
          lst_ssim.append(utl.SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
      
      lst_cost.append((1/p*torch.sum(torch.abs(Ax(x)-b)**p)**(1.0/p)+TR_off/q*torch.sum(torch.abs(Phix(x))**q)**(1.0/q)).cpu().numpy())
      if save != None:
          if original is not None:
              Dict = {'lst_time':np.cumsum(lst_time),'lst_cost':lst_cost,'lst_iter':lst_iter,'lst_psnr':lst_psnr,'lst_ssim':lst_ssim}
              scipy.io.savemat("%s/TimeCostiter.mat" % save,Dict)
          else:
              Dict = {'lst_time':np.cumsum(lst_time),'lst_cost':lst_cost,'lst_iter':lst_iter}
              scipy.io.savemat("%s/TimeCostiter.mat" % save,Dict)
          if SaveIter:
              if isColor:
                  temp_merge = utl.MergeChannels(b_noisy,x)
                  Dict_image = {'im':torch.squeeze(temp_merge).cpu().numpy()}
                  scipy.io.savemat("%s/iter_%03d.mat" % (save, k),Dict_image)
              else:
                  Dict_image = {'im':torch.squeeze(x).cpu().numpy()}
                  scipy.io.savemat("%s/iter_%03d.mat" % (save, k),Dict_image)
      if verbose:
          pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
          pbar.update()
          pbar.refresh()
  if verbose:
      pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
      pbar.close()      
  return x,lst_cost,lst_psnr,lst_ssim,np.cumsum(lst_time),lst_iter

def utl_forward_reg(x,Phix,PhiTxBatch,W_R,sketch_size):
    # forward part for the regularization linear operator 
    # implemet L^T W_R L(x)
    temp = Phix(x)
    if temp.shape[0]>sketch_size:
        Y = PhiTxBatch(torch.cat((W_R[0,:,:,:].unsqueeze(0)*temp[0:sketch_size,:,:,:],W_R[1,:,:,:].unsqueeze(0)*temp[sketch_size:,:,:,:]),0))
    else:
        Y = PhiTxBatch(W_R*temp)
    return Y

def utl_Build_Sketch_Pred(Ax,Ax_2,im_size,im_size_prod,sketch_size,device='cpu'):
    epsi = 1e-10*np.sqrt(im_size_prod)
    Omega = torch.randn(sketch_size,1,im_size[0],im_size[1],device=device)/np.sqrt(im_size_prod)
    Y = Ax(Omega)+Ax_2(Omega)
    Omega = torch.reshape(torch.permute(Omega,(2,3,1,0)),(im_size_prod,sketch_size))
    Y = torch.reshape(torch.permute(Y,(2,3,1,0)),(im_size_prod,sketch_size))
    Y_corr = Y+epsi*Omega
    L = torch.linalg.cholesky(Omega.t()@Y_corr)
    B = torch.linalg.solve_triangular(L,Y_corr.t(),upper=False).t()
    USV = torch.linalg.svd(B,full_matrices=False)
    S = USV.S
    lambda_l = S[-1]
    S = torch.maximum(torch.zeros_like(S),S**2-epsi)
    U = USV.U
    return U,S,lambda_l

def utl_P_invx(x,Lambda_inv,U,par_fix,im_size):
    x = torch.squeeze(x)
    temp = U.t()@x.flatten()
    y = (U@(par_fix*(Lambda_inv@temp)-temp) + x.flatten()).view([1, 1, im_size[0], im_size[1]])
    return y