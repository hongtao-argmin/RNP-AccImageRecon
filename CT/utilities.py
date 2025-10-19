#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Tao Hong, Zhaoyi Xu, Jason Hu, and Jeffrey A. Fessler, 
# ``Using Randomized Nyström Preconditioners to Accelerate Variational Image Reconstruction'', To appear in IEEE Transactions on Computational Imaging, arXiv:2411.08178, 2025.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import cv2
import scipy.io

def PowerIterWav(im_size,num_level,im_coeff_size,A=lambda x: x,AT=lambda x: x,Wx=lambda x: x,WTx=lambda x: x,P=lambda x: x,device='cuda:0',tol = 1e-6):
    ''' 
    Power iteration to estimate the maximal eigenvalue of AHA.
    L_scale is only for leapProj
    '''
    b_k = torch.randn(im_size).unsqueeze(0).unsqueeze(0).to(device)
    Ab_k = P(wavArray2vec(Wx(AT(A(WTx(vec2wavArray(b_k,num_level,im_coeff_size))))),num_level,im_size[0],im_coeff_size,device=device))
    norm_b_k = torch.norm(Ab_k)
    while True:
        b_k = Ab_k/norm_b_k
        Ab_k = P(wavArray2vec(Wx(AT(A(WTx(vec2wavArray(b_k,num_level,im_coeff_size))))),num_level,im_size[0],im_coeff_size,device=device))
        norm_b_k_1 = torch.norm(Ab_k)
        if torch.abs(norm_b_k_1-norm_b_k)<=tol:#/norm_b_k
            break
        else:
            norm_b_k = norm_b_k_1
    L = torch.vdot(b_k.flatten(),Ab_k.flatten()/torch.vdot(b_k.flatten(),b_k.flatten()))
    return torch.real(L)

def CG_Pre(x,RHS,Ax,P = lambda x:x,Max_Iter = 50,tol=1e-4):
    # solve Ax=RHS with P as the preconditioner
    # implement in pytorch
    r_k = RHS-Ax(x)
    count_iter = 2
    z_k = P(r_k)
    for _ in range(Max_Iter):
        p_k = r_k
        Ap_k = Ax(p_k)
        count_iter = count_iter+2 
        alpha_k = torch.sum(r_k*z_k)/torch.sum(p_k*Ap_k)
        x = x+alpha_k*p_k
        r_k_1 = r_k-alpha_k*Ap_k
        if torch.norm(r_k_1)<tol:
            break
        else:
            z_k_1 = P(r_k_1)
            beta_k = torch.sum(r_k_1*z_k_1)/torch.sum(r_k*z_k)
            p_k_1 = z_k_1+beta_k*p_k
            p_k = p_k_1
            r_k = r_k_1
            z_k = z_k_1
    return x,count_iter

def WPMIter(x,v,P,alpha_p,TR,MaxIter = 1000,tol=1e-6):
    # min_x 0.5*\|x-y\|_P^2+lambda*\|x\|_1
    x_old = x
    z = x
    t_k_1 = 1
    for _ in range(MaxIter):
        temp_grad = z - alpha_p*P(z-v)
        x = torch.sign(temp_grad) * torch.clamp(torch.abs(temp_grad) - TR*alpha_p, min=0)
        if torch.norm(x-x_old)<tol:
            break
        t_k = t_k_1
        t_k_1 = (1+np.sqrt(1+4*t_k**2))/2
        z = x + ((t_k-1)/t_k_1)*(x - x_old)
        x_old = x
    return x

def WPMTVReg(v,TR_off,PreOp=lambda x:x,P_1=None,P_2=None,\
                     Num_iter=100,tol=1e-6,TV_bound='Dirchlet',\
                         TV_type = 'l1',device='cpu',isComplex = False):
    # Compute the WPM. 
    v = torch.squeeze(v)
    m,n = v.shape
    if P_1 is None:
        if TV_bound == 'Neumann':
            if isComplex:
                P_1 = torch.zeros(m-1,n,dtype=torch.complex32,device=device)
                P_2 = torch.zeros(m,n-1,dtype=torch.complex32,device=device)
                R_1 = torch.zeros(m-1,n,dtype=torch.complex32,device=device)
                R_2 = torch.zeros(m,n-1,dtype=torch.complex32,device=device) 
            else:
                P_1 = torch.zeros(m-1,n,device=device)
                P_2 = torch.zeros(m,n-1,device=device)
                R_1 = torch.zeros(m-1,n,device=device)
                R_2 = torch.zeros(m,n-1,device=device)   
        elif TV_bound == 'Dirchlet' or TV_bound == 'Periodic':
            if isComplex:
                P_1 = torch.zeros(m,n,dtype=torch.complex32,device=device)
                P_2 = torch.zeros(m,n,dtype=torch.complex32,device=device)
                R_1 = torch.zeros(m,n,dtype=torch.complex32,device=device)
                R_2 = torch.zeros(m,n,dtype=torch.complex32,device=device)
            else:
                P_1 = torch.zeros(m,n,device=device)
                P_2 = torch.zeros(m,n,device=device)
                R_1 = torch.zeros(m,n,device=device)
                R_2 = torch.zeros(m,n,device=device)
    else:
        R_1 = P_1
        R_2 = P_2
    t_k_1 = 1
    if isComplex:
        x_out = torch.zeros(m,n,dtype=torch.complex32,device=device)
    else:
        x_out = torch.zeros(m,n,device=device)
    #sigma_max = sigma_max/1.3 # ref our Nesterov paper Hong and Yavneh NLAA2022 
    for iter in range(Num_iter):
        x_old = x_out
        P_1_old = P_1
        P_2_old = P_2
        temp = Lforward(R_1,R_2,TV_bound,isComplex=False,device=device)
        x_out = v-TR_off*torch.squeeze(PreOp(temp))
        re = torch.norm(x_old-x_out)/torch.norm(x_out)
        if re<tol or iter == Num_iter-1:
            break 
        Q_1,Q_2 = Ltrans(x_out,m,n,TV_bound)
        P_1 = R_1+1/(8*TR_off)*Q_1
        P_2 = R_2+1/(8*TR_off)*Q_2
        #perform project step
        if TV_type == 'iso':
            if TV_bound == 'Neumann':
                if isComplex:
                    temp = torch.abs(torch.vstack((P_1,torch.zeros(1,n,dtype=torch.complex32,device=device))))**2+\
                    torch.abs(torch.column_stack((P_2,torch.zeros(m,1,dtype=torch.complex32,device=device))))**2
                else:
                    temp = torch.abs(torch.vstack((P_1,torch.zeros(1,n,device=device))))**2+\
                    torch.abs(torch.column_stack((P_2,torch.zeros(m,1,device=device))))**2
                temp = torch.sqrt(torch.maximum(temp,torch.ones_like(temp,device=device)))
                P_1 = P_1/temp[0:m-1,:]
                P_2 = P_2/temp[:,0:n-1]
            elif TV_bound == 'Dirchlet' or TV_bound == 'Periodic':
                temp = torch.abs(P_1)**2+torch.abs(P_2)**2
                temp = torch.sqrt(torch.maximum(temp,torch.ones_like(temp,device=device)))
                P_1 = P_1/temp
                P_2 = P_2/temp                            
        elif TV_type == 'l1':
            P_1_abs = torch.abs(P_1)
            P_2_abs = torch.abs(P_2)
            P_1 = P_1/torch.maximum(P_1_abs,torch.ones_like(P_1_abs,device=device))
            P_2 = P_2/torch.maximum(P_2_abs,torch.ones_like(P_2_abs,device=device))
        t_k = t_k_1
        t_k_1 = (1+np.sqrt(1+4*t_k**2))/2
        R_1 = P_1 + ((t_k-1)/t_k_1)*(P_1-P_1_old)
        R_2 = P_2 + ((t_k-1)/t_k_1)*(P_2-P_2_old)
    return x_out.unsqueeze(0).unsqueeze(0),P_1,P_2

def Power_Iter(im_size,W=None,L_scale=1,A=lambda x: x,AT=lambda x: x,P=lambda x: x,isComplex = False,dim=2,device='cuda:0',tol = 1e-6):
    ''' 
    Power iteration to estimate the maximal eigenvalue of AHA.
    L_scale is only for leapProj
    '''
    if isComplex:
        if dim==2:
            b_k = torch.randn(im_size,dtype=torch.complex32).unsqueeze(0).unsqueeze(0).to(device)
        elif dim==3:
            b_k = torch.randn(im_size,dtype=torch.complex32).unsqueeze(0).to(device)
    else:
        if dim==2:
            b_k = torch.randn(im_size).unsqueeze(0).unsqueeze(0).to(device)
        elif dim == 3:
            b_k = torch.randn(im_size).unsqueeze(0).to(device)
    if W==None:
        Ab_k = P((AT(A(b_k))))
    else:
        Ab_k = P((AT(W*A(b_k))))
    norm_b_k = torch.norm(Ab_k)
    while True:
        b_k = Ab_k/norm_b_k
        if W==None:
            Ab_k = P((AT(A(b_k))))
        else:
            Ab_k = P((AT(W*A(b_k))))
        norm_b_k_1 = torch.norm(Ab_k)
        if torch.abs(norm_b_k_1-norm_b_k)<=tol:#/norm_b_k
            break
        else:
            norm_b_k = norm_b_k_1
    #b = b_k
    L = torch.vdot(b_k.flatten(),Ab_k.flatten()/torch.vdot(b_k.flatten(),b_k.flatten()))
    return torch.real(L)

# functions for TV proximal part
def Lforward(P_1,P_2,TV_bound='Dirchlet',isComplex = False,device = 'cpu'):
    m2,n2 = P_1.shape
    m1,n1 = P_2.shape
    if TV_bound=='Neumann':
        if n2!=n1+1:
            print('dimensions are not consistent\n')
        if m1!=m2+1:
            print('dimension are not consistent\n')
        m = m2+1
        n = n2
        if isComplex:
            X = torch.zeros(m,n,dtype=torch.complex32,device=device)
        else:
            X = torch.zeros(m,n,device=device)
        X[0:m-1,:] = P_1
        X[:,0:n-1] = X[:,0:n-1]+P_2
        X[1:m,:] = X[1:m,:]-P_1
        X[:,1:n] = X[:,1:n]-P_2
    elif TV_bound=='Dirchlet':
        m = m2
        n = n2
        if isComplex:
            X = torch.zeros(m,n,dtype=torch.complex32,device=device)
        else:
            X = torch.zeros(m,n,device=device)
        X[0:m-1,:] = P_1[0:m-1,:]
        X[:,0:n-1] = X[:,0:n-1]+P_2[:,0:n-1]
        X[1:m,:] = X[1:m,:]-P_1[0:m-1,:]
        X[:,1:n] = X[:,1:n]-P_2[:,0:n-1]
        # correct boundary
        X[0:m-1,n-1] = X[0:m-1,n-1]+P_2[0:m-1,n-1]
        X[m-1,0:n-1] = X[m-1,0:n-1]+P_1[m-1,0:n-1]
        X[m-1,n-1] = X[m-1,n-1]+P_1[m-1,n-1]+P_2[m-1,n-1]
    elif TV_bound=='Periodic':
        m = m2
        n = n2
        if isComplex:
            X = torch.zeros(m,n,dtype=torch.complex32,device=device)
        else:
            X = torch.zeros(m,n,device=device)
        X[0:m-1,:] = P_1[0:m-1,:]
        X[:,0:n-1] = X[:,0:n-1]+P_2[:,0:n-1]
        X[1:m,:] = X[1:m,:]-P_1[0:m-1,:]
        X[:,1:n] = X[:,1:n]-P_2[:,0:n-1]
        # correct boundary
        X[0:m-1,n-1] = X[0:m-1,n-1]+P_2[0:m-1,n-1]
        X[0:m-1,0] = X[0:m-1,0]-P_2[0:m-1,n-1]
        X[m-1,0:n-1] = X[m-1,0:n-1]+P_1[m-1,0:n-1]
        X[0,0:n-1] = X[0,0:n-1]-P_1[m-1,0:n-1]
        X[m-1,n-1] = X[m-1,n-1]+P_1[m-1,n-1]+P_2[m-1,n-1]
        X[0,n-1] = X[0,n-1]-P_1[m-1,n-1]
        X[m-1,0] = X[m-1,0]-P_2[m-1,n-1]
    return X

def Ltrans(X,m,n,TV_bound):
    if TV_bound == 'Neumann':
        P_1 = X[0:m-1,:]-X[1:m,:]
        P_2 = X[:,0:n-1]-X[:,1:n]
    elif TV_bound=='Dirchlet':
        P_1 =  torch.vstack((X[0:m-1,:]-X[1:m,:],X[m-1,:]))
        P_2 =  torch.column_stack((X[:,0:n-1]-X[:,1:n],X[:,n-1]))
    elif TV_bound=='Periodic':
        P_1 = X-torch.vstack((X[1:m,:],X[0,:]))
        P_2 = X-torch.column_stack((X[:,1:n-1],X[:,0]))
    return P_1,P_2

def GetGradSingle(X,TV_bound='Dirchlet',Dir='x-axis',isAdjoint = False,device='cpu'):
     """
    Define the x and y direction gradient operation with the adjoint operator.
   """
     X = torch.squeeze(X)
     m,n = X.shape
     if isAdjoint:
         if Dir == 'x-axis':
             if TV_bound == 'Dirchlet':
                 Dx = torch.vstack((X[0,:],X[1:m-1,:]-X[0:m-2,:]))
                 Dx = torch.vstack((Dx,X[m-1,:]-X[m-2,:]))
             elif TV_bound == 'Neumann':
                 Dx = torch.vstack((X[0,:],X[1:m-1,:]-X[0:m-2]))
                 Dx = torch.vstack((Dx,-X[m-2,:]))
             elif TV_bound == 'Periodic':
                 Dx = torch.vstack((X[0,:]-X[m-1,:],X[1:m,:]-X[0:m-1,:]))
         elif Dir == 'y-axis':
             if TV_bound == 'Dirchlet':
                 Dx = torch.column_stack((X[:,0],X[:,1:n-1]-X[:,0:n-2]))
                 Dx = torch.column_stack((Dx,X[:,n-1]-X[:,n-2]))
             elif TV_bound == 'Neumann':
                 Dx = torch.column_stack((X[:,0],X[:,1:n-1]-X[:,0:n-2]))
                 Dx = torch.column_stack((Dx,-X[:,n-2]))
             elif TV_bound == 'Periodic':
                 Dx = torch.column_stack((X[:,0]-X[:,n-1],X[:,1:n]-X[:,0:n-1]))
     else:
         if Dir == 'x-axis':
             if TV_bound == 'Dirchlet':
                 Dx = torch.vstack((X[0:m-1,:]-X[1:m,:],X[m-1,:]))
             elif TV_bound == 'Neumann':
                 Dx = torch.vstack((X[0:m-1,:]-X[1:m,:],torch.zeros(1,n,dtype=torch.float32,device=device)))
             elif TV_bound == 'Periodic':
                 Dx = torch.vstack((X[0:m-1,:]-X[1:m,:],X[m-1,:]-X[0,:]))
         elif Dir == 'y-axis':
             if TV_bound == 'Dirchlet':
                 Dx = torch.column_stack((X[:,0:n-1]-X[:,1:n],X[:,n-1]))
             elif TV_bound == 'Neumann':
                 Dx = torch.column_stack((X[:,0:n-1]-X[:,1:n],torch.zeros(m,1,dtype=torch.float32,device=device)))
             elif TV_bound == 'Periodic':
                 Dx = torch.column_stack((X[:,0:n-1]-X[:,1:n],X[:,n-1]-X[:,0]))
     return Dx

def GetGradSingle_Pytorch(X,TV_bound='Dirchlet',Dir='x-axis',isAdjoint = False,device='cpu'):
     """
    Define the x and y direction gradient operation with the adjoint operator.
    # work on batch mode
   """
     numBatch,_,m,n = X.shape
     if isAdjoint:
         if Dir == 'x-axis':
             if TV_bound == 'Dirchlet':
                 Dx = torch.cat(((X[:,:,0,:]).unsqueeze(2),X[:,:,1:m-1,:]-X[:,:,0:m-2,:]),2)
                 Dx = torch.cat((Dx,(X[:,:,m-1,:]-X[:,:,m-2,:]).unsqueeze(2)),2)
             elif TV_bound == 'Neumann':
                 Dx = torch.cat(((X[:,:,0,:]).unsqueeze(2),X[:,:,1:m-1,:]-X[:,:,0:m-2]),2)
                 Dx = torch.cat((Dx,(-X[:,:,m-2,:]).unsqueeze(2)),2)
             elif TV_bound == 'Periodic':
                 Dx = torch.cat(((X[:,:,0,:]-X[:,:,m-1,:]).unsqueeze(2),X[:,:,1:m,:]-X[:,:,0:m-1,:]),2)
         elif Dir == 'y-axis':
             if TV_bound == 'Dirchlet':
                 Dx = torch.cat(((X[:,:,:,0]).unsqueeze(3),X[:,:,:,1:n-1]-X[:,:,:,0:n-2]),3)
                 Dx = torch.cat((Dx,(X[:,:,:,n-1]-X[:,:,:,n-2]).unsqueeze(3)),3)
             elif TV_bound == 'Neumann':
                 Dx = torch.cat(((X[:,:,:,0]).unsqueeze(3),X[:,:,:,1:n-1]-X[:,:,:,0:n-2]),3)
                 Dx = torch.cat((Dx,(-X[:,:,:,n-2]).unsqueeze(3)),3)
             elif TV_bound == 'Periodic':
                 Dx = torch.cat(((X[:,:,:,0]-X[:,:,:,n-1]).unsqueeze(3),X[:,:,:,1:n]-X[:,:,:,0:n-1]),3)
     else:
         if Dir == 'x-axis':
             if TV_bound == 'Dirchlet':
                 Dx = torch.cat((X[:,:,0:m-1,:]-X[:,:,1:m,:],(X[:,:,m-1,:]).unsqueeze(2)),2)
             elif TV_bound == 'Neumann':
                 Dx = torch.cat((X[:,:,0:m-1,:]-X[:,:,1:m,:],torch.zeros(numBatch,1,1,n,dtype=torch.float32,device=device)),2)
             elif TV_bound == 'Periodic':
                 Dx = torch.cat((X[:,:,0:m-1,:]-X[:,:,1:m,:],(X[:,:,m-1,:]-X[:,:,0,:]).unsqueeze(2)),2)
         elif Dir == 'y-axis':
             if TV_bound == 'Dirchlet':
                 Dx = torch.cat((X[:,:,:,0:n-1]-X[:,:,:,1:n],(X[:,:,:,n-1]).unsqueeze(3)),3)
             elif TV_bound == 'Neumann':
                 Dx = torch.cat((X[:,:,:,0:n-1]-X[:,:,:,1:n],torch.zeros(numBatch,1,m,1,dtype=torch.float32,device=device)),3)
             elif TV_bound == 'Periodic':
                 Dx = torch.cat((X[:,:,:,0:n-1]-X[:,:,:,1:n],(X[:,:,:,n-1]-X[:,:,:,0]).unsqueeze(3)),3)
     return Dx

def TV_Projection(P_1,P_2,m,n,TV_bound='Dirchlet',TV_type='l1',device='cpu'):    
      if TV_type == 'iso':
          if TV_bound == 'Neumann':
              temp = torch.vstack((P_1,torch.zeros(1,n,dtype=torch.complex32,device=device)))**2+torch.column_stack((P_2,torch.zeros(m,1,dtype=torch.complex64,device=device)))**2
              temp = torch.sqrt(torch.maximum(temp,torch.ones_like(temp,device=device)))
              P_1 = P_1/temp[0:m-1,:]
              P_2 = P_2/temp[:,0:n-1]
          elif TV_bound == 'Dirchlet' or TV_bound == 'Periodic':
              temp = P_1**2+P_2**2
              temp = torch.sqrt(torch.maximum(temp,torch.ones_like(temp,device=device)))
              P_1 = P_1/temp
              P_2 = P_2/temp                            
      elif TV_type == 'l1':
          P_1 = P_1/torch.maximum(torch.abs(P_1),torch.ones_like(P_1,device=device))
          P_2 = P_2/torch.maximum(torch.abs(P_2),torch.ones_like(P_2,device=device))
      return P_1,P_2 

def SSIM(original,compressed):
    if np.max(original)<1.1:
        data_range = 1.0
    else:
        data_range = 255.0
    return ssim(original,compressed,\
                data_range=data_range)

def PSNR(original, compressed):
    if np.max(original)<1.1:
        data_range = 1.0
    elif np.max(original)<256:
        data_range = 255.0
    else:
        data_range = 1.0
        original_max = np.max(original)
        original = original/original_max
        compressed = compressed/original_max
    return psnr(original,compressed,\
                data_range=data_range)

def obj_TV(X,TV_bound,TV_type,device='cpu'):
    # compute the cost value the TV part
    X = torch.squeeze(X)
    m,n = X.shape
    P_1,P_2 = Ltrans(X,m,n,TV_bound) 
    if TV_type == 'iso':
        if TV_bound=='Neumann':
            D = torch.zeros(m,n,device=device)
            D[0:m-1,:] = torch.abs(P_1)**2
            D[:,0:n-1] = D[:,0:n-1]+torch.abs(P_2)**2
            f_TV = torch.sum(torch.sqrt(D))
        elif TV_bound=='Dirchlet' or TV_bound=='Periodic':
            f_TV = torch.sum(torch.sqrt(torch.abs(P_1)**2+torch.abs(P_2)**2))
    elif TV_type == 'l1':
        f_TV = torch.sum(torch.abs(P_1))+torch.sum(torch.abs(P_2))
    return f_TV.cpu().numpy()

def mulAX(X,A,ATx,rho,TV_bound):
    Y = ATx(A(X))
    Dx = GetGradSingle(X,TV_bound=TV_bound,Dir='x-axis',isAdjoint = False)
    Dy = GetGradSingle(X,TV_bound=TV_bound,Dir='y-axis',isAdjoint = False)
    Y = Y+rho*(GetGradSingle(Dx,TV_bound=TV_bound,Dir='x-axis',isAdjoint = True)+\
    GetGradSingle(Dy,TV_bound=TV_bound,Dir='y-axis',isAdjoint = True)).unsqueeze(0).unsqueeze(0)
    return Y

def CG_Alg(x,RHS,A,ATx,rho,TV_bound,P=lambda x:x,MaxCG_Iter=20,tol=1e-4):
    r_k = RHS-mulAX(x,A,ATx,rho,TV_bound)
    iter_count = 2
    z_k = P(r_k)
    for _ in range(MaxCG_Iter):
        p_k = r_k
        Ap_k = mulAX(p_k,A,ATx,rho,TV_bound)
        iter_count = iter_count+2
        alpha_k = torch.sum(r_k*z_k)/torch.sum(p_k*Ap_k)
        x = x+alpha_k*p_k
        r_k_1 = r_k-alpha_k*Ap_k
        if torch.norm(r_k_1)<tol:
            break
        else:
            z_k_1 = P(r_k_1)
            beta_k = torch.sum(r_k_1*z_k_1)/torch.sum(r_k*z_k)
            p_k_1 = z_k_1+beta_k*p_k
            p_k = p_k_1
            r_k = r_k_1
            z_k = z_k_1
    return x,iter_count

def FISTA_TV(num_iters, Ax, ATx,b,L,TR_off,W=None,x_ini=None,RestartIter=50,isRestart = False,Num_iter = 100,P = lambda x: x,L_P_inv = 1,TV_bound = 'Dirchlet',TV_type = 'l1',\
             isColor=False,b_noisy=None,save=None, verbose = True,original= None,SaveIter=True,device='cpu'):
  """
  Unconstrained Optimization.
  Solve the optimization problem using FISTA:

    \min_x \frac{1}{2} \| A x - b \|_2^2 + TR_off*TV(x)
    or 
    \min_x \frac{1}{2} \| A x - b \|_w^2 + TR_off*TV(x)
    for this, the w is included in applying ATx.

  Inputs:
    num_iters : Maximum number of outer iterations.
    Ax  : Forward model.
    ATx : adjoint of the forward model.
    b (Array): Measurement.
    Num_iter : the number of iterations for computing the TV proximal operator.
    P : the preconditioned operator.
    L_P_inv : inversion of the maximal eigenvalue of the preconditioned matrix. 
    save (None or String): If specified, path to save iterations and timings.
    verbose (Bool): Print information.
    TV_bound, TV_type: the boundary condition and the type of TV ('iso' or 'l1'')
  Returns:
    x (Array): Reconstruction.
  """
  AHb = ATx(b)
  if x_ini==None:
      x = AHb
  else:
      x = x_ini
  z = x.clone()
  lst_time  = []
  lst_cost = []
  lst_psnr = []
  lst_ssim = []
  lst_mse = []
  P_1 = None
  P_2 = None
  if verbose:
      pbar = tqdm(total=num_iters, desc="FISTA_TV", \
                leave=True)
  if W==None:
      lst_cost.append((0.5*torch.norm(Ax(x)-b)**2).cpu().numpy()+TR_off*obj_TV(x,TV_bound,TV_type,device=device))
  else:
      lst_cost.append((0.5 * torch.norm((Ax(x) - b)*W*(Ax(x) - b))**2).cpu().numpy() + TR_off * obj_TV(x, TV_bound, TV_type, device=device))
  lst_time.append(0)
  if original is not None:
      lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
      lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
      lst_mse.append(np.linalg.norm(np.abs(original) - torch.squeeze(torch.abs(x)).cpu().numpy()))
  t_k_1 = 1
  for k in range(num_iters):
      start_time = time.perf_counter()
      x_old = x.clone()
      if W == None:
          gr = P(ATx(Ax(z)-b))/L
      else:
          gr = P(ATx(W*(Ax(z) - b)))/L
      temp_grad = z-gr
      # compute the wpm
      x,P_1,P_2 = WPMTVReg(temp_grad,(L_P_inv*TR_off)/L,PreOp=P,P_1=P_1,P_2=P_2,Num_iter=Num_iter,\
                                   TV_bound=TV_bound,TV_type=TV_type,device=device)
      t_k = t_k_1
      t_k_1 = (1+np.sqrt(1+4*t_k**2))/2
      if isRestart:
          if np.mod(k,RestartIter)==0:
            t_k = 1
            t_k_1 = 1
      #t_k = 1
      z = x + ((t_k-1)/t_k_1)*(x - x_old)
      end_time = time.perf_counter()
      lst_time.append(end_time - start_time)
      if original is not None:
          lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
          lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
          lst_mse.append(np.linalg.norm(np.abs(original) - torch.squeeze(torch.abs(x)).cpu().numpy()))
      if W==None:
          lst_cost.append((0.5*torch.norm(Ax(x)-b)**2).cpu().numpy()+TR_off*obj_TV(x,TV_bound,TV_type,device=device))
      else:
          lst_cost.append((0.5 * torch.norm((Ax(x) - b) * W * (Ax(x) - b))**2).cpu().numpy() + TR_off * obj_TV(x, TV_bound, TV_type,device=device))
      if save != None:
          if original is not None:
              Dict = {'lst_time':np.cumsum(lst_time),'lst_cost':lst_cost,'lst_psnr':lst_psnr,'lst_ssim':lst_ssim}
              scipy.io.savemat("%s/TimeCost.mat" % save,Dict)
          else:
              Dict = {'lst_time':np.cumsum(lst_time),'lst_cost':lst_cost}
              scipy.io.savemat("%s/TimeCost.mat" % save,Dict)
          if SaveIter:
              if isColor:
                  temp_merge = MergeChannels(b_noisy,x)
                  Dict_Im = {'im', torch.squeeze(temp_merge).cpu().numpy()}
                  scipy.io.savemat("%s/iter_%03d.mat" % (save, k),Dict_Im)
              else:
                  Dict_Im = {'im':torch.squeeze(x).cpu().numpy()}
                  scipy.io.savemat("%s/iter_%03d.mat" % (save, k),Dict_Im)
      if verbose:
          pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
          pbar.update()
          pbar.refresh()
  if verbose:
      pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
      pbar.close()
      
  return x,lst_cost,lst_psnr,lst_ssim,np.cumsum(lst_time),lst_mse

def FISTA_Wav(num_iters,Ax,ATx,b,L,TR_off,Wx,WTx,im_size,num_level,list_coeff_size,isPoly = False,isIterative=False,x_ini=None,U=None,RestartIter=50,isRestart = False,isPre=False,P = lambda x: x,L_P_inv = 1,\
             isColor=False,b_noisy=None,save=None,verbose = True,original= None,SaveIter=True,device='cpu'):
  """
  Unconstrained Optimization.
  Solve the optimization problem using FISTA:
    \min_x \frac{1}{2} \| A W^{-1}z - b \|_2^2 + TR_off*\|z\|_1
    # image x = W^{-1}z
    using orthogonal wavelet that WTx is equivalent to "WTx=W^{-1}".
  Inputs:
    num_iters : Maximum number of outer iterations.
    Ax  : Forward model.
    ATx : adjoint of the forward model.
    b (Array): Measurement.
    P : the preconditioned operator.
    L: corresponding Lipschitz constant
    L_P_inv : inversion of the maximal eigenvalue of the preconditioned matrix. 
    save (None or String): If specified, path to save iterations and timings.
    verbose (Bool): Print information.
    TR_off: trade-off parameter.
    Wx,WTx: wavelet and its inversion transforms. ** we only consider orthogonal wavelet.
    num_level: the levels of used wavelet.
    isIterative: True - use CG algorithm to get the inversion in the semi-smooth algorithm. If the sketch-size is too large, setting isIterative=True would be more efficiency.
    U: The rank-sketch_size of the preconditioner.
    isPre: True - use preconditioners.
    P: setting preconditioner.
    isIter: 
  Returns:
    x (Array): Reconstruction.
  """
  AHb = ATx(b)
  if x_ini==None:
      x = wavArray2vec(Wx(AHb),num_level,im_size,list_coeff_size,device=device)
  else:
      x = x_ini
  if isPre:
      size_U = U.size(1)
  z = x.clone()
  lst_time  = []
  lst_cost = []
  lst_psnr = []
  lst_ssim = []
  lst_mse = []
  if verbose:
      pbar = tqdm(total=num_iters, desc="FISTA_Wav", \
                leave=True)
  lst_cost.append((0.5 * torch.norm((Ax(WTx(vec2wavArray(x,num_level,list_coeff_size))) - b))**2).cpu().numpy() + TR_off * torch.sum(torch.abs(x)).cpu().numpy())
  lst_time.append(0)
  if original is not None:
      lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(WTx(vec2wavArray(x,num_level,list_coeff_size)))).cpu().numpy()))
      lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(WTx(vec2wavArray(x,num_level,list_coeff_size)))).cpu().numpy()))
      lst_mse.append(np.linalg.norm(np.abs(original) - torch.squeeze(torch.abs(WTx(vec2wavArray(x,num_level,list_coeff_size)))).cpu().numpy()))
  t_k_1 = 1
  TR_Off_WPM = (L_P_inv*TR_off)/L
  for k in range(num_iters):
      start_time = time.perf_counter()
      x_old = x.clone()
      gr = P(wavArray2vec(Wx(ATx(Ax(WTx(vec2wavArray(z,num_level,list_coeff_size)))-b)),num_level,im_size,list_coeff_size,device=device))/L
      temp_grad = z-gr
      if isPre:
          if isPoly:
              x = WPMIter(z,temp_grad,P,L_P_inv,TR_off/L)
          else:
            # compute the wpm with semi-smooth Newton
            # P = D+UU^T
            if k == 0:
                alpha_star = torch.zeros(size_U,1,device=device)

            temp_grad_vec = torch.reshape(temp_grad,(im_size*im_size,1))
            while True:
                # compute the Jacobi matrix
                temp_1 = temp_grad_vec-U@alpha_star
                ST_temp = torch.sign(temp_1) * torch.clamp(torch.abs(temp_1) - TR_Off_WPM, min=0.0)
                temp_grad_alpha = U.t()@(temp_grad_vec-ST_temp)+alpha_star
                if torch.norm(temp_grad_alpha)<1e-4:
                    break
                index_1 = torch.nonzero(ST_temp.squeeze())
                U_temp = U[index_1.squeeze(),:]
                #H_x = lambda x: x+U_temp.t()@(U_temp@x)
                # run CG to get the inversion or use the direct inversion
                if isIterative:
                    H_x = lambda x: x+U_temp.t()@(U_temp@x)
                    alpha_star_grad = alpha_star
                    r_k = temp_grad_alpha-H_x(alpha_star_grad)
                    z_k = r_k
                    while True:
                        p_k = r_k
                        Ap_k = H_x(p_k)
                        alpha_k = torch.vdot(r_k.flatten(),z_k.flatten())/torch.vdot(p_k.flatten(),Ap_k.flatten())
                        alpha_star_grad = alpha_star_grad+alpha_k*p_k
                        r_k_1 = r_k-alpha_k*Ap_k
                        if torch.norm(r_k_1)<1e-4:
                            break
                        else:
                            z_k_1 = r_k_1
                            beta_k = torch.vdot(r_k_1.flatten(),z_k_1.flatten())/torch.vdot(r_k.flatten(),z_k.flatten())
                            p_k_1 = z_k_1+beta_k*p_k
                            p_k = p_k_1
                            r_k = r_k_1
                            z_k = z_k_1
                    alpha_star = alpha_star-alpha_star_grad
                else:
                    H_mat = U_temp.t()@U_temp
                    H_mat = torch.eye(H_mat.size(0),device=device)+H_mat
                    alpha_star = alpha_star-torch.linalg.inv(H_mat)@temp_grad_alpha
            x = torch.reshape(ST_temp,(1,1,im_size,im_size))
      else:
          x = torch.sign(temp_grad) * torch.clamp(torch.abs(temp_grad) - TR_Off_WPM, min=0.0)
      t_k = t_k_1
      t_k_1 = (1+np.sqrt(1+4*t_k**2))/2
      if isRestart:
          if np.mod(k,RestartIter)==0:
            t_k = 1
            t_k_1 = 1
      #t_k = 1
      z = x + ((t_k-1)/t_k_1)*(x - x_old)
      end_time = time.perf_counter()
      lst_time.append(end_time - start_time)
      if original is not None:
          lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(WTx(vec2wavArray(x,num_level,list_coeff_size)))).cpu().numpy()))
          lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(WTx(vec2wavArray(x,num_level,list_coeff_size)))).cpu().numpy()))
          lst_mse.append(np.linalg.norm(np.abs(original) - torch.squeeze(torch.abs(WTx(vec2wavArray(x,num_level,list_coeff_size)))).cpu().numpy()))
      lst_cost.append((0.5*torch.norm(Ax(WTx(vec2wavArray(z,num_level,list_coeff_size)))-b)**2).cpu().numpy()+TR_off*torch.sum(torch.abs(x)).cpu().numpy())
      if save != None:
          if original is not None:
              Dict = {'lst_time':np.cumsum(lst_time),'lst_cost':lst_cost,'lst_psnr':lst_psnr,'lst_ssim':lst_ssim}
              scipy.io.savemat("%s/TimeCost.mat" % save,Dict)
          else:
              Dict = {'lst_time':np.cumsum(lst_time),'lst_cost':lst_cost}
              scipy.io.savemat("%s/TimeCost.mat" % save,Dict)
          if SaveIter:
              if isColor:
                  temp_merge = MergeChannels(b_noisy,WTx(vec2wavArray(x,num_level,list_coeff_size)))
                  Dict_im = {'im': torch.squeeze(temp_merge).cpu().numpy()}
                  scipy.io.savemat("%s/iter_%03d.mat" % (save, k),Dict_im)
              else:
                  Dict_im = {'im': torch.squeeze(WTx(vec2wavArray(x,num_level,list_coeff_size))).cpu().numpy()}
                  scipy.io.savemat("%s/iter_%03d.mat" % (save, k),Dict_im)
      if verbose:
          pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
          pbar.update()
          pbar.refresh()
  if verbose:
      pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
      pbar.close()      
  return WTx(vec2wavArray(x,num_level,list_coeff_size)),lst_cost,lst_psnr,lst_ssim,np.cumsum(lst_time)

# impement the wavelet coefficients list to a tensor 
def vec2wavArray(x,num_level,list_coeff_size):
    # assume the image is square
    y = []
    size_batch,_,im_size,_ = x.size()
    x = torch.reshape(x,(size_batch,im_size*im_size))
    count_begin = np.int32(list_coeff_size[0]**2)
    y.append(torch.reshape(x[:,0:count_begin],(size_batch,1,list_coeff_size[0],list_coeff_size[0])))
    count_begin = np.int32(list_coeff_size[0]**2)
    for iter in range(num_level):
        temp = []
        interval = np.int32(list_coeff_size[iter+1]**2)
        for iter_inner in range(3):
            temp.append(torch.reshape(x[:,count_begin:count_begin+interval],(size_batch,1,list_coeff_size[iter+1],list_coeff_size[iter+1])))
            count_begin = count_begin+interval
        y.append(temp)
    return y

def wavArray2vec(y,num_level,im_size,list_coeff_size,device='cpu'):
    size_batch = y[0].size(0)
    x = torch.zeros(size_batch,im_size*im_size,device=device)
    count_begin = np.int32(list_coeff_size[0]**2)
    x[:,0:count_begin] = torch.reshape(y[0],(size_batch,count_begin))
    for iter in range(num_level):
        interval = np.int32(list_coeff_size[iter+1]**2)
        for iter_inner in range(3):
            count_begin_end = count_begin+interval
            x[:,count_begin:count_begin_end] = torch.reshape(y[iter+1][iter_inner],(size_batch,interval))
            count_begin = count_begin_end
    x = torch.reshape(x,(size_batch,1,im_size,im_size))
    return x