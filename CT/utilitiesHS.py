# ---------------------- algorithms, utilities for HS based reconstruction --------------
# Author: Tao Hong. 12 Oct. 2024
# Tao Hong, Zhaoyi Xu, Jason Hu, and Jeffrey A. Fessler, 
# ``Using Randomized Nyström Preconditioners to Accelerate Variational Image Reconstruction'', To appear in IEEE Transactions on Computational Imaging, arXiv:2411.08178, 2025.
# ----------------------------------------------------------------------------------------
import torch
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
import utilities as utl

def Hx(x,isBatch=False,bnd='Dirchlet',device='cpu'):
    # x the image itself
    x = torch.squeeze(x)
    if isBatch:
        numBatch,M,N = x.size()
        x_temp = torch.zeros(numBatch,M+2,N+2,device=device)
        x_temp[:,1:-1,1:-1] = x
    else:
        M,N = x.size()
        x_temp = torch.zeros(M+2,N+2,device=device)
        x_temp[1:-1,1:-1] = x
    # boundary correction
    if bnd=='Periodic':
        if isBatch:
            # correct rows
            x_temp[:,0,1:-1] = x[:,-1,:]
            x_temp[:,-1,1:-1] = x[:,0,:]
            # correct col
            x_temp[:,1:-1,0] = x[:,:,-1]  
            x_temp[:,1:-1,-1] = x[:,:,0]
            # correct 4 corners.
            x_temp[:,0,0] = x[:,-1,-1]
            x_temp[:,0,-1] = x[:,-1,0]
            x_temp[:,-1,0] = x[:,0,-1]
            x_temp[:,-1,-1] = x[:,0,0]
        else:
            # correct rows
            x_temp[0,1:-1] = x[-1,:]
            x_temp[-1,1:-1] = x[0,:]
            # correct col
            x_temp[1:-1,0] = x[:,-1]  
            x_temp[1:-1,-1] = x[:,0]
            # correct 4 corners.
            x_temp[0,0] = x[-1,-1]
            x_temp[0,-1] = x[-1,0]
            x_temp[-1,0] = x[0,-1]
            x_temp[-1,-1] = x[0,0]
    elif bnd=='Neumann':
        if isBatch:
            # correct rows
            x_temp[:,0,1:-1] = x[:,0,:]
            x_temp[:,-1,1:-1] = x[:,-1,:]
            # correct col
            x_temp[:,1:-1,0] = x[:,:,0]  
            x_temp[:,1:-1,-1] = x[:,:,-1]
            # correct 4 corners.
            x_temp[:,0,0] = x[:,0,0]
            x_temp[:,0,-1] = x[0,-1]
            x_temp[:,-1,0] = x[:,-1,0]
            x_temp[:,-1,-1] = x[:,-1,-1]
        else:
            # correct rows
            x_temp[0,1:-1] = x[0,:]
            x_temp[-1,1:-1] = x[-1,:]
            # correct col
            x_temp[1:-1,0] = x[:,0]  
            x_temp[1:-1,-1] = x[:,-1]
            # correct 4 corners.
            x_temp[0,0] = x[0,0]
            x_temp[0,-1] = x[0,-1]
            x_temp[-1,0] = x[-1,0]
            x_temp[-1,-1] = x[-1,-1]
    if isBatch:
        temp_11 = x_temp[:,2:,1:-1]-2*x_temp[:,1:-1,1:-1]+x_temp[:,0:-2,1:-1]
        temp_22 = x_temp[:,1:-1,2:]-2*x_temp[:,1:-1,1:-1]+x_temp[:,1:-1,0:-2]
        temp_12 = 0.25*(x_temp[:,2:,2:]-x_temp[:,0:-2,2:]-x_temp[:,2:,0:-2]+x_temp[:,0:-2,0:-2])
    else:
        temp_11 = x_temp[2:,1:-1]-2*x_temp[1:-1,1:-1]+x_temp[0:-2,1:-1]
        temp_22 = x_temp[1:-1,2:]-2*x_temp[1:-1,1:-1]+x_temp[1:-1,0:-2]
        temp_12 = 0.25*(x_temp[2:,2:]-x_temp[0:-2,2:]-x_temp[2:,0:-2]+x_temp[0:-2,0:-2])
    return temp_11,temp_22,temp_12

def H_adj_Y(Y_11,Y_22,Y_12,isBatch=False,device = 'cpu'):#bnd='Dirchlet',
    if isBatch:
        numBatch,M,N = Y_11.size()
        Y_temp_11 = torch.zeros(numBatch,M+2,N+2,device=device)
        Y_temp_22 = torch.zeros(numBatch,M+2,N+2,device=device)
        Y_temp_12 = torch.zeros(numBatch,M+2,N+2,device=device)
        Y_temp_11[:,1:-1,1:-1] = Y_11
        Y_temp_22[:,1:-1,1:-1] = Y_22
        Y_temp_12[:,1:-1,1:-1] = Y_12
    else:
        M,N = Y_11.size()
        Y_temp_11 = torch.zeros(M+2,N+2,device=device)
        Y_temp_22 = torch.zeros(M+2,N+2,device=device)
        Y_temp_12 = torch.zeros(M+2,N+2,device=device)
        Y_temp_11[1:-1,1:-1] = Y_11
        Y_temp_22[1:-1,1:-1] = Y_22
        Y_temp_12[1:-1,1:-1] = Y_12

    if isBatch:
        x_temp_11 = Y_temp_11[:,2:,1:-1]-2*Y_temp_11[:,1:-1,1:-1]+Y_temp_11[:,0:-2,1:-1]
        x_temp_22 = Y_temp_22[:,1:-1,2:]-2*Y_temp_22[:,1:-1,1:-1]+Y_temp_22[:,1:-1,0:-2]
        x_temp_12 = 0.5*(Y_temp_12[:,2:,2:]-Y_temp_12[:,0:-2,2:]-Y_temp_12[:,2:,0:-2]+Y_temp_12[:,0:-2,0:-2])
        x = x_temp_11+x_temp_22+x_temp_12
    else:
        x_temp_11 = Y_temp_11[2:,1:-1]-2*Y_temp_11[1:-1,1:-1]+Y_temp_11[0:-2,1:-1]
        x_temp_22 = Y_temp_22[1:-1,2:]-2*Y_temp_22[1:-1,1:-1]+Y_temp_22[1:-1,0:-2]
        x_temp_12 = 0.5*(Y_temp_12[2:,2:]-Y_temp_12[0:-2,2:]-Y_temp_12[2:,0:-2]+Y_temp_12[0:-2,0:-2])
        x = x_temp_11+x_temp_22+x_temp_12
    return x

def HS_obj(x,Ax,y,TR_off,p=1,bnd='Dirchlet',device = 'cpu'):
    obj_data = 0.5*torch.norm(Ax(x)-y)**2
    temp_11,temp_22,temp_12 = Hx(torch.squeeze(x),bnd=bnd,device=device)
    _,_,M,N = x.size()
    obj_HS = 0
    temp_HS = torch.zeros(M,N,2,2,device=device)
    if p==float('inf'):
        temp_HS[:,:,0,0] = temp_11
        temp_HS[:,:,0,1] = temp_12
        temp_HS[:,:,1,0] = temp_12
        temp_HS[:,:,1,1] = temp_22
        S = torch.linalg.svdvals(temp_HS)
        obj_HS = torch.sum(S[:,:,0])
    elif p==2:
        temp_HS[:,:,0,0] = temp_11
        temp_HS[:,:,0,1] = temp_12
        temp_HS[:,:,1,0] = temp_12
        temp_HS[:,:,1,1] = temp_22
        obj_HS = torch.sum(torch.norm(temp_HS,dim=(2,3)))
    elif p==1:
        temp_HS[:,:,0,0] = temp_11
        temp_HS[:,:,0,1] = temp_12
        temp_HS[:,:,1,0] = temp_12
        temp_HS[:,:,1,1] = temp_22
        obj_HS = torch.sum(torch.norm(temp_HS,dim=(2,3),p='nuc'))
    else:
        temp_HS[:,:,0,0] = temp_11
        temp_HS[:,:,0,1] = temp_12
        temp_HS[:,:,1,0] = temp_12
        temp_HS[:,:,1,1] = temp_22
        S = torch.linalg.svdvals(temp_HS)
        S_p = S ** p
        sum_S_p = torch.sum(S_p,dim=(2,3))
        obj_HS = torch.sum(sum_S_p ** (1/p))
    obj = obj_data+TR_off*obj_HS
    return obj

def Prj_mixedl1lp(temp_11,temp_22,temp_12,p,device='cpu'):
    # obtain the projection of the mixed l1-lp norm
    M,N = temp_11.size()
    temp_HS = torch.zeros(M,N,2,2,device=device)
    temp_HS[:,:,0,0] = temp_11
    temp_HS[:,:,0,1] = temp_12
    temp_HS[:,:,1,0] = temp_12
    temp_HS[:,:,1,1] = temp_22
    if p==2:
        temp_norm = torch.norm(temp_HS,dim=(2,3))
        index = torch.where(temp_norm>1)
        temp_HS[index[0],index[1],:,:] =  temp_HS[index[0],index[1],:,:]/temp_norm[index[0],index[1]].unsqueeze(1).unsqueeze(1)
        temp_11 = temp_HS[:,:,0,0]
        temp_12 = temp_HS[:,:,0,1]
        temp_22 = temp_HS[:,:,1,1]
    elif p == float('inf'):
        U, S, Vh = torch.linalg.svd(temp_HS)
        S = torch.minimum(S,torch.ones_like(S))
        S_diag = torch.zeros_like(temp_HS)  # Create a tensor of zeros with the same shape as temp_HS
        min_dim = S.shape[-1]  # min(M, N)
        S_diag[..., :min_dim, :min_dim] = torch.diag_embed(S)
        temp = U@S_diag@Vh
        temp_11 = temp[:,:,0,0]
        temp_12 = temp[:,:,0,1]
        temp_22 = temp[:,:,1,1]
    elif p==1:
        U, S, Vh = torch.linalg.svd(temp_HS)
        gamma = torch.zeros_like(S)
        index = torch.where((1-S[:,:,1]<S[:,:,0]) & (S[:,:,0]<=1+S[:,:,1]))
        if index[0].numel()>0:
            gamma[index[0],index[1],0] = (S[index[0],index[1],0]+S[index[0],index[1],1]-1)/2
            gamma[index[0],index[1],1] = gamma[index[0],index[1],0]
        index = torch.where(S[:,:,0]>1+S[:,:,1])
        if index[0].numel()>0:
            gamma[index[0],index[1],0] = (S[index[0],index[1],0]-1)
            gamma[index[0],index[1],1] = gamma[index[0],index[1],0]
        S = torch.maximum(S-gamma,torch.zeros_like(S))
        S_diag = torch.zeros_like(temp_HS)  # Create a tensor of zeros with the same shape as temp_HS
        min_dim = S.shape[-1]  # min(M, N)
        S_diag[..., :min_dim, :min_dim] = torch.diag_embed(S)
        temp = U@S_diag@Vh
        temp_11 = temp[:,:,0,0]
        temp_12 = temp[:,:,0,1]
        temp_22 = temp[:,:,1,1]

    return temp_11,temp_22,temp_12

def WPMHSReg(v,TR_off,p = 2,Proj_C = lambda x:x,PreOp = lambda x:x, P_1 = None,P_2 = None,P_3 = None,\
                     Num_iter = 100,tol = 1e-7,bnd = 'Dirchlet',device = 'cpu'):
    # Compute the WPM. for the HS regularizer
    v = torch.squeeze(v)
    M,N = v.shape
    if P_1 is None:
        P_1 = torch.zeros(M,N,device=device)
        P_2 = torch.zeros(M,N,device=device)
        P_3 = torch.zeros(M,N,device=device)
        R_1 = torch.zeros(M,N,device=device) # 11
        R_2 = torch.zeros(M,N,device=device) # 22
        R_3 = torch.zeros(M,N,device=device) # 12
    else:
        R_1 = P_1
        R_2 = P_2
        R_3 = P_3
    t_k_1 = 1
    x_out = torch.zeros(M,N,device=device)

    for iter in range(Num_iter):
        x_old = x_out
        P_1_old = P_1
        P_2_old = P_2
        P_3_old = P_3
        temp = H_adj_Y(R_1,R_2,R_3,device = device) # this one only implement the Dirchlet boundary condition
        x_out = Proj_C(v-TR_off*torch.squeeze(PreOp(temp)))
        re = torch.norm(x_old-x_out)/torch.norm(x_out)
        if re<tol or iter == Num_iter-1:
            break 
        Q_1,Q_2,Q_3 = Hx(x_out,bnd=bnd,device=device)
        P_1 = R_1+1/(64*TR_off)*Q_1
        P_2 = R_2+1/(64*TR_off)*Q_2
        P_3 = R_3+1/(64*TR_off)*Q_3
        #perform project step
        P_1,P_2,P_3 = Prj_mixedl1lp(P_1,P_2,P_3,p=p,device=device)
        t_k = t_k_1
        t_k_1 = (1+np.sqrt(1+4*t_k**2))/2
        R_1 = P_1 + ((t_k-1)/t_k_1)*(P_1-P_1_old)
        R_2 = P_2 + ((t_k-1)/t_k_1)*(P_2-P_2_old)
        R_3 = P_3 + ((t_k-1)/t_k_1)*(P_3-P_3_old)
    return x_out.unsqueeze(0).unsqueeze(0),P_1,P_2,P_3


def FISTA_HS(num_iters, Ax, ATx,b,L,TR_off,p=2,Proj_C = lambda x: x,x_ini=None,Num_iter = 100,P = lambda x: x,L_P_inv = 1,bnd = 'Dirchlet',\
             save=None, verbose = True,original= None,SaveIter=True,device='cpu'):
  """
  Unconstrained Optimization.
  Solve the optimization problem with FISTA:

    \min_{x\in C} \frac{1}{2} \| A x - b \|_2^2 + TR_off*HS(x)
    
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
    bound: the boundary condition.
  Returns:
    x (Array): Reconstruction.
  """
  AHb = ATx(b)
  if x_ini==None:
      x = AHb #torch.zeros_like(AHb) #
  else:
      x = x_ini
  z = x.clone()
  lst_time  = []
  lst_cost = []
  lst_psnr = []
  lst_ssim = []
  #lst_mse = []
  P_1 = None # 11
  P_2 = None # 22
  P_3 = None # 12
  if verbose:
      pbar = tqdm(total=num_iters, desc="FISTA_HS", \
                leave=True)
  lst_cost.append(HS_obj(x,Ax,b,TR_off,p=p,bnd=bnd,device = device).cpu().numpy())
  lst_time.append(0)
  if original is not None:
      lst_psnr.append(utl.PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
      lst_ssim.append(utl.SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
  t_k_1 = 1
  for k in range(num_iters):
      start_time = time.perf_counter()
      x_old = x.clone()
      gr = P(ATx((Ax(z) - b)))/L
      temp_grad = z-gr
      x,P_1,P_2,P_3 = WPMHSReg(temp_grad,(L_P_inv*TR_off)/L,p = p,Proj_C = Proj_C,PreOp = P, P_1 = P_1,P_2 = P_2,P_3 = P_3,\
                     Num_iter = Num_iter,bnd = bnd,device = device)
      t_k = t_k_1
      t_k_1 = (1+np.sqrt(1+4*t_k**2))/2
      #t_k = 1
      z = x + ((t_k-1)/t_k_1)*(x - x_old)
      end_time = time.perf_counter()
      lst_time.append(end_time - start_time)
      if original is not None:
          lst_psnr.append(utl.PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
          lst_ssim.append(utl.SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
      lst_cost.append(HS_obj(x,Ax,b,TR_off,p=p,bnd=bnd,device = device).cpu().numpy())
      if save != None:
          if original is not None:
              Dict = {'lst_time':np.cumsum(lst_time),'lst_cost':lst_cost,'lst_psnr':lst_psnr,'lst_ssim':lst_ssim,'im':torch.squeeze(torch.abs(x)).cpu().numpy()}
              scipy.io.savemat("%s/TimeCost.mat" % save,Dict)
          else:
              Dict = {'lst_time':np.cumsum(lst_time),'lst_cost':lst_cost,'im':torch.squeeze(torch.abs(x)).cpu().numpy()}
              scipy.io.savemat("%s/TimeCost.mat" % save,Dict)
          if SaveIter:
              Dict_Im = {'im':torch.squeeze(x).cpu().numpy(),'im':torch.squeeze(torch.abs(x)).cpu().numpy()}#np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())
              scipy.io.savemat("%s/iter_%03d.mat" % (save, k),Dict_Im)
      if verbose:
          pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
          pbar.update()
          pbar.refresh()
  if verbose:
      pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
      pbar.close()
  return x,lst_cost,lst_psnr,lst_ssim,np.cumsum(lst_time)

def mulAX(X,A,ATx,rho,bnd,device='cpu'):
    Y = ATx(A(X))
    Z_11,Z_22,Z_12 = Hx(torch.squeeze(X),bnd=bnd,device=device)
    Y = Y+rho*(H_adj_Y(Z_11,Z_22,Z_12,device=device).unsqueeze(0).unsqueeze(0))
    return Y

def CG_Alg(x,RHS,A,ATx,rho,bnd,P,MaxCG_Iter=300,tol=1e-4,device='cpu'):
    r_k = RHS-mulAX(x,A,ATx,rho,bnd,device=device)
    count_iter = 2
    z_k = P(r_k)
    for _ in range(MaxCG_Iter):
        p_k = r_k
        Ap_k = mulAX(p_k,A,ATx,rho,bnd,device=device)
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

def Proj_nonegative(v,U=None,MaxIter=30,tol=1e-6,device='cpu'):
    # run semi-smooth Newton method to solve 
    # \min_x>0 0.5*\|x-v\|^2_P, P=I+UU^T -- *** not well tested ***
    v = torch.squeeze(v)
    M,N = v.size()
    Proj_Orth = lambda x: torch.maximum(x,torch.zeros_like(x))
    if U==None:
        x = Proj_Orth(v)
    else:
        _,K = U.size()
        gamma = torch.zeros(K,device = device)
        Eval_grad = lambda x: U.t()@(v.flatten()-Proj_Orth(v.flatten()-U@x))+x
        count = 0
        while True:
            count = count+1
            grad = Eval_grad(gamma)
            if count>MaxIter or torch.norm(grad)<tol:
                break
            temp = Proj_Orth(v.flatten()-U@gamma)
            index = temp==0
            U_temp = U
            U_temp[index,:] = 0
            Jaco_eval = U_temp.t()@U_temp+torch.eye(K,device=device)
            gamma = gamma-torch.linalg.inv(Jaco_eval)@grad
        x = torch.reshape(Proj_Orth(v.flatten()-U@gamma),(M,N))
    return x.unsqueeze(0).unsqueeze(0)