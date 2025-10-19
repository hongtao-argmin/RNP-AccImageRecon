#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Tao Hong, Zhaoyi Xu, Jason Hu, and Jeffrey A. Fessler, 
# ``Using Randomized Nyström Preconditioners to Accelerate Variational Image Reconstruction'', To appear in IEEE Transactions on Computational Imaging, arXiv:2411.08178, 2025.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import math
import cv2

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
    for iter in range(MaxIter):
        temp_grad = z - alpha_p*P(z-v)
        x = torch.sign(temp_grad) * torch.clamp(torch.abs(temp_grad) - TR*alpha_p, min=0)
        if torch.norm(x-x_old)<tol:
            break
        t_k = t_k_1
        t_k_1 = (1+np.sqrt(1+4*t_k**2))/2
        z = x + ((t_k-1)/t_k_1)*(x - x_old)
        x_old = x
    return x

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

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob,isColor=False):
    """
    Add salt and pepper noise to an image.
    
    Args:
        image (Tensor): Input image.
        salt_prob (float): Probability of adding salt noise.
        pepper_prob (float): Probability of adding pepper noise.
    
    Returns:
        Tensor: Noisy image.
    """
    
    noisy_image = image.clone()
    device = image.device
    if isColor:
        _,_,H, W,_ = noisy_image.shape
        # Add salt noise
        num_salt = int(salt_prob * H * W)
        torch.manual_seed(10)
        coords_salt = [torch.randint(0, i - 1, (num_salt,)) for i in (H, W)]
        noisy_image[:,:, coords_salt[0], coords_salt[1],:] = 1.0  # Salt noise is white (1.0)
        # Add pepper noise
        num_pepper = int(pepper_prob * H * W)
        torch.manual_seed(20)
        coords_pepper = [torch.randint(0, i - 1, (num_pepper,)) for i in (H, W)]
        noisy_image[:, :,coords_pepper[0], coords_pepper[1],:] = 0.0
        input_image = (torch.squeeze(noisy_image)).cpu().numpy()
        imgYCC = cv2.cvtColor(input_image, cv2.COLOR_RGB2YCrCb)
        # apply median filter
        padded_image = cv2.copyMakeBorder(imgYCC, 4, 4, 4, 4, borderType=cv2.BORDER_REPLICATE) #cv2.BORDER_REFLECT
        median_filtered_padded = cv2.medianBlur(padded_image, ksize=5)
        imgYCC[:,:,1::1] = median_filtered_padded[4:-4, 4:-4,1::1]
        noisy_image_filter = cv2.cvtColor(imgYCC, cv2.COLOR_YCrCb2RGB)
        noisy_image_filter = torch.from_numpy(noisy_image_filter).unsqueeze(0).unsqueeze(0)
        noisy_image_filter = noisy_image_filter.to(device)
    else:
        # Generate salt noise
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        salt_mask = torch.rand_like(noisy_image) < salt_prob
        noisy_image[salt_mask] = 1.0
        # Generate pepper noise
        torch.manual_seed(20)
        torch.cuda.manual_seed(20)
        pepper_mask = torch.rand_like(noisy_image) < pepper_prob
        noisy_image[pepper_mask] = 0.0
        noisy_image_filter = noisy_image

    return noisy_image,noisy_image_filter


def gaussian_kernel(size: int, sigma: float):
    """Generates a Gaussian kernel."""
    kernel = torch.tensor([
        [math.exp(-(x - size // 2) ** 2 / (2 * sigma ** 2) - (y - size // 2) ** 2 / (2 * sigma ** 2))
         for x in range(size)] for y in range(size)])
    return kernel.unsqueeze(0).unsqueeze(0) / kernel.sum()


def utl_imshow(im_noisy,im_reco,im_GT,cmap_type='gray',vmin=0,vmax=1):
    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plt.imshow(im_noisy,cmap=cmap_type,vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.grid()
    plt.title('im-Noisy')
    plt.subplot(1,3,2)
    plt.imshow(im_reco,cmap=cmap_type,vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.grid()
    plt.title('im-Reco.')
    plt.subplot(1,3,3)
    plt.imshow(im_GT,cmap=cmap_type,vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.grid()
    plt.title('GT')
    plt.show()
    return

# Define the convolutional layer with the uniform blur kernel
class Blur(nn.Module):
    def __init__(self,kernel,kernel_size,boundary=None):
        super(Blur, self).__init__()
        self.kernel_size = kernel_size
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        if boundary==None:
            self.boundary = 'reflect'
        else:
            self.boundary = boundary
    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.kernel_size//2, padding_mode=self.boundary,bias=None)

# Define the transpose convolutional layer (adjoint operator)
class AdjointBlur(nn.Module):
    def __init__(self, kernel,kernel_size,boundary=None):
        super(AdjointBlur, self).__init__()
        self.kernel_size = kernel_size
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        if boundary ==None:
            self.boundary = 'reflect'
        else:
            self.boundary = boundary
    def forward(self, x):
        return F.conv_transpose2d(x, self.kernel, padding=self.kernel_size//2, padding_mode=self.boundary,bias=None)

def PrepareImage(input_image):
    '''
    input: pytorch tensor RGB. size: [numBatch,1,im_size,im_size,colorchannel]; range: [0,1]
    output: pytorch tensor illumination channel. [numBatch,1,im_size,im_size]; range: [0,1]
    return the luma channel
    '''
    input_image = (torch.squeeze(input_image)).cpu().numpy()
    imgYCC = cv2.cvtColor(input_image, cv2.COLOR_RGB2YCrCb)
    luma_image = imgYCC[:,:,0]
    luma_image = torch.from_numpy(luma_image)
    return luma_image.unsqueeze(0).unsqueeze(0)

def MergeChannels(in_im,enhanced_im):
    '''
    input: pytorch tensor RGB. range: [0,1]; 
    in_im size: [numBatch,1,im_size,im_size,colorchannel]; 
    enhanced_im size: [numBatch,Channel,im_size,im_size];
    output: pytorch tensor RGB but with illmination enhanced channel;  range: [0,1]
    [numBatch,1,im_size,im_size,colorchannel]; 
    '''
    in_im = torch.squeeze(in_im).cpu().numpy()

    out_im = cv2.cvtColor(in_im, cv2.COLOR_RGB2YCrCb)
    out_im[:, :, 0] = torch.squeeze(enhanced_im).cpu().numpy()
    out_im = cv2.cvtColor(out_im, cv2.COLOR_YCrCb2RGB)
    out_im = torch.from_numpy(out_im).clamp(0,1)
    return out_im.unsqueeze(0).unsqueeze(0)

def crop_to_scale(image, scale,isColor=False):
    """
    Crop the image tensor to make its dimensions divisible by the given scale.
    
    Parameters:
    image (torch.Tensor): Input image tensor of shape (Batch,C, H, W,[ColorChannel]).
    scale (int): The scale factor to make the dimensions divisible by.
    
    Returns:
    torch.Tensor: Cropped image tensor.
    """
    if isColor:
        _, _,h, w,_ = image.shape
    else:
        _, _,h, w = image.shape
    new_h = h - (h % scale)
    new_w = w - (w % scale)
    if isColor:
        cropped_image = image[:, :, :new_h, :new_w, :]
    else:
        cropped_image = image[:, :,:new_h, :new_w]
    return cropped_image

def crop_to_scale_Numpy(image, scale, isColor=False):
    """
    Crop the image tensor to make its dimensions divisible by the given scale.

    Parameters:
    image (torch.Tensor): Input image tensor of shape (H, W,[ColorChannel]).
    scale (int): The scale factor to make the dimensions divisible by.

    Returns:
    torch.Tensor: Cropped image tensor.
    """
    if isColor:
        h, w, _ = image.shape
    else:
        h, w = image.shape
    new_h = h - (h % scale)
    new_w = w - (w % scale)
    if isColor:
        cropped_image = image[:new_h, :new_w, :]
    else:
        cropped_image = image[:new_h, :new_w]
    return cropped_image
