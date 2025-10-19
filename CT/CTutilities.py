# Tao edited May/2024
# Tao Hong, Zhaoyi Xu, Jason Hu, and Jeffrey A. Fessler, 
# ``Using Randomized Nyström Preconditioners to Accelerate Variational Image Reconstruction'', To appear in IEEE Transactions on Computational Imaging, arXiv:2411.08178, 2025.

import numpy as np
import odl
from operator2 import *
import torch 
import utilities as utl
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class parbeam(object):
    def __init__(self, sigSize, numAngles, numDetector,
                 min_pt=[-20, -20], max_pt=[20, 20], 
                 y_area=[-40, 40], Hf=None):#-30, 30
        self.sigSize = sigSize # 
        self.numAngles = numAngles
        self.numDetector = numDetector
        self.Hf = Hf
        self.name = 'parbeam'
        
        self.y_area = y_area
        self.reco_space = odl.uniform_discr(min_pt=min_pt, max_pt=max_pt, shape=sigSize, dtype='float32')
        self.angle_partition = odl.uniform_partition(0, np.pi, numAngles)
        self.detector_partition = odl.uniform_partition(self.y_area[0], self.y_area[1], self.numDetector)
            
        self.geometry = odl.tomo.Parallel2dGeometry(self.angle_partition, self.detector_partition)
        #self.geometry = odl.tomo.FanBeamGeometry(self.angle_partition, self.detector_partition, src_radius=40, det_radius=40)
        # A operator
        self.A = odl.tomo.RayTransform(self.reco_space, self.geometry, impl='astra_cuda')
        
        # Fourier transform in detector direction
        #self.fourier = odl.trafos.FourierTransform(self.A.range, axes=[1])

        # Create ramp in the detector direction
        #self.ramp_function = self.fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))

        # Create ramp filter via the convolution formula with fourier transforms
        #self.ramp_filter = self.fourier.inverse * self.ramp_function * self.fourier

        # Create filtered back-projection by composing the back-projection (adjoint)
        # with the ramp filter.
        #self.fbp = self.A.adjoint * self.ramp_filter
        #self.fbp = odl.tomo.fbp_op(self.A, filter_type='Shepp-Logan', frequency_scaling=0.8)

        self.fbp = odl.tomo.fbp_op(self.A, filter_type='Hann', frequency_scaling=0.8)
        #phantom = odl.phantom.shepp_logan(self.reco_space, modified=True)
        #y = self.A(phantom)
        #yT = self.A.adjoint(y)

    def grad(self, x, y):
        if self.Hf is None: # AT*(Ax - y)
            Ax = self.fmult(x, self.A)
            grad_d = self.ftran(Ax - y, self.A)
        else: # AT*HT*H*(Ax - y)
            pass
        return grad_d
    
    def eval(self, x, y):
        if self.Hf is None:
            Ax = self.fmult(x, self.A)
            d = torch.square(Ax - y)
            d = 1 / (2 * d.shape[0]) * torch.sum(d)
        else:
            pass
        return d

    def fmult(self, x, A): # x: B X Y Z
        Ax = OperatorFunction.apply(self.A, x)
        return Ax

    def ftran(self, z, A):# z: B A H W
        ATx = OperatorFunction.apply(self.A.adjoint, z)
        return ATx
    
    def Adagger(self, y, method='fbp', freq=0.3): # y: A H W
        if method == 'fbp':
            fbp_op = odl.tomo.fbp_op(self.A, filter_type='Hann', frequency_scaling=freq)
            x_init = OperatorFunction.apply(fbp_op, y)
        return x_init 
    
    def Atimes(self, x): #return A*x
        return self.fmult(x, self.A)
    
    def ATtimes(self, y): #return AT*x
        return self.ftran(y, self.A)
    def fbp(self,y):
        return self.fbp(y)

class fanbeam(object):
    def __init__(self, sigSize, numAngles, numDetector,
                 min_pt=[-20, -20], max_pt=[20, 20], 
                 y_area=[-60, 60], Hf=None):
        self.sigSize = sigSize # 
        self.numAngles = numAngles
        self.numDetector = numDetector
        self.Hf = Hf
        self.name = 'fanbeam'
        self.y_area = y_area
        self.reco_space = odl.uniform_discr(min_pt=min_pt, max_pt=max_pt, shape=sigSize, dtype='float32')
        self.angle_partition = odl.uniform_partition(0, 2*np.pi, numAngles)
        self.detector_partition = odl.uniform_partition(self.y_area[0], self.y_area[1], self.numDetector)
        # large fan angle
        self.geometry = odl.tomo.FanBeamGeometry(self.angle_partition, self.detector_partition, src_radius=40, det_radius=40)
        # A operator
        self.A = odl.tomo.RayTransform(self.reco_space, self.geometry, impl='astra_cuda')
        self.fbp = odl.tomo.fbp_op(self.A, filter_type='Hann', frequency_scaling=0.8)

    def grad(self, x, y):
        if self.Hf is None: # AT*(Ax - y)
            Ax = self.fmult(x, self.A)
            grad_d = self.ftran(Ax - y, self.A)
        else: # AT*HT*H*(Ax - y)
            pass
        return grad_d
    
    def eval(self, x, y):
        if self.Hf is None:
            Ax = self.fmult(x, self.A)
            d = torch.square(Ax - y)
            d = 1 / (2 * d.shape[0]) * torch.sum(d)
        else:
            pass
        return d

    def fmult(self, x, A): # x: B X Y Z
        Ax = OperatorFunction.apply(self.A, x)
        return Ax

    def ftran(self, z, A):# z: B A H W
        ATx = OperatorFunction.apply(self.A.adjoint, z)
        return ATx
    
    def Adagger(self, y, method='fbp', freq=0.3): # y: A H W
        if method == 'fbp':
            fbp_op = odl.tomo.fbp_op(self.A, filter_type='Hann', frequency_scaling=freq)
            x_init = OperatorFunction.apply(fbp_op, y)
        return x_init 
    
    def Atimes(self, x): #return A*x
        return self.fmult(x, self.A)
    
    def ATtimes(self, y): #return A*x
        return self.ftran(y, self.A)
    def fbp(self,y):
        return self.fbp(y)

def PCG_iter(A,b,x0,obj_cal,im_original,MaxIter=100,epsi=1e-6,P_inv=lambda x:x):
    r0 = b-A(x0)
    z0 = P_inv(r0)
    p0 = z0
    resi = []
    obj_set = []
    psnr_set = []
    resi.append(torch.norm(b - A(x0)).cpu().numpy())
    obj_set.append(obj_cal(x0).cpu().numpy())
    psnr_set.append(psnr(torch.squeeze(im_original).cpu().numpy(),torch.squeeze(x0).cpu().numpy()))
    for iter in range(MaxIter):
        v = A(p0)
        alpha = (r0.flatten().t()@z0.flatten())/(p0.flatten().t()@v.flatten())
        x = x0+alpha*p0
        r = r0-alpha*v
        #err = torch.norm(r).cpu().numpy()
        err = torch.norm(b - A(x0)).cpu().numpy()
        resi.append(err)
        obj_set.append(obj_cal(x0).cpu().numpy())
        psnr_set.append(psnr(torch.squeeze(im_original).cpu().numpy(), torch.squeeze(x0).cpu().numpy()))
        if err<epsi:
            break
        z = P_inv(r)
        beta = (r.flatten().t()@z.flatten())/(r0.flatten().t()@z0.flatten())
        x0 = x 
        r0 = r
        p0 = z+beta*p0 
        z0 = z
    return x,resi,obj_set,psnr_set

def CG_Alg_Handle(x_k,RHS,A,MaxCG_Iter,tol=1e-6):
    r_k = RHS - A(x_k)
    p_k = r_k
    resi = []
    for iter in range(MaxCG_Iter):
        Ap_k = A(p_k)
        alpha_k = torch.vdot(r_k.flatten(),r_k.flatten())/torch.vdot(p_k.flatten(),Ap_k.flatten())
        x_k_1 = x_k+alpha_k*p_k
        if iter<MaxCG_Iter:
            r_k_1 = r_k - alpha_k*A(p_k)
            resi.append(torch.norm(r_k_1).cpu().numpy())
            if torch.norm(r_k_1)<tol:
                break
            beta_k = torch.vdot(r_k_1.flatten(),r_k_1.flatten())/torch.vdot(r_k.flatten(),r_k.flatten())
            p_k_1 = r_k_1+beta_k*p_k
            p_k = p_k_1
            r_k = r_k_1
            x_k = x_k_1
    return x_k_1,resi

def Build_Sketch_Pred(Ax,ATx,im_size,im_size_prod,sketch_size,isHS=False,isBatch=True,W=None,Lx = lambda x:x,LTx = lambda x:x,TR_off=0,device='cpu',dim=2):
    #sketch_size = sketch_size+1#modified the last element 1st Aug. 2024 // to delete the last column
    epsi = 1e-10*np.sqrt(im_size_prod)
    if dim==2:
        Omega = torch.randn(sketch_size,1,im_size,im_size,device=device)/np.sqrt(im_size_prod)
    elif dim==3:
        Omega = torch.randn(sketch_size,im_size,im_size,im_size,device=device)/np.sqrt(im_size_prod)
    if isBatch:
        if TR_off!=0:
            if W==None:
                if isHS:
                    Lxtemp_1,Lxtemp_2,Lxtemp_3 = Lx(Omega)
                    Y = ATx(Ax(Omega))+TR_off*LTx(Lxtemp_1,Lxtemp_2,Lxtemp_3)
                else:
                    Lxtemp_1,Lxtemp_2 = Lx(Omega)
                    Y = ATx(Ax(Omega))+TR_off*LTx(Lxtemp_1,Lxtemp_2)
            else:
                if isHS:
                    Lxtemp_1,Lxtemp_2,Lxtemp_3 = Lx(Omega)
                    Y = ATx(W*(Ax(Omega)))+TR_off*LTx(Lxtemp_1,Lxtemp_2,Lxtemp_3)
                else:
                    Lxtemp_1,Lxtemp_2 = Lx(Omega)
                    Y = ATx(W*(Ax(Omega)))+TR_off*LTx(Lxtemp_1,Lxtemp_2)
        else:
            if W==None:
                Y = ATx(Ax(Omega))
            else:
                Y = ATx(W*(Ax(Omega)))
    else:
        if TR_off!=0:
            if W==None:
                if isHS:
                    Lxtemp_1,Lxtemp_2,Lxtemp_3 = Lx(Omega[0,:,:,:])
                    Y_temp = ATx(Ax(Omega[0,:,:,:]))+TR_off*LTx(Lxtemp_1,Lxtemp_2,Lxtemp_3)
                else:
                    Lxtemp_1,Lxtemp_2 = Lx(Omega[0,:,:,:])
                    Y_temp = ATx(Ax(Omega[0,:,:,:]))+TR_off*LTx(Lxtemp_1,Lxtemp_2)
            else:
                if isHS:
                    Lxtemp_1,Lxtemp_2,Lxtemp_3 = Lx(Omega[0,:,:,:])
                    Y_temp = ATx(W*Ax(Omega[0,:,:,:]))+TR_off*LTx(Lxtemp_1,Lxtemp_2,Lxtemp_3)
                else:
                    Lxtemp_1,Lxtemp_2 = Lx(Omega[0,:,:,:])
                    Y_temp = ATx(W*Ax(Omega[0,:,:,:]))+TR_off*LTx(Lxtemp_1,Lxtemp_2)
            size_1,size_2,size_3 = Y_temp.size()
            Y = torch.zeros(sketch_size,size_1,size_2,size_3,device=device)
            Y[0,:,:,:] = Y_temp
            if W==None:
                for sketch_iter in range(sketch_size-1):
                    if isHS:
                        Lxtemp_1,Lxtemp_2,Lxtemp_3 = Lx(Omega[0,:,:,:])
                        Y_temp = ATx(Ax(Omega[sketch_iter+1,:,:,:]))+TR_off*LTx(Lxtemp_1,Lxtemp_2,Lxtemp_3)
                    else:
                        Lxtemp_1,Lxtemp_2 = Lx(Omega[0,:,:,:])
                        Y_temp = ATx(Ax(Omega[sketch_iter+1,:,:,:]))+TR_off*LTx(Lxtemp_1,Lxtemp_2)
                    Y[sketch_iter+1,:,:,:] = Y_temp
            else:
                for sketch_iter in range(sketch_size-1):
                    if isHS:
                        Lxtemp_1,Lxtemp_2,Lxtemp_3 = Lx(Omega[0,:,:,:])
                        Y_temp = ATx(W*Ax(Omega[sketch_iter+1,:,:,:]))+TR_off*LTx(Lxtemp_1,Lxtemp_2,Lxtemp_3)
                    else:
                        Lxtemp_1,Lxtemp_2 = Lx(Omega[0,:,:,:])
                        Y_temp = ATx(W*Ax(Omega[sketch_iter+1,:,:,:]))+TR_off*LTx(Lxtemp_1,Lxtemp_2)
                    Y[sketch_iter+1,:,:,:] = Y_temp
        else:
            if W==None:
                Y_temp = ATx(Ax(Omega[0,:,:,:]))
            else:
                Y_temp = ATx(W*Ax(Omega[0,:,:,:]))
            _,size_1,size_2,size_3 = Y_temp.size()
            Y = torch.zeros(sketch_size,size_1,size_2,size_3,device=device)
            Y[0,:,:,:] = Y_temp
            if W==None:
                for sketch_iter in range(sketch_size-1):
                    Y_temp = ATx(Ax(Omega[sketch_iter+1,:,:,:]))
                    Y[sketch_iter+1,:,:,:] = Y_temp
            else:
                for sketch_iter in range(sketch_size-1):
                    Y_temp = ATx(W*Ax(Omega[sketch_iter+1,:,:,:]))
                    Y[sketch_iter+1,:,:,:] = Y_temp

    if dim==2:
        Omega = torch.reshape(torch.permute(Omega,(2,3,1,0)),(im_size_prod,sketch_size))
        Y = torch.reshape(torch.permute(Y,(2,3,1,0)),(im_size_prod,sketch_size))
    elif dim==3:
        Omega = torch.reshape(torch.permute(Omega,(1,2,3,0)),(im_size_prod,sketch_size))
        Y = torch.reshape(torch.permute(Y,(1,2,3,0)),(im_size_prod,sketch_size))
    Y_corr = Y+epsi*Omega
    L = torch.linalg.cholesky(Omega.t()@Y_corr)
    B = torch.linalg.solve_triangular(L,Y_corr.t(),upper=False).t()
    USV = torch.linalg.svd(B,full_matrices=False)
    S = USV.S
    lambda_l = S[-1]
    S = (torch.maximum(torch.zeros_like(S),S**2-epsi))#torch.diag
    U = USV.U
    return U,S,lambda_l

def Build_Sketch_Wav_Pred(Ax,ATx,Wx,WTx,im_size,list_coeff_size,im_size_prod,sketch_size,num_level,TR_off=0,device='cpu',dim=2):
    #sketch_size = sketch_size+1#modified the last element 1st Aug. 2024 // to delete the last column
    epsi = 1e-10*np.sqrt(im_size_prod)
    Omega = torch.randn(sketch_size,1,im_size,im_size,device=device)/np.sqrt(im_size_prod)
    
    if TR_off!=0:
        Y = utl.wavArray2vec(Wx(ATx(Ax(WTx(utl.vec2wavArray(Omega,num_level,list_coeff_size))))),num_level,im_size,list_coeff_size,device=device)+TR_off*Omega
    else:
        Y = utl.wavArray2vec(Wx(ATx(Ax(WTx(utl.vec2wavArray(Omega,num_level,list_coeff_size))))),num_level,im_size,list_coeff_size,device=device)
    Omega = torch.reshape(torch.permute(Omega,(2,3,1,0)),(im_size_prod,sketch_size))
    Y = torch.reshape(torch.permute(Y,(2,3,1,0)),(im_size_prod,sketch_size))
    Y_corr = Y+epsi*Omega
    L = torch.linalg.cholesky(Omega.t()@Y_corr)
    B = torch.linalg.solve_triangular(L,Y_corr.t(),upper=False).t()
    USV = torch.linalg.svd(B,full_matrices=False)
    S = USV.S
    lambda_l = S[-1]
    S = (torch.maximum(torch.zeros_like(S),S**2-epsi))#torch.diag
    U = USV.U
    return U,S,lambda_l

def P_invx(x,Lambda_inv,U,par_fix,im_size,dim=2):
    x = torch.squeeze(x)
    temp = U.t()@x.flatten()
    if dim==2:
        y = (U@(par_fix*(Lambda_inv@temp)-temp) + x.flatten()).view([1, 1, im_size, im_size])
    elif dim==3:
        y = (U@(par_fix*(Lambda_inv@temp)-temp) + x.flatten()).view([1, im_size, im_size, im_size])
    return y

def P_invx_Simp(x,U,im_size):
    y = x-(U@(U.t()@torch.squeeze(x).flatten())).view(1, 1, im_size, im_size)
    return y

def Build_Sketch_Real_Pred(Ax,ATx,im_size,im_size_prod,sketch_size,isHS=False,isBatch=True,W=None,Lx = lambda x:x,LTx = lambda x:x,TR_off=0,device='cpu',dim=2):
    #sketch_size = sketch_size+1#modified the last element 1st Aug. 2024 // to delete the last column
    epsi = 1e-10*np.sqrt(im_size_prod)
    Omega = torch.randn(sketch_size,im_size,im_size,device=device)/np.sqrt(im_size_prod)
    Y_temp = ATx(Ax(Omega[0,:,:]))
    _,_,size_1,size_2 = Y_temp.size()
    Y = torch.zeros(sketch_size,size_1,size_2,device=device)
    Y[0,:,:] = torch.squeeze(Y_temp)
    for sketch_iter in range(sketch_size-1):
        Y_temp = ATx(Ax(Omega[sketch_iter+1,:,:]))
        Y[sketch_iter+1,:,:] = torch.squeeze(Y_temp)
    if dim==2:
        Omega = torch.reshape(torch.permute(Omega,(2,1,0)),(im_size_prod,sketch_size))
        Y = torch.reshape(torch.permute(Y,(2,1,0)),(im_size_prod,sketch_size))
    Y_corr = Y+epsi*Omega
    L = torch.linalg.cholesky(Omega.t()@Y_corr)
    B = torch.linalg.solve_triangular(L,Y_corr.t(),upper=False).t()
    USV = torch.linalg.svd(B,full_matrices=False)
    S = USV.S
    lambda_l = S[-1]
    S = (torch.maximum(torch.zeros_like(S),S**2-epsi))#torch.diag
    U = USV.U
    return U,S,lambda_l

def P_invx_SimpReal(x,U,im_size):
    y = x-torch.reshape(U@(U.t()@torch.reshape(torch.squeeze(x).t(),(im_size*im_size,1))),(im_size,im_size)).t().unsqueeze(0).unsqueeze(0)
    return y

def Px_Simp(x,U,im_size):
    y = x+(U@(U.t()@torch.squeeze(x).flatten())).view(1, 1, im_size, im_size)
    return y

def P_invx_SimpBatch(x,U_batched,im_size):
    im_squeeze = torch.squeeze(x)
    Ux = torch.einsum('rhw,hw->r', U_batched, im_squeeze)
    UUx = torch.einsum('rhw,r->hw', U_batched, Ux)
    y = x - UUx  
    return y.view(1, 1, im_size, im_size)


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
    else:
        data_range = 255.0
    return psnr(original,compressed,\
                data_range=data_range)

# def SSIM(original,compressed):
#     return ssim(original,compressed,\
#                 data_range=compressed.max() - compressed.min())

# def PSNR(original, compressed):
#     mse = np.mean((np.abs(original - compressed)) ** 2)
#     if(mse == 0):
#         return 100
#     # decide the scale of the image
#     if np.max(np.abs(original))<1.01:
#         max_pixel = 1
#     else:
#         max_pixel = 255.0
#     psnr = 20 * np.log10(max_pixel/np.sqrt(mse))
#     return psnr
