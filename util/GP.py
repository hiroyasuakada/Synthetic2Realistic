import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


import numpy as np
import random
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from numpy.linalg import inv
from sklearn.metrics.pairwise import cosine_similarity
import math

from torch import inverse


def kronecker(A, B):
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.view(A.size(0)*B.size(0), A.size(1)*B.size(1))
    return AB

# # torch version of Linear kernel function 
# def kernel_linear(X_u,X_l,vector_length,Num_A,Num_B):   
#     x_l_t =  X_l.repeat(Num_A,1)
#     x_u_t =  X_u.repeat(1,Num_B)
#     x_u_t = x_u_t.view(Num_A*Num_B,vector_length)
#     ker_t = torch.nn.functional.cosine_similarity(x_u_t,x_l_t)
#     ker_t = ker_t.view(Num_A,Num_B)
#     return ker_t

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    distt  =torch.clamp(dist, 0.0, np.inf)
    dist[dist != dist] = 0
    return dist

# # torch version of Squared Exponential kernel function 
# def kernel_se(x,y,var):
#     sigma_1 = 1.0
#     pw = 0.6
#     l_1 = var.max(axis=-1).max(axis=-1).max(axis=-1)#1.0#(np.sum(mu**2))**(pw)
#     d = pairwise_distances(x,y)
#     Ker = sigma_1**2 *torch.exp(-0.5*d/l_1**2)
#     return Ker

# # numpy version of Squared Exponential kernel function 
# def kernel_se_np(x,y,var):
#     sigma_1 = 1.0
#     pw = 0.6
#     l_1 = var.max(axis=-1).max(axis=-1).max(axis=-1)#1.0#(np.sum(mu**2))**(pw)
#     d = cdist(x,y)**2
#     Ker = sigma_1**2 * np.exp(-0.5*d/l_1**2)
#     return Ker

# # torch version of Rational Quadratic kernel function
# def kernel_rq(x,y,var,alpha=0.5):
#     sigma_1 = 1.0
#     pw = 0.6
#     l_1 = var.max(axis=-1).max(axis=-1).max(axis=-1)#1.0#(np.sum(mu**2))**(pw)
#     d = pairwise_distances(x,y)
#     Ker = sigma_1**2 *(1+(0.5*d/(alpha*l_1**2)))**(-1*alpha)
#     return Ker

# # numpy version of Rational Quadratic kernel function 
# def kernel_rq_np(x,y,var,alpha=0.5):
#     sigma_1 = 1.0
#     pw = 0.6
#     l_1 = var.max(axis=-1).max(axis=-1).max(axis=-1)#1.0#(np.sum(mu**2))**(pw)
#     d = cdist(x,y)**2
#     Ker = sigma_1**2 * (1+(0.5*d/(alpha*l_1**2)))**(-1*alpha)
#     return Ker

########################################################################################################

# torch version of Squared Exponential kernel function 
def kernel_se(x,y, theta1, theta2):
    distance = pairwise_distances(x,y)
    Ker = theta1 ** 2 * torch.exp(-0.5 * distance / theta2 ** 2)
    return Ker

def kernel_linear(X_u,X_l,vector_length,Num_A,Num_B, theta):   
    x_l_t =  X_l.repeat(Num_A,1)
    x_u_t =  X_u.repeat(1,Num_B)
    x_u_t = x_u_t.view(Num_A*Num_B,vector_length)
    ker_t = torch.nn.functional.cosine_similarity(x_u_t,x_l_t)
    ker_t = ker_t.view(Num_A,Num_B)
    return ker_t * theta

class SquaredExponentialKernel(object):
    def __init__(self, params):
        self.__params = params

    def get_params(self):
        return torch.clone(self.__params)

    def __call__(self, x, y):
        # compute Squared Exponential kernel
        distance = pairwise_distances(x,y)       
        ker = self.__params[0]**2 * torch.exp(-0.5 * distance / self.__params[1]**2)
        return ker

    def derivatives(self, x, y):
        distance = pairwise_distances(x,y)
        ker = self.__params[0]**2 * torch.exp(-0.5 * distance / self.__params[1]**2)
        ker.backward()
        return self.__params[0].grad, self.__params[1].grad

    def update_parameters(self, updates):
        self.__params += updates


class CustomKernel(object):
    def __init__(self, params, kernel_type):
        self.__params = params
        self.kernel_type = kernel_type

    def get_params(self):
        return self.__params

    def __call__(self, x, y, vector_length=None, Num_A=None, Num_B=None):
        
        if self.kernel_type == 'Linear':
            ker = kernel_linear(x, y, vector_length, Num_A, Num_B, 1.0)

        elif self.kernel_type == 'Squared_exponential':
            ker = kernel_se(x, y, self.__params[0], self.__params[1])

        elif self.kernel_type == 'Mix':
            ker_se = kernel_se(x, y, self.__params[0], self.__params[1])
            ker_l = kernel_linear(x, y, vector_length, Num_A, Num_B, self.__params[2])
            ker = ker_se + ker_l
        
        return ker

    def derivatives(self, x, y, ker_LL, inv_ker, vector_length=None, Num_A=None, Num_B=None):

        # zero gradient for all variables
        for i in range(len(self.__params)):
            if self.__params[i].grad:
                self.__params[i].grad.zero_()
                print('yes')

        # compute kernel values
        if self.kernel_type == 'Linear':
            ker = kernel_linear(x, y, vector_length, Num_A, Num_B, 1.0)

        elif self.kernel_type == 'Squared_exponential':
            ker = kernel_se(x, y, self.__params[0], self.__params[1])

        elif self.kernel_type == 'Mix':
            ker_se = kernel_se(x, y, self.__params[0], self.__params[1])
            ker_l = kernel_linear(x, y, vector_length, Num_A, Num_B, self.__params[2])
            ker = ker_se + ker_l

        log_likelihood = -torch.sum(torch.log(torch.symeig(ker_LL)[0])) - torch.t(y).matmul(inv_ker.matmul(y))

        # ker_sum = ker.sum()
        # ker_sum.backward()
        log_likelihood_sum = log_likelihood.sum()
        log_likelihood_sum.backward()

        gradients = [torch.tensor(self.__params[i].grad.item(), device='cuda').unsqueeze(dim=0) for i in range(len(self.__params))]

        return gradients

    def update_parameters(self, updates):
        self.__params += updates


class GPRegression(object):
    def __init__(self, kernel, z_height, z_width, z_numchnls, num_nearest):
        self.kernel = kernel
        self.num_nearest = num_nearest
        self.vector_length = z_height * z_width * z_numchnls
        self.Eye_num_nearest = torch.eye(num_nearest).cuda()

    def fit(self, X, y, Num_A=None, Num_B=None):
        self.X = X
        self.y = y
        
        # compute (K_LL+sig^2I)^(-1)
        self.ker_LL = self.kernel(X, X, self.vector_length, Num_A, Num_B) + 1.0 * self.Eye_num_nearest
        self.inv_ker = inverse(self.ker_LL)

    def fit_kernel(self, X, y, Num_A=None, Num_B=None, lr=0.1, iter_max=1000):
        for i in range(iter_max):
            params = self.kernel.get_params()
            self.fit(X, y, Num_A, Num_B)
            gradients = self.kernel.derivatives(X, X, self.ker_LL, self.inv_ker, self.vector_length, Num_A, Num_B)
            
            # term1 = -torch.trace(self.inv_ker.matmul(gradients[0].repeat(Num_A, 1)))
            # term2 = self.inv_ker.matmul(y)
            # term3 = torch.t(term2).matmul(gradients[0].repeat(Num_A, 1))

            # updates = [-torch.trace(self.inv_ker.matmul(grad.repeat(Num_A, 1))) + torch.t(self.inv_ker.matmul(y)).matmul(grad.repeat(Num_A, 1)).matmul(self.inv_ker.matmul(y)) for grad in gradients]
            self.kernel.update_parameters(lr * gradients)

            if i % 100 == 0:
                print(i) 

            if torch.allclose(params, self.kernel.get_params()):
                break

        else:
            # 既定の更新回数だけ更新してもパラメータの更新量が小さくない場合以下の文を出力
            print("parameters may not have converged")

        # get inv_ker with the latest hyperparameters
        self.fit(X, y, Num_A, Num_B)

    def predict_dist(self, X_new):
        # compute (K_LL+sig^2I)^(-1) * y
        mn_pre = torch.matmul(self.inv_ker, self.y) # used for computing mean prediction, torch.Size([8, 196608])

        # compute kernel values with training dataset and new data points
        ker_UU = self.kernel(X_new, X_new, self.vector_length, 1, 1)
        ker_UL = self.kernel(X_new, self.X, self.vector_length, 1, self.num_nearest)
        
        # computing mean and variance prediction
        mean_pred = torch.matmul(ker_UL, mn_pre) #mean prediction (mu) or z_pseudo
        sigma_est = ker_UU - torch.matmul(ker_UL, torch.matmul(self.inv_ker, ker_UL.t())) + 1.0

        return mean_pred, sigma_est


########################################################################################################


class GPStruct(object):
    def __init__(self, num_lbl, num_unlbl, train_batch_size, version, kernel_type, pre_trained_enc, img_size):
        self.num_lbl = num_lbl # number of labeled images
        self.num_unlbl = num_unlbl # number of unlabeled images
        # self.z_height= 32 # height of the feature map z i.e dim 2 
        # self.z_width = 32 # width of the feature map z i.e dim 3
        # self.z_numchnls = 32 # number of feature maps in z i.e dim 1
        # self.num_nearest = 8 #number of nearest neighbors for unlabeled vector

        if img_size == [256, 192]:
            self.z_height= 24 
            self.z_width = 32 
        elif img_size == [640, 192]:
            self.z_height= 24 
            self.z_width = 80 

        if pre_trained_enc is None:
            self.z_numchnls = 256 # number of feature maps in z i.e dim 1
        elif pre_trained_enc == 'B5':
            self.z_numchnls = 64 
        elif pre_trained_enc == 'B7':
            self.z_numchnls = 80 
        elif pre_trained_enc == 'B5_full':
            self.z_height = 12
            self.z_width = 16
            self.z_numchnls = 176 

        self.num_nearest = 8 #number of nearest neighbors for unlabeled vector
        self.Fz_lbl = np.zeros((self.num_lbl,self.z_numchnls,self.z_height,self.z_width),dtype=np.float32) #Feature matrix Fzl for latent space labeled vector matrix
        self.Fz_unlbl = np.zeros((self.num_unlbl,self.z_numchnls,self.z_height,self.z_width),dtype=np.float32) #Feature matrix Fzl for latent space unlabeled vector matrix
        self.ker_lbl = np.zeros((self.num_lbl,self.num_lbl)) # kernel matrix of labeled vectors
        self.ker_unlbl = np.zeros((self.num_unlbl,self.num_lbl)) # kernel matrix of unlabeled vectors
        self.dict_lbl ={} # dictionary helpful in saving the feature vectors
        self.dict_unlbl ={} # dictionary helpful in saving the feature vectors
        self.lambda_var = 0.33 # factor multiplied with minimizing variance
        self.train_batch_size = train_batch_size
        self.version = version # version1 is GP SIMO model and version2 is GP MIMO model
        self.kernel_type = kernel_type

    def gen_featmaps_unlbl(self,dataloader,net,device):
        print("Unlabelled: started storing feature vectors and kernel matrix")
        count =0
        for batch_id, train_data in enumerate(dataloader):

            # input_im, gt, imgid = train_data
            input_im = train_data['img_target']
            gt = train_data['lab_target']
            imgid = train_data['img_target_paths']

            input_im = input_im.to(device)
            gt = gt.to(device)
            
            net.eval()

            ### center out
            _, zy_in = net(input_im, gp=True)

            tensor_mat = zy_in.data

            # saving latent space feature vectors 
            for i in range(tensor_mat.shape[0]):
                if imgid[i] not in self.dict_unlbl.keys():
                    self.dict_unlbl[imgid[i]] = count
                    count += 1
                tmp_i = self.dict_unlbl[imgid[i]]
                self.Fz_unlbl[tmp_i,:,:,:] = tensor_mat[i,:,:,:].cpu().numpy()

        X = self.Fz_unlbl.reshape(-1,self.z_numchnls*self.z_height*self.z_width)
        Y = self.Fz_lbl.reshape(-1,self.z_numchnls*self.z_height*self.z_width)
        dist = euclidean_distances(X,Y)
        self.ker_unlbl = np.exp(-0.5*dist)
        print("Unlabelled: stored feature vectors and kernel matrix")
        return

    def gen_featmaps(self,dataloader,net,device):
        
        count =0
        print("Labelled: started storing feature vectors and kernel matrix")
        for batch_id, train_data in enumerate(dataloader):

            # input_im, gt, imgid = train_data

            input_im = train_data['img_source']
            gt = train_data['lab_source']
            imgid = train_data['img_source_paths']

            input_im = input_im.to(device)
            gt = gt.to(device)
            
            net.eval()

            ### center out
            _, zy_in = net(input_im, gp=True)

            tensor_mat = zy_in.data

            # saving latent space feature vectors
            for i in range(tensor_mat.shape[0]):
                if imgid[i] not in self.dict_lbl.keys():
                    self.dict_lbl[imgid[i]] = count
                    count += 1
                tmp_i = self.dict_lbl[imgid[i]]
                self.Fz_lbl[tmp_i,:,:,:] = tensor_mat[i,:,:,:].cpu().numpy()

        X = self.Fz_lbl.reshape(-1,self.z_numchnls*self.z_height*self.z_width)
        Y = self.Fz_lbl.reshape(-1,self.z_numchnls*self.z_height*self.z_width)
        self.var_Fz_lbl = np.std(self.Fz_lbl,axis=0)
        # dist = euclidean_distances(X,Y)**2
        dist = euclidean_distances(X,Y)
        self.ker_lbl = np.exp(-0.5*dist**2)
        print("Labelled: stored feature vectors and kernel matrix")
        return

    def compute_gploss(self,zy_in,imgid,batch_id,label_flg=0):
        tensor_mat = zy_in
        
        # Sg_Pred = torch.zeros([self.train_batch_size,1])
        # Sg_Pred = Sg_Pred.cuda()
        # LSg_Pred = torch.zeros([self.train_batch_size,1])
        # LSg_Pred = LSg_Pred.cuda()
        gp_loss = 0

        theta1 = torch.tensor(1.0, requires_grad=True, device='cuda')
        theta2 = torch.tensor(self.var_Fz_lbl.max(axis=-1).max(axis=-1).max(axis=-1), requires_grad=True, device='cuda')  #1.0  #(np.sum(mu**2))**(pw)
        theta3 = torch.tensor(1.0, requires_grad=True, device='cuda')
        self.params = [theta1, theta2, theta3]
        
        for i in range(tensor_mat.shape[0]):
            tmp_i = self.dict_unlbl[imgid[i]] if label_flg==0 else self.dict_lbl[imgid[i]] # imag_id in the dictionary
            tensor = tensor_mat[i,:,:,:] # z tensor

            if self.version == 'version1':
                tensor_vec = tensor.view(-1,self.z_numchnls*self.z_height*self.z_width) # z tensor to a vector
            else:
                tensor_vec = tensor.view(-1,self.z_height*self.z_width) # z tensor to a vector

            ########################################
            ### Get Nearest Neighbors (Farthest) ###
            ########################################
            nearest_vl = self.ker_unlbl[tmp_i,:] if label_flg==0 else self.ker_lbl[tmp_i,:] #kernel values are used to get neighbors
            # Nearest and Farthest neighbors
            tp32_vec = np.array(sorted(range(len(nearest_vl)), key=lambda k: nearest_vl[k])[-1*self.num_nearest:])
            # lt32_vec = np.array(sorted(range(len(nearest_vl)), key=lambda k: nearest_vl[k])[:self.num_nearest])

            # Nearest neighbor latent space labeled vectors
            near_dic_lbl = np.zeros((self.num_nearest,self.z_numchnls,self.z_height,self.z_width))
            for j in range(self.num_nearest):
                near_dic_lbl[j,:] = self.Fz_lbl[tp32_vec[j],:,:,:]  # (8, 256, 24, 32)
            if self.version == 'version1':
                near_vec_lbl = np.reshape(near_dic_lbl,(self.num_nearest,self.z_numchnls*self.z_height*self.z_width))  # (8, 256*24*32)
            else :
                near_vec_lbl = np.reshape(near_dic_lbl,(self.num_nearest*self.z_numchnls,self.z_height*self.z_width))
                
            # Farthest neighbor latent space labeled vectors
            # far_dic_lbl = np.zeros((self.num_nearest,self.z_numchnls,self.z_height,self.z_width))
            # for j in range(self.num_nearest):
            #     far_dic_lbl[j,:] = self.Fz_lbl[lt32_vec[j],:,:,:]
            # if self.version == 'version1':
            #     far_vec_lbl = np.reshape(far_dic_lbl,(self.num_nearest,self.z_numchnls*self.z_height*self.z_width))
            # else :
            #     far_vec_lbl = np.reshape(far_dic_lbl,(self.num_nearest*self.z_numchnls,self.z_height*self.z_width))

            ######################
            ### numpy to torch ###
            ######################
            # converting require variables to cuda tensors 
            near_vec_lbl = torch.from_numpy(near_vec_lbl.astype(np.float32))
            # far_vec_lbl = torch.from_numpy(far_vec_lbl.astype(np.float32))
            near_vec_lbl = near_vec_lbl.cuda()
            # far_vec_lbl = far_vec_lbl.cuda()

            ################################
            ### Compute Gaussian Process ###
            ################################

            ### set kernel and GP model for nearest vectors
            custom_kernel = CustomKernel(self.params, self.kernel_type)
            GPRegression_near = GPRegression(custom_kernel, self.z_height, self.z_width, self.z_numchnls, self.num_nearest)
            
            ### optimize hyperparameters or not for nearest vectors
            GPRegression_near.fit(near_vec_lbl, near_vec_lbl, Num_A=self.num_nearest, Num_B=self.num_nearest)
            # GPRegression_near.fit_kernel(near_vec_lbl, near_vec_lbl, Num_A=self.num_nearest, Num_B=self.num_nearest)
            
            ### set kernel and GP model for farthest vectors
            # custom_kernel_latest = GPRegression_near.kernel
            # GPRegression_far = GPRegression(custom_kernel_latest, self.z_height, self.z_width, self.z_numchnls, self.num_nearest)
            # GPRegression_far.fit(far_vec_lbl, far_vec_lbl, Num_A=self.num_nearest, Num_B=self.num_nearest)

            ### predict mean and variance with new data points for both nearest and farthese vectors
            mean_pred, sigma_est = GPRegression_near.predict_dist(tensor_vec)
            # _, far_sigma_est = GPRegression_far.predict_dist(tensor_vec)
            
            sigma_est_nograd = sigma_est[0].detach()
            if self.version == 'version1':
                # loss_unsup = torch.mean(((tensor_vec-mean_pred)**2)/sigma_est[0]) + 1.0*self.lambda_var*torch.log(torch.det(sigma_est)) - 0.000001*self.lambda_var*torch.log(torch.det(far_sigma_est))
                loss_unsup = torch.mean(((tensor_vec-mean_pred)**2)/sigma_est_nograd) + 1.0*self.lambda_var*torch.log(torch.det(sigma_est))  # DLB_c
            else:
                inv_sigma = torch.inverse(sigma_est)
                loss_unsup = torch.mean(torch.matmul((tensor_vec-mean_pred).t(),torch.matmul(inv_sigma,(tensor_vec-mean_pred)))) + 1.0*self.lambda_var*torch.log(torch.det(sigma_est))  #torch.mean(torch.matmul((tensor_vec-mean_pred).t(),torch.matmul(inv_sigma,(tensor_vec-mean_pred))))
            if loss_unsup==loss_unsup:
                gp_loss += ((1.0*loss_unsup/self.train_batch_size))
            # Sg_Pred[i,:] = torch.log(torch.det(sigma_est))
            # LSg_Pred[i,:] = torch.log(torch.det(far_sigma_est))
        
        # if not (batch_id % 100) and loss_unsup==loss_unsup:
            # print(LSg_Pred.max().item(),Sg_Pred.max().item(),gp_loss.item()/self.train_batch_size,Sg_Pred.mean().item())        

        return gp_loss