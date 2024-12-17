
import torch as th
import torch.nn as nn
import torch.nn.init as init
import pandas as pd
import numpy as np
import logging
import csv 
from time import localtime, strftime
import os 

seed = 1787
#random.seed(seed)
#import os
#os.environ['PYTHONHASHSEED'] = str(seed)

#th.manual_seed(seed)
#th.cuda.manual_seed(seed)
#th.cuda.manual_seed_all(seed)
#th.backends.cudnn.deterministic = True


class Network():


    def dist_mat(self, x):

        try:
            x = th.from_numpy(x)
        except TypeError:
            x = x

        if len(x.size()) == 4:
            x = x.view(x.size()[0], -1)
        dist = th.norm(x[:, None] - x, dim=2)
        return dist

    def entropy(self, *args):

        for idx, val in enumerate(args):
            if idx == 0:
                k = val.clone()
            else:
                k *= val

        k /= k.trace()

        #c = th.tensor([0]).cuda()
        c = th.tensor([0])
        #print(not(th.symeig(k)[0] < c).any())

        eigv = th.abs(th.symeig(k, eigenvectors=False)[0])
        temp=eigv.clone()
        #eigv_log2= temp.log2().cuda()
        eigv_log2= temp.log2()

        if((eigv==c).any()):

          zero_indices=(eigv == 0).nonzero().tolist()
          #small=th.tensor([0.999999999]).cuda()
          #all_value=small.detach().clone()
          small=th.tensor([0.0000000099])
          small_value=small.detach()
          for i in zero_indices:
            eigv_log2[i]=small_value
          #print(eigv*eigv_log2.sum())

        if(th.isnan(-((eigv*eigv_log2).sum()))):
          #if(th.tensor(True).cuda() in th.isnan(eigv*(eigv_log2))):
          if(th.tensor(True) in th.isnan(eigv*(eigv_log2))):
            print('yesssssssssssssssssss')
            print(eigv)
            print(eigv_log2)
            print(eigv*(eigv_log2))
          else:
            print('nooooooooooooooooooo')
            print(eigv)

        return -(eigv*(eigv_log2)).sum()

    def kernel_mat(self, x, k_y, sigma=None, epoch=None):

        d = self.dist_mat(x)
        #print('ready for sigma calculation',epoch,sigma)
        if sigma is None:
            if epoch > 20:
                #sigma_vals = th.linspace(0.3, 10*d.mean(), 100).cuda() #for vgg
                sigma_vals = th.linspace(0.3, 10*d.mean(), 100)
                #sigma_vals = th.linspace(0.1*d.mean(), 10*d.mean(), 50).cuda() #---for lenet
              
            else:
                #sigma_vals = th.linspace(0.3, 10*d.mean(), 300).cuda() #for vgg
                sigma_vals = th.linspace(0.3, 10*d.mean(), 300)
                #sigma_vals = th.linspace(0.1*d.mean(), 10*d.mean(), 75).cuda()#for lenet
            L = []
            for sig in sigma_vals:
                k_l = th.exp(-d ** 2 / (sig ** 2)) / d.size(0)
                L.append(self.kernel_loss(k_y, k_l))

            if epoch == 0:
                self.sigmas[epoch] = sigma_vals[L.index(max(L))]
            else:
                self.sigmas[epoch] = 0.9*self.sigmas[epoch-1] + 0.1*sigma_vals[L.index(max(L))]
            #print('---',L.index(max(L)))
            sigma = self.sigmas[epoch]
        return th.exp(-d ** 2 / (sigma ** 2))


    def kernel_mat_training(self, x, k_y, sigma=None, epoch=None):

        d = self.dist_mat(x)
        #print('ready for sigma calculation',epoch,sigma)
        if sigma is None:
            if epoch > 20:
                #sigma_vals = th.linspace(0.3, 10*d.mean(), 100).cuda() #for vgg
                sigma_vals = th.linspace(0.3, 10*d.mean(), 100)
                #sigma_vals = th.linspace(0.1*d.mean(), 10*d.mean(), 50).cuda() #---for lenet
              
            else:
                #sigma_vals = th.linspace(0.3, 10*d.mean(), 300).cuda() #for vgg
                sigma_vals = th.linspace(0.3, 10*d.mean(), 300)
                #sigma_vals = th.linspace(0.1*d.mean(), 10*d.mean(), 75).cuda()#for lenet
            L = []
            for sig in sigma_vals:
                k_l = th.exp(-d ** 2 / (sig ** 2)) / d.size(0)
                L.append(self.kernel_loss(k_y, k_l))
            #print(idx,epoch)
            if epoch == 0:
                self.sigmas_training[epoch] = sigma_vals[L.index(max(L))]
            else:
                self.sigmas_training[epoch] = 0.9*self.sigmas_training[epoch-1] + 0.1*sigma_vals[L.index(max(L))]

            #print('---',L.index(max(L)))
            sigma = self.sigmas_training[ epoch]
        return th.exp(-d ** 2 / (sigma ** 2))



    def kernel_loss(self, k_y, k_l):

        beta = 1.0

        L = th.norm(k_l)
        Y = th.norm(k_y) ** beta
        #X = th.norm(k_x) ** (1-beta)

        LY = th.trace(th.matmul(k_l, k_y))**beta
        #LX = th.trace(th.matmul(k_l, k_x))**(1-beta)

        #return 2*th.log2((LY*LX)/(L*Y*X))
        return 2*th.log2((LY)/(L*Y))


    def cal_mi(self, x, y, output, model, gpu, current_iteration):

        #model.eval()
        data= [output]
        data.insert(0, x)
        data.append(self.one_hot(y, gpu))

        k_x = self.kernel_mat(data[0], [], sigma=th.tensor(8.0))
        k_y = self.kernel_mat(data[-1], [],sigma=th.tensor(0.1))

        k_list = [k_x]
        for idx_l, val in enumerate(data[1:-1]):
            k_list.append(self.kernel_mat(val.reshape(data[0].size(0), -1),
                                          k_y, epoch=current_iteration))
        k_list.append(k_y)

        e_list = [self.entropy(i) for i in k_list]
        j_XT = [self.entropy(k_list[0], k_i) for k_i in k_list[1:-1]]
        j_TY = [self.entropy(k_i, k_list[-1]) for k_i in k_list[1:-1]]

        for idx_mi, val_mi in enumerate(e_list[1:-1]):
            self.MI[current_iteration, 0] = e_list[0]+val_mi-j_XT[idx_mi]
            self.MI[current_iteration, 1] = e_list[-1]+val_mi-j_TY[idx_mi]

        return



    def cal_mi_training(self, x, y, output, model, gpu, current_iteration):

        #model.eval()
        data= [output]
        data.insert(0, x)
        data.append(self.one_hot(y, gpu))

        k_x = self.kernel_mat_training(data[0], [], sigma=th.tensor(8.0))
        k_y = self.kernel_mat_training(data[-1], [],sigma=th.tensor(0.1))

        k_list = [k_x]
        #print('shapuuuuuuuuuu', len(data))
        for idx_l, val in enumerate(data[1:-1]):
            #print(idx_l)
            k_list.append(self.kernel_mat_training(val.reshape(data[0].size(0), -1),
                                           k_y, epoch=current_iteration))
        k_list.append(k_y)

        e_list = [self.entropy(i) for i in k_list]
        j_XT = [self.entropy(k_list[0], k_i) for k_i in k_list[1:-1]]
        j_TY = [self.entropy(k_i, k_list[-1]) for k_i in k_list[1:-1]]

        for idx_mi, val_mi in enumerate(e_list[1:-1]):
            self.MI_training[current_iteration, 0] = e_list[0]+val_mi-j_XT[idx_mi]
            self.MI_training[current_iteration, 1] = e_list[-1]+val_mi-j_TY[idx_mi]

        return


    def weight_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            init.constant_(m.bias.data, 0)
            '''if self.a_type == 'relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'leaky_relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'tanh':
                g = init.calculate_gain(self.a_type)
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'sigmoid':
                g = init.calculate_gain(self.a_type)
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            else:
                raise
                return NotImplemented'''



    def one_hot(self, y, gpu):

        try:
            y = th.from_numpy(y)
        except TypeError:
            None

        y_1d = y
        if gpu:
            y_hot = th.zeros((y.size(0), th.max(y).int()+1)).cuda()
        else:
            y_hot = th.zeros((y.size(0), th.max(y).int()+1))

        for i in range(y.size(0)):
            y_hot[i, y_1d[i].int()] = 1

        return y_hot


    def intialize_layer_name_num(self,model):

        layer_number=0
        for layer_name, layer_module in model.named_modules():
           #if(isinstance(layer_module, th.nn.Conv2d) or  isinstance(layer_module,th.nn.Linear)): -----for both conv and fc layers
           if(isinstance(layer_module, th.nn.Conv2d)):
              self.layer_name_num[layer_number]=layer_name
              layer_number=layer_number+1
        #print('layer name vs number:',self.layer_name_num)
        return


 

    def create_folders(self):

      main_dir=strftime("\Results\%b%d_%H_%M_%S%p", localtime() )+"_CNN-2"
      import os
      current_dir =  os.path.abspath(os.path.dirname(__file__))
      par_dir = os.path.abspath(current_dir)

      parent_dir=par_dir+main_dir
      os.makedirs(parent_dir)

      return parent_dir


    def get_writerow(self,k):

      s='wr.writerow(['

      for i in range(k):

          s=s+'d['+str(i)+']'

          if(i<k-1):
             s=s+','
          else:
             s=s+'])'

      return s

    def get_logger(self,file_path):

        logger = logging.getLogger('gal')
        log_format = '%(asctime)s | %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

        return logger
