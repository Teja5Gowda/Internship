''' 1.Traning: whole train data
    2.Testing: Whole test data batch wise'''
import random
import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
#from CNN_Hero import CNN_Hero
from updated_LeNet import LeNet
import csv
from itertools import zip_longest
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from matplotlib.ticker import PercentFormatter
from torch.optim.lr_scheduler import MultiStepLR
import math

seed =1787
random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)

th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


#th.cuda.set_device(0)
gpu = th.cuda.is_available()

if gpu:
    #export CUDA_VISIBLE_DEVICES="0"
    th.cuda.set_device(1)

trainloader = th.utils.data.DataLoader(datasets.MNIST('../mnist_data', 
                                                          download=True, 
                                                          train=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                              #transforms.Normalize((0.5,), (0.5,)) # normalize inputs
                                                          ])), 
                                           batch_size=100, 
                                           shuffle=True,num_workers=0)

# download and transform test dataset
testloader = th.utils.data.DataLoader(datasets.MNIST('../mnist_data', 
                                                          download=True, 
                                                          train=False,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                              #transforms.Normalize((0.5,), (0.5,)) # normalize inputs
                                                          ])), 
                                           batch_size=100, 
                                           shuffle=True,num_workers=0)

N = 1

batch_size_tr = 100
batch_size_te = 100

epochs = 120

short_train=False


tr_size = 60000
te_size=10000


activation = 'softplus'

#tr_size = 300
#te_size=300
#short_train=True


n_iterations_per_epoch_training = (tr_size // batch_size_tr) #600
n_iterations_training = (tr_size // batch_size_tr)*(epochs)  #600*120

n_iterations_per_epoch= (te_size // batch_size_te)  
n_iterations = (te_size // batch_size_te)*(epochs)

print('n_iterations.....######################################3',n_iterations)

if gpu:
    model=LeNet(activation, n_iterations, n_iterations_training).cuda()
else:
    model=LeNet(activation, n_iterations, n_iterations_training)


#QUESTION - why are we using no_grad()

with th.no_grad():

  folder_name=model.create_folders()
  print(folder_name)
  logger=model.get_logger(folder_name+'/logger.log')

  #---------intialize layer numbers and names dictionary------
  #model.intialize_layer_name_num(model)

  #---------intialize pruned and remaning filters-----------
  #model.filters_in_each_layer(model)
  
  #--------calculate remaining filters in each epoch--------
  #model.remaining_filters_per_epoch(model=model,initial=True)


  #------------calculating parameters and values------------------------------
  #model.calculate_total_parameters()
  #model.calculate_total_flops(model)

optimizer = th.optim.SGD(model.parameters(), lr=0.1,momentum=0.9)
scheduler = MultiStepLR(optimizer, milestones=[60,40], gamma=0.1)


print(model.sigmas_training.shape)
criterion = nn.CrossEntropyLoss()

heading=['MI_x_tr','MI_y_tr','MI_x_te','MI_y_te','Acc_tr','Acc_te','loss','epochs']
with open(folder_name+'/lenetPrune.csv', 'a', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(heading)

myfile.close()


for n in range(N):


    
    mi_iteration =0
    mi_iteration_training =0

    for epoch in range(epochs):

      running_loss=0
      train_acc=[]

      for batch_num, (inputs, targets) in enumerate(trainloader):

        if(batch_num==3 and (short_train == True)):
                break
        #print('stopped',batch_num)
        #inputs = inputs.cuda()
        inputs = inputs
        #targets = targets.cuda()
        targets = targets


        model.train()

        optimizer.zero_grad()
        output = model(inputs)
        #print(output.shape)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
      
        

        #model.train_loss.append(loss.item())

        with th.no_grad():
         
          running_loss += loss.item()
          
          y_hat = th.argmax(output, 1)
          score = th.eq(y_hat, targets).sum()

          train_acc.append(score.item())
          model.cal_mi_training(inputs, targets, output, model, gpu, mi_iteration_training)
          mi_iteration_training +=1
          #print(mi_iteration, mi_iteration_training)
          
              

          #---------------------------hooks end here------------------------------

      
      with th.no_grad():

        model.train_accuracy.append((sum(train_acc)*100)/tr_size)
        #model.train_loss.append(running_loss/len(trainloader)) 
        model.train_loss.append(running_loss/len(trainloader))       
        test_acc=[]
        model.eval()
        for batch_nums, (inputs2, targets2) in enumerate(testloader):
            if(batch_nums==3 and (short_train == True)):
                break

            #inputs2, targets2 = inputs2.cuda(), targets2.cuda()
            inputs2, targets2 = inputs2, targets2
             
            output2= model(inputs2)
            y_hat = th.argmax(output2, 1)
            score = th.eq(y_hat, targets2).sum()
            test_acc.append(score.item())
     
            model.cal_mi(inputs2, targets2, output2, model, gpu, mi_iteration)

            mi_iteration+=1



        model.test_accuracy.append((sum(test_acc)*100)/te_size)        



        d=[]
        #--------MI-TRAIN---------
        MI_x_train=model.MI_training[epoch * n_iterations_per_epoch_training : (epoch+1)* n_iterations_per_epoch_training ,0] 
        MI_x_train= np.mean(MI_x_train.cpu().detach().numpy().astype('float16')).round(decimals=3)
 
        MI_y_train=model.MI_training[epoch * n_iterations_per_epoch_training : (epoch+1)* n_iterations_per_epoch_training ,1] 
        MI_y_train= np.mean(MI_y_train.cpu().detach().numpy().astype('float16')).round(decimals=3)            

        d.append(MI_x_train)
        d.append(MI_y_train)

        #-------MI-Test-----------
        MI_x = model.MI[epoch * n_iterations_per_epoch : (epoch+1)* n_iterations_per_epoch ,0] 
        MI_x = np.mean(MI_x.cpu().detach().numpy().astype('float16')).round(decimals=3)
 
        MI_y=model.MI[epoch * n_iterations_per_epoch : (epoch+1)* n_iterations_per_epoch,1] 
        MI_y= np.mean(MI_y.cpu().detach().numpy().astype('float16')).round(decimals=3)            

        d.append(MI_x)
        d.append(MI_y)

        #-------ACC-----------
        d.append(round(model.train_accuracy[-1],2))
        d.append(round(model.test_accuracy[-1],2))
 
        #-------Loss---------
        d.append(round(model.train_loss[-1],3))

        #------epochs-------
        d.append(epoch+1)

        with open(folder_name+'/lenetPrune.csv', 'a', newline='') as myfile:
            wr = csv.writer(myfile)
            command=model.get_writerow(len(d))
            eval(command)

        myfile.close()


      logger.info('\n---------------Epoch number: {}---Train accuracy: {}----Test accuracy: {}'.format(epoch,model.train_accuracy[-1],model.test_accuracy[-1]))

      scheduler.step()
      


print(model.MI.size())
print(model.sigmas.size())


mi = model.MI.cpu().detach().numpy().astype('float16')
sigmas=model.sigmas.cpu().detach().numpy().astype('float16')
iter_per_epoch=np.int64(te_size // batch_size_te)

mi_training = model.MI_training.cpu().detach().numpy().astype('float16')
sigmas_training=model.sigmas_training.cpu().detach().numpy().astype('float16')
iter_per_epoch_training=np.int64(tr_size // batch_size_tr)

#prune_iters=np.int64((tr_size // batch_size_tr)*mi_calc_epochs)
epochs= np.int64(epochs)
train_acc= np.array(model.train_accuracy,dtype=np.float16)
test_acc= np.array(model.test_accuracy,dtype=np.float16)
train_loss= np.array(model.train_loss,dtype=np.float16)
test_loss= np.array(model.test_loss,dtype=np.float16)

np.savez_compressed(folder_name+'/lenet_'+activation+'.npz',
                      a=mi, b=sigmas,c=iter_per_epoch, 
                      d=mi_training, e=sigmas_training, f=iter_per_epoch_training, 
                      g=n_iterations_training, h=train_acc, i=test_acc, j=train_loss)



from zipfile import ZipFile
import os,glob

directory = os.path.dirname(os.path.realpath(__file__)) #location of running file
file_paths = []
os.chdir(directory)
for filename in glob.glob("*.py"):
	filepath = os.path.join(directory, filename)
	file_paths.append(filepath)
	#print(filename)

print('Following files will be zipped:')
for file_name in file_paths:
	print(file_name)
saving_loc = folder_name #location of results
os.chdir(saving_loc)
# writing files to a zipfile
with ZipFile('python_files.zip','w') as zip:
	# writing each file one by one
	for file in file_paths:
		zip.write(file)

