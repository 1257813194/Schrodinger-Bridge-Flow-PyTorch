import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import sb.networks
import time
import os
from datetime import datetime
import ot


class sb_muti_model(object):
    def __init__(self,pi_list,timepoints,N_pretraining=1000,N_finetuning=1000,B=128,steps=60,eps=10,base_lr=1e-3,finetuning_lr=1e-5,decay=0.8,change_eps=False,pic_min=-1,pic_max=1,patience=1e8,lambda_=1e-3,clamp=False,save=False):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.length=len(pi_list)
        self.pi=[sb.networks.CustomDataset(pi_list[i]) for i in range(self.length)]
        self.timepoints = timepoints
        self.N_pretraining=N_pretraining
        self.N_finetuning=N_finetuning
        self.B=B
        self.b=int(B/2)
        self.d=pi_list[0].shape[1]
        self.steps=steps
        self.eps=eps
        self.base_lr=base_lr
        self.finetuning_lr=finetuning_lr
        self.limit=1e-8
        self.decay=decay
        self.epoch=1600
        self.change_eps = change_eps
        self.min = pic_min 
        self.max = pic_max
        self.delta_t = (timepoints[-1] - timepoints[0])/self.steps
        self.patience=patience
        self.lambda_ = lambda_
        self.clamp = clamp
        self.t_list=list(np.arange(timepoints[0],timepoints[-1],self.delta_t)) + [timepoints[-1]]
        self.save=save
        
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        directory_name = 'model_history/'+current_time
        os.makedirs(directory_name, exist_ok=True)
        self.path = directory_name

        self.t_lists_stage=[[] for i in range(len(self.timepoints)-1)]
        for t in self.t_list:
            for t_index,t_point in enumerate(self.timepoints[1:]):
                if t >= t_point:
                    continue
                else:
                    self.t_lists_stage[t_index].append(t)
                    break
        for stage,t_lst in enumerate(self.t_lists_stage):
            t_lst.append(float(self.timepoints[stage+1]))

        self.v_fore=sb.networks.SimpleUNet().to(self.device)
        self.v_back=sb.networks.SimpleUNet().to(self.device)
        self.scale_m_fore=sb.networks.scale_model_muti(output_size=32).to(self.device)
        self.scale_m_back=sb.networks.scale_model_muti(output_size=32).to(self.device)

    def run_base(self,loss_plot=True):
        criterion_fore = nn.MSELoss()
        criterion_back = nn.MSELoss()
        optimizer = optim.Adam(list(self.v_fore.parameters())+list(self.v_back.parameters())+list(self.scale_m_fore.parameters())+list(self.scale_m_back.parameters()), lr=self.base_lr)
        v_fore_params = {name: param.clone().detach() for name, param in self.v_fore.named_parameters()}
        v_back_params = {name: param.clone().detach() for name, param in self.v_back.named_parameters()}
        for param in v_fore_params.values():
            param.requires_grad = False  
        for param in v_back_params.values():
            param.requires_grad = False  
        self.loss_history={'loss':[],"loss_fore":[],"loss_back":[]}
        best_loss = np.inf
        epochs_no_improve = 0  # 没有改善的轮数
        early_stop = False
        with tqdm(total=self.N_pretraining) as pbar:
            for n in range(self.N_pretraining):
                if early_stop:
                    print("Early stopping at epoch", n)
                    break
                epoch_loss=0
                epoch_loss_fore = 0
                epoch_loss_back = 0
                dataloader_list=[DataLoader(self.pi[i],shuffle=True,batch_size=self.B) for i in range(self.length)]
                optimizer.zero_grad()
                for data_tuple in zip(*dataloader_list):
                    num_all=[data_tuple[i].shape[0] for i in range(self.length)]
                    if len(set(num_all))!=1:
                        break
                    for stage in range(self.length-1):
                        x_0=data_tuple[stage].cuda()
                        x_1=data_tuple[stage+1].cuda()
                        x_t,t,t_ceil,t_floor=self.Interp_t(x_0,x_1,stage,bondary_constrain=True)
                        batch_size=x_0.shape[0]
                        b=batch_size//2
                        t_fore = t[:b]
                        t_back = t[b:]
                        x_t_fore = x_t[:b]
                        x_t_back = x_t[b:]
                        y_t_fore=self.scale_m_fore(t_fore,t_floor)
                        y_t_back=self.scale_m_back(t_back,t_floor)
                        x_fore=self.v_fore(x_t_fore,y_t_fore)
                        x_back=self.v_back(x_t_back,y_t_back)
                        loss_fore=criterion_fore(x_fore,(x_1[:b]-x_t_fore)/(t_ceil-t_fore).view(-1,1)) 
                        loss_back=criterion_back(x_back,(x_0[b:]-x_t_back)/(t_back-t_floor).view(-1,1))
                        loss=0.5*(loss_fore+loss_back)
                        epoch_loss += loss
                        epoch_loss_fore += loss_fore
                        epoch_loss_back += loss_back
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.v_fore.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(self.v_back.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(self.scale_m_fore.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(self.scale_m_back.parameters(), max_norm=1.0)
                        optimizer.step()
                        self.apply_ema_to_model(self.v_fore,v_fore_params)
                        self.apply_ema_to_model(self.v_back,v_back_params)
                if n%2 == 1: 
                    with torch.no_grad():
                        self.loss_history['loss'].append(epoch_loss.item())
                        # self.loss_history['loss_fore'].append(epoch_loss_fore.item())
                        # self.loss_history['loss_back'].append(epoch_loss_back.item())
                pbar.set_description('processed: %d' % (1 + n))
                pbar.set_postfix({'loss':epoch_loss.detach().cpu().numpy(),'loss_fore':epoch_loss_fore.detach().cpu().numpy(),'loss_back':epoch_loss_back.detach().cpu().numpy(),})
                pbar.update(1)
                if self.save and n%100==0 and n!=0:
                    models_dict = {
                        'v_fore': self.v_fore.state_dict(),
                        'v_back': self.v_back.state_dict(),
                        'scale_m_fore': self.scale_m_fore.state_dict(),
                        'scale_m_back': self.scale_m_back.state_dict(),
                        'loss_history':self.loss_history
                        }
                    torch.save(models_dict, self.path+'/backbone_'+str(n)+'.pt')
                    torch.cuda.empty_cache()
                    self.backbone_load(self.path+'/backbone_'+str(n)+'.pt')                    
                if epoch_loss.item() < best_loss:
                    best_loss = epoch_loss.item()
                    epochs_no_improve = 0  # 重置没有改善的轮次
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    early_stop = True
        if self.save:
            models_dict = {
                'v_fore': self.v_fore.state_dict(),
                'v_back': self.v_back.state_dict(),
                'scale_m_fore': self.scale_m_fore.state_dict(),
                'scale_m_back': self.scale_m_back.state_dict(),
                'loss_history':self.loss_history
                        }
            torch.save(models_dict, self.path+'/backbone_end.pt')
            torch.cuda.empty_cache()
        if loss_plot:
            list1 = self.loss_history['loss']
            x = range(len(list1))
            plt.plot(x, list1, label='loss')
            plt.legend()
            plt.show()
        return None
    
    def backbone_load(self,model_path='backbone.pt'):
        models_dict=torch.load(model_path)
        self.v_fore.load_state_dict(models_dict['v_fore'])
        self.v_back.load_state_dict(models_dict['v_back'])
        self.scale_m_fore.load_state_dict(models_dict['scale_m_fore'])
        self.scale_m_back.load_state_dict(models_dict['scale_m_back'])
        list1 = models_dict['loss_history']['loss']
        x = range(len(list1))
        plt.plot(x, list1, label='loss')
        plt.legend()
        plt.show()
        
    def finetuning_load(self,model_path='finetuning.pt'):
        models_dict=torch.load(model_path)
        self.v_fore_copy=copy.deepcopy(self.v_fore).train()
        self.v_back_copy=copy.deepcopy(self.v_back).train()
        self.v_fore_copy.load_state_dict(models_dict['v_fore_copy'])
        self.v_back_copy.load_state_dict(models_dict['v_fore_copy'])
        

    def finetuning(self,change=10,loss_plot=True):
        self.v_fore_copy=copy.deepcopy(self.v_fore).train()
        self.v_back_copy=copy.deepcopy(self.v_back).train()
        self.scale_m_fore.eval()
        self.scale_m_back.eval()
        criterion_fore = nn.MSELoss()
        criterion_back = nn.MSELoss()
        optimizer_fore = optim.Adam(self.v_fore_copy.parameters(), lr=self.finetuning_lr)
        optimizer_back = optim.Adam(self.v_back_copy.parameters(), lr=self.finetuning_lr)
        self.finetuning_loss_history={'loss':[],"loss_fore":[],"loss_back":[]}
        fore_params = {name: param.clone().detach() for name, param in self.v_fore_copy.named_parameters()}
        back_params = {name: param.clone().detach() for name, param in self.v_back_copy.named_parameters()}
        for param in fore_params.values():
            param.requires_grad = False  
        for param in back_params.values():
            param.requires_grad = False
        with tqdm(total=self.N_finetuning) as pbar:
            for n in range(self.N_finetuning):
                dataloader_list=[DataLoader(self.pi[i],shuffle=True,batch_size=self.B) for i in range(self.length)]
                epoch_loss=0
                epoch_loss_fore = 0
                epoch_loss_back = 0
                for data_tuple in zip(*dataloader_list):
                    num_all=[data_tuple[i].shape[0] for i in range(self.length)]
                    if len(set(num_all))!=1:
                        break
                    for stage_index in range(self.length-1):
                        stage_real=self.timepoints[stage_index]
                        if (n//change)%2==0:
                            x_1=data_tuple[stage_index+1].cuda().double()
                            optimizer_fore.zero_grad()
                            x_0_hat=self.bwd_one_step(x_1,self.v_back_copy,stage_index)[-1].cuda()
                            x_t_fore,t_fore,t_ceil,t_floor=self.Interp_t(x_0_hat,x_1,stage_index)
                            index_fore = (t_fore!=t_ceil).squeeze()
                            x_fore = self.v_fore_copy(x_t_fore[index_fore,:],self.scale_m_fore(t_fore[index_fore,:],stage_real))
                            loss_fore=criterion_fore(x_fore,
                            (x_1[index_fore,:]-x_t_fore[index_fore,:])/(t_ceil-t_fore[index_fore,:]).view(-1,1)) + self.lambda_ * torch.mean(torch.sum(torch.abs(x_fore),dim=1))
                            loss=loss_fore
                            epoch_loss += loss
                            epoch_loss_fore += loss_fore
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.v_fore_copy.parameters(), max_norm=1.0)
                            optimizer_fore.step()
                            self.apply_ema_to_model(self.v_fore_copy,fore_params)
                        else:
                            x_0=data_tuple[stage_index].cuda().double()
                            optimizer_back.zero_grad()
                            x_1_hat=self.fwd_one_step(x_0,self.v_fore_copy,stage_index)[-1].cuda()
                            x_t_back,t_back,t_ceil,t_floor=self.Interp_t(x_0,x_1_hat,stage_index)
                            index_back = (t_back!=t_floor).squeeze()
                            x_back = self.v_back_copy(x_t_back[index_back,:],self.scale_m_back(t_back[index_back,:],stage_real))
                            loss_back=criterion_back(x_back,
                            (x_0[index_back,:]-x_t_back[index_back,:])/(t_back[index_back,:]-t_floor).view(-1,1)) + self.lambda_ * torch.mean(torch.sum(torch.abs(x_back),dim=1))
                            loss=loss_back
                            epoch_loss += loss
                            epoch_loss_back += loss_back
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.v_back_copy.parameters(), max_norm=1.0)
                            optimizer_back.step()
                            self.apply_ema_to_model(self.v_back_copy,back_params)
                if n % 2 == 1:
                    with torch.no_grad():
                        self.finetuning_loss_history['loss'].append(epoch_loss.item())
                        if (n//change)%2==0:
                            self.finetuning_loss_history['loss_fore'].append(epoch_loss_fore.item())
                        else:
                            self.finetuning_loss_history['loss_back'].append(epoch_loss_back.item())
                if self.save and n%100==0 and n!=0:
                    models_dict = {
                        'v_fore_copy': self.v_fore_copy.state_dict(),
                        'v_back_copy': self.v_back_copy.state_dict(),
                        }
                    torch.save(models_dict, self.path+'/finetuning_'+str(n)+'.pt')
                    torch.cuda.empty_cache()
                    self.finetuning_load(self.path+'/finetuning_'+str(n)+'.pt')
                pbar.set_description('processed: %d' % (1 + n))
                pbar.set_postfix({'loss':epoch_loss.detach().cpu().numpy()})
                pbar.update(1)
        if self.save :
            models_dict = {
                'v_fore_copy': self.v_fore_copy.state_dict(),
                'v_back_copy': self.v_back_copy.state_dict()}
            torch.save(models_dict, self.path+'/finetuning_end.pt')
            torch.cuda.empty_cache()
        if loss_plot:
            list1 = self.finetuning_loss_history['loss']
            x = range(len(list1))
            plt.plot(x, list1, label='loss')
            plt.legend()
            plt.show()

    def eval_fore(self,test_0,v_fore,eps_test=None):
        v_fore.eval()
        self.scale_m_fore.eval()
        with torch.no_grad():
            x_f=self.fwd(test_0,v_fore,eps_test)
        return x_f

    def eval_back(self,test_1,v_back,eps_test=None):
        v_back.eval()
        self.scale_m_back.eval()
        with torch.no_grad():
            x_b=self.bwd(test_1,v_back,eps_test)
        return x_b


    def Interp_t(self,x_0,x_1,stage_index,bondary_constrain=False):
        B=x_0.shape[0]
        d=x_0.shape[1]
        assert B%2 == 0
        #t=torch.from_numpy(np.random.uniform(0, 1, B).reshape(-1,1)).cuda()
        if bondary_constrain:
            t1=torch.from_numpy(np.random.choice(self.t_lists_stage[stage_index][:-1],int(B/2))).reshape(-1,1).cuda()
            t2=torch.from_numpy(np.random.choice(self.t_lists_stage[stage_index][1:],int(B/2))).reshape(-1,1).cuda()
            t=torch.cat([t1,t2],dim=0)
        else:
            t=torch.from_numpy(np.random.choice(self.t_lists_stage[stage_index],B)).reshape(-1,1).cuda()
        Z=torch.from_numpy(np.array([np.random.normal(loc=0, scale=1, size=d) for i in range(B)])).cuda()
        t_ceil = self.t_lists_stage[stage_index][-1]
        t_floor = self.t_lists_stage[stage_index][0]
        x_t=((t_ceil-t)/(t_ceil-t_floor))*x_0+((t-t_floor)/(t_ceil-t_floor))*x_1+torch.sqrt(self.eps*((t-t_floor)/(t_ceil-t_floor))*((t_ceil-t)/(t_ceil-t_floor)))*Z
        return x_t,t,t_ceil,t_floor

 
    def Interp_t_for_fine_tuning(self,data_tuple,data_generated,training_fore=True):
        B=data_tuple[0].shape[0]
        d=data_tuple[0].shape[1]
        #t=torch.from_numpy(np.random.uniform(0, 1, B).reshape(-1,1)).cuda()
        stage = np.random.choice(self.length-1)
        t=torch.from_numpy(np.random.choice(self.t_lists_stage[stage],B)).reshape(-1,1).cuda()
        Z=torch.from_numpy(np.array([np.random.normal(loc=0, scale=1, size=d) for i in range(B)])).cuda()
        if training_fore:
            x_0 = data_generated[stage].cuda()
            x_1 = data_tuple[stage+1].cuda()
        else:
            x_0 = data_tuple[stage].cuda()
            x_1 = data_generated[stage+1].cuda()
        t_ceil = self.t_lists_stage[stage][-1]
        t_floor = self.t_lists_stage[stage][0]
        x_t=((t_ceil-t)/(t_ceil-t_floor))*x_0+((t-t_floor)/(t_ceil-t_floor))*x_1+torch.sqrt(self.eps*((t-t_floor)/(t_ceil-t_floor))*((t_ceil-t)/(t_ceil-t_floor)))*Z
        return x_0,x_1,x_t,t,t_ceil,t_floor

    def fwd_one_step(self,x_0,v_m,stage_index):
        B=x_0.shape[0]
        x=[]
        x_t=x_0
        x.append(x_t.detach().cpu())
        t_list_stage=self.t_lists_stage[stage_index]
        t_floors_stage=self.t_lists_stage[stage_index][0]
        for t_new in t_list_stage[:-1]:
            t=torch.from_numpy(np.repeat(t_new,B)).reshape(-1,1).double().cuda()
            x_t=x[-1].cuda()
            if t_new != t_list_stage[-2]:
                z=torch.from_numpy(np.array([np.random.normal(loc=0, scale=self.delta_t, size=self.d) for i in range(B)])).cuda()
                del_x_t=v_m(x_t,self.scale_m_fore(t,t_floors_stage))*self.delta_t+np.sqrt(self.eps)*z
            else:
                del_x_t=v_m(x_t,self.scale_m_fore(t,t_floors_stage))*self.delta_t
            x.append((x_t+del_x_t.cuda()).detach().cpu())
        return x

    def bwd_one_step(self,x_0,v_m,stage_index):
        B=x_0.shape[0]
        x=[]
        x_t=x_0
        x.append(x_t.detach().cpu())
        t_list_stage=self.t_lists_stage[stage_index][::-1]
        t_floors_stage=self.t_lists_stage[stage_index][0]
        for t_new in t_list_stage[:-1]:
            t=torch.from_numpy(np.repeat(t_new,B)).reshape(-1,1).double().cuda()
            x_t=x[-1].cuda()
            if t_new != t_list_stage[-2]:
                z=torch.from_numpy(np.array([np.random.normal(loc=0, scale=self.delta_t, size=self.d) for i in range(B)])).cuda()
                del_x_t=v_m(x_t,self.scale_m_back(t,t_floors_stage))*self.delta_t+np.sqrt(self.eps)*z
            else:
                del_x_t=v_m(x_t,self.scale_m_back(t,t_floors_stage))*self.delta_t
            x.append((x_t+del_x_t.cuda()).detach().cpu())
        return x

    def fwd(self,x_0,v_m,eps_test=None):
        if eps_test!= None:
            eps=eps_test
        else:
            eps=self.eps
        B=x_0.shape[0]
        x=[]
        x_t=x_0
        x.append(x_t.detach().cpu())
        self.t_floors = [timestage[0] for timestage in self.t_lists_stage]
        stage_id = 0
        for step in range(self.steps):
            if stage_id < len(self.t_floors)-1:
                if self.t_list[step] >= self.t_floors[stage_id+1]:
                    stage_id += 1
            t=torch.from_numpy(np.repeat(self.t_list[step],B)).reshape(-1,1).double().cuda()
            x_t=x[-1].cuda()
            if step != self.steps-1:
                z=torch.from_numpy(np.array([np.random.normal(loc=0, scale=self.delta_t, size=self.d) for i in range(B)])).cuda()
                del_x_t=v_m(x_t,self.scale_m_fore(t,self.t_floors[stage_id]))*self.delta_t+np.sqrt(eps)*z
            else:
                del_x_t=v_m(x_t,self.scale_m_fore(t,self.t_floors[stage_id]))*self.delta_t
            if self.clamp:
                del_x_t = torch.clamp(del_x_t,min=self.min,max=self.max)
            x.append((x_t+del_x_t.cuda()).detach().cpu())
        return x

    def bwd(self,x_1,v_m,eps_test=None):
        if eps_test!= None:
            eps=eps_test
        else:
            eps=self.eps
        B=x_1.shape[0]
        x=[]
        x_t=x_1
        x.append(x_t.detach().cpu())
        self.t_floors = [timestage[0] for timestage in self.t_lists_stage]
        stage_id = len(self.t_floors)-1
        for step in range(self.steps):
            if self.t_list[::-1][step] <= self.t_floors[stage_id]:
                stage_id -= 1
            t=torch.from_numpy(np.repeat(self.t_list[::-1][step],B)).reshape(-1,1).double().cuda()
            x_t=x[-1].cuda()
            if step != self.steps-1:
                z=torch.from_numpy(np.array([np.random.normal(loc=0, scale=self.delta_t, size=self.d) for i in range(B)])).cuda()
                del_x_t=v_m(x_t,self.scale_m_back(t,self.t_floors[stage_id]))*self.delta_t+np.sqrt(self.eps)*z
            else:
                del_x_t=v_m(x_t,self.scale_m_back(t,self.t_floors[stage_id]))*self.delta_t
            if self.clamp:
                del_x_t = torch.clamp(del_x_t,min=self.min,max=self.max)
            x.append((x_t+del_x_t.cuda()).detach().cpu())
        return x
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def apply_ema_to_model(self,model, ema_params):
        for name, param in model.named_parameters():
            ema_params[name].mul_(self.decay).add_(param, alpha=1 - self.decay)
            # 将模型参数直接更新为 EMA 值
            param.data.copy_(ema_params[name])

