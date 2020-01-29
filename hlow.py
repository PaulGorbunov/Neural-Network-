import numpy as np
import matplotlib.pyplot as plt
from functools import *
from random import shuffle
from time import time,sleep
from multiprocessing import Process
import pickle

from sys import exit

class Net:
    
    def __init__(self,**opt):
        self.struct = opt['struct']
        self.cl_num = opt['classes_num']
        self.data,size = self.create_data(opt['data'])
        self.labels = opt['labels']
        self.epochs = opt['epochs']
        self.t_data = self.create_data(opt['test'])[0]
        self.t_labels = opt['tlabels']
        
        if 'optimizer_funct' in opt.keys():
            if opt['optimizer_funct'] == 'gd':
                self.grad_f = Net_layers.Optimizer.Gradient_dessent
            else:
                self.grad_f = Net_layers.Optimizer.Adam
        else:
            self.grad_f = Net_layers.Optimizer.Adam
        if 'learning_rate' in opt.keys():
            self.nu = opt['learning_rate']
        else:
            self.nu = 0.009
        if 'mini_batch_size' in opt.keys():
            self.btch_size = (lambda: opt['mini_batch_size'] if opt['mini_batch_size'] < len(self.data) else len(self.data))()
        else:
            self.btch_size = (lambda: 60 if 60 < len(self.data) else len(self.data))()
        if 'activation_funct' in opt.keys():
            self.activ_f = opt['activation_funct']
        else :
            self.activ_f = Net_layers.ReLu
        if 'backup_upd' in opt.keys():
            self.vis_p = opt['backup_upd']
        else:
            self.vis_p = 20
        if 'graph_upd' in opt.keys():
            self.poi_upd = opt['graph_upd']
        else:
            self.poi_upd = 5
        if 'batch_norm' in opt.keys():
            self.flag_b_n = opt['batch_norm']
        else:
            self.flag_b_n = True
        if 'validation_samp_n' in opt.keys():
            self.val_n = opt['validation_samp_n']
            self.val_par = []
        else:
            self.val_n = 1
            self.val_par = []
        
        self.stat_poi = []
        self.construct(size)
    
    @staticmethod
    def info(self = True):
        inf = "struct* classes_num* data* labels* epochs* test* tlabels* optimizer_funct learning_rate mini_batch_size activation_funct backup_upd graph_upd batch_norm validation_samp_n"
        for u in inf.split():
            print(u)
        
    
    def create_data(self,data):
        d_sh = np.shape(data)
        if type(data) != np.ndarray:
            data = np.array(data)
        data.shape = (d_sh[0],1,int(reduce(lambda x,y:x*y,d_sh)/d_sh[0]))
        return data,data[0].size
    
    def cr_use_inds(self):
        lis = [u for u in range(len(self.data))]
        shuffle(lis)
        val = []
        if self.val_n > 0:
            trans_f = lambda x: [u[:(lambda: int(self.val_n) if self.val_n <= np.floor(len(u)*0.1) else int(np.floor(len(u)*0.1)) )()] for u in x]
            data_cl_ind = [[t for t in range(len(self.data)) if self.labels[t] == u] for u in range(self.cl_num)]
            (lambda:[shuffle(data_cl_ind[u]) for u in range(len(data_cl_ind))])()
            data_cl_ind = trans_f(data_cl_ind)
            val = reduce(lambda x,y:x+y,data_cl_ind)
        return [[el for el in  lis if not el in val],val]
        
    def mini_batch(self):
        inds = []
        try :
            (lambda: [inds.append(self.use_inds.pop()) for u in range(self.btch_size)])()
        except IndexError:
            pass
        return [[self.data[u],self.labels[u]] for u in inds]
        
    def construct(self,size):
        ch_size = lambda x: size if x == 0 else self.struct[x-1] 
        self.net_struct = [(lambda :Net_layers.Neu_lay(ch_size(u//2),self.struct[u//2],self.grad_f) if u%2 == 0 else self.activ_f())() for u in range(len(self.struct)*2)]
        self.net_struct += [(lambda: Net_layers.Softmax(self.struct[-1],self.cl_num,self.grad_f) if len(self.struct) > 0 else Net_layers.Softmax(size,self.cl_num,self.grad_f))()]
        self.batch_norm_lay = []
        if(self.flag_b_n):
            self.batch_norm_lay = [Net_layers.BatchNorm() for u in range(len(self.struct))]
            
    def prepare_data(self,data):
        try:
            self.p_size +=1
        except:
            self.p_size = 1
            self.ave_arr = np.zeros(np.shape(data[0][0]))
            self.st_der = np.zeros(np.shape(data[0][0]))
        ave_arr = np.zeros(np.shape(data[0][0]))
        for y in range(np.size(ave_arr)):
            ave_arr[0,y] += sum([d[0][0,y] for d in data])
        ave_arr /= len(data)
        pre_st_der = [(data[i][0] - ave_arr)**2 for i in range(len(data))]
        st_der = np.sqrt(sum(pre_st_der)/len(data)+10**(-8))
        self.ave_arr += ave_arr
        self.st_der += st_der
        return [((data[y][0]-ave_arr)/st_der,data[y][1]) for y in range(len(data))]
    
    def engine(self):
        cur_ep = 1
        self.loss = 0
        inds = self.cr_use_inds()
        self.use_inds = inds[0]
        t1 = time()
        while cur_ep < (self.epochs+1):
            batch = self.prepare_data(self.mini_batch())
            self.batch_proc(batch)
            for r in self.net_struct:
                r.change_param((cur_ep,self.nu))
            if len(self.use_inds) == 0:
                val_val = 0
                if len(inds[1]) > 0:
                    val_val = self.validation(inds[1])
                    self.check_engine(val_val,cur_ep)
                cur_ep +=1
                self.stat_poi.append((-1*self.loss,-1*val_val))
                self.loss = 0
                inds = self.cr_use_inds()
                self.use_inds = inds[0]
                if (((cur_ep/self.epochs)*100)%self.poi_upd == 0):
                    t2 = time()
                    with open("poi",'wb') as fi:
                        pickle.dump((self.stat_poi,round((t2-t1)/60,2)),fi)                    
                if (((cur_ep/self.epochs)*100)%self.vis_p == 0):
                    t2 = time()
                    with open("net",'wb') as f:
                        pickle.dump(self,f)
                    print("---",int((cur_ep/self.epochs)*100),"---",round((t2-t1)/60,2))
                
    def batch_proc(self,batch):
        ch_b = 0
        for s in range(len(self.net_struct)):
            new_data = []
            for d in batch:
                new_data.append(self.net_struct[s].forv_pass(d))
            if (self.flag_b_n and self.net_struct[s].btch_fl):
                new_data = self.batch_norm_lay[ch_b].forv_pass(new_data)
                ch_b +=1
            batch = new_data
        self.loss += np.log(sum([res[0][0][res[1]] for res in batch]))
        ch_b = len(self.batch_norm_lay)-1
        d_res = self.net_struct[-1].back_pass(0)
        for u in self.net_struct[-2::-1]:
            d_res = u.back_pass(d_res)
            if (self.flag_b_n and not u.btch_fl):
                d_res = self.batch_norm_lay[ch_b].back_pass(d_res)
                ch_b -=1
        
    def start(self):
        self.engine()
    
    def check_engine(self,c_v,c_ep):
        reg = 0.00001
        ok_up = 15
        ave_dis = 0.8
        if (len(self.val_par) == 0):
            self.val_par.append([0,c_v])
            self.val_par.append(c_v)
            self.val_par.append(0)
            
        else:
            self.val_par[2] += (lambda: 0 if self.val_par[0][1] - c_v > 0 else 1  )()
            self.val_par[0][0] = abs(self.val_par[0][1] - c_v)
            self.val_par[0][1] = c_v
            self.val_par[1] += c_v
            if (self.val_par[0][0] < (self.val_par[1]/c_ep) * ave_dis) or self.val_par[2] > ok_up :
                self.nu *= np.e**(-1*reg*c_ep)
                self.val_par[2] = 0
            
            
        
    def validation(self,inds,fl = True):
        if fl:
            data = [(self.data[u],self.labels[u]) for u in inds]
        else :
            data = [(self.t_data[u],self.t_labels[u]) for u in range(len(self.t_data))]
        ave_arr = self.ave_arr/self.p_size
        st_der = self.st_der/self.p_size
        data = [((data[y][0]-ave_arr)/st_der,data[y][1]) for y in range(len(data))]
        loss = 0
        pr_ind = 0
        for u in range(len(data)):
            res = data[u]
            c = 0
            for s in self.net_struct:
                res = s.forv_pass(res)
                if (self.flag_b_n and s.btch_fl):
                    res = self.batch_norm_lay[c].data_pass([res])[0]
                    c +=1
            loss += res[0][0,res[1]]
            for s in self.net_struct:
                s.null()
            pr_ind += (np.where(res[0][0] == np.amax(res[0][0]))[0][0] == res[1])
            if not fl and inds:
                    pr_ind += (np.where(res[0][0] == np.amax(res[0][0]))[0][0] == self.t_labels[u])
                    print("class - ",self.t_labels[u])
                    print(res)
                    print(np.where(res[0][0] == np.amax(res[0][0]))[0][0])
        
        if(not fl):
            print("test result =",(pr_ind/len(self.t_labels))*100,"%")
                
        return loss
        
    def test(self,pr = False):
        self.validation(pr,False)

    
    def draw(self):
        try:
            val = [t[1] for t in self.stat_poi]
            loss = [t[0] for t in self.stat_poi]
            proc = int(np.ceil(len(val)/100))
            val = reduce(lambda x,y:x+y,[[sum(val[t:t+proc])/proc]*proc for t in range(0,len(val),proc)])
            loss = reduce(lambda x,y:x+y,[[sum(loss[t:t+proc])/proc]*proc for t in range(0,len(loss),proc)])
            plt.plot([u for u in range(len(self.stat_poi))],[t[0] for t in self.stat_poi],"b-")
            plt.plot([u for u in range(len(self.stat_poi))],val,"g--")
            plt.title("loss")
            plt.grid(True)
            plt.show()
        except AttributeError:
            print ("not available now")
            

class Net_layers:
    
    class Neurons:
        def __init__(self,n_in,n_out,opt):
            self.HE_init_val(n_in,n_out)
            self.optimize = opt()
            
        def HE_init_val(self,n_in,n_out):
            a = 1/np.sqrt(n_in/2)
            a *= 0.5
            b_cont = 10
            disp = (lambda: 0.0 if n_in < n_out else -0.0)() 
            w = np.random.uniform(low=a*(n_in-disp), high=a*(n_out+disp), size=(n_in,n_out))
            b = np.random.uniform(low=b_cont*a*(n_in-disp), high=b_cont*a*(n_out+disp), size=(1,n_out))
            self.init(w,b)
            
        def sum_all_delt(self):
            new_arr = [np.zeros(np.shape(self.delt[0][1])),np.zeros(np.shape(self.delt[0][2]))]
            for l in self.delt:
                new_arr[0] += l[1]
                new_arr[1] += l[2]
            self.delt = new_arr
            
        
        def change_param(self,par):
            self.sum_all_delt()
            self.optimize.change(self.w,self.b,self.delt,par)
            self.null()
        
        def null(self):
            self.delt = []
            self.cache = []
            
    class Neu_lay(Neurons) :
        
        def init(self,w,b):
            self.btch_fl = True
            self.w = w
            self.b = b
            self.cache = []
            self.delt = []
            
        def forv_pass(self,x):
            m = np.dot(x[0],self.w)
            s = m+np.array(self.b)
            self.cache.append([x,m,s])
            return [s,x[1]]  
        
        def back_pass(self,dx):
            res = []
            for u in range(len(self.cache)):
                d_x = np.dot(dx[u],np.transpose(self.w))
                d_w = np.dot(np.transpose(self.cache[u][0][0]),dx[u])
                self.delt.append([d_x,d_w,dx[u]])
                res.append(d_x)
            return d_x
        
    class Softmax(Neu_lay):
        
        def init(self,w,b):
            self.btch_fl = False
            self.w = w
            self.b = b
            self.cache = []
            self.delt = []
        
        def forv_pass(self,x):
            res = super().forv_pass(x)
            self.softmax_layer(res)
            return [self.cache[-1][-1],self.cache[-1][0][1]] 
        
        def softmax(self,lis):
            e_sum = np.sum(np.vectorize(lambda x:np.e**x)(lis))
            return np.vectorize(lambda x: (np.e**x)/e_sum)(lis)
    
        def softmax_layer(self,lis):
            self.cache[-1][-1] = self.softmax(lis[0])
            self.cache[-1][-2] = np.array(self.w)

        def back_pass(self,dx):
            res = []
            for u in range(len(self.cache)):
                self.cache[u][-1][0,self.cache[u][0][1]] -=1 
                gr_x = np.dot(self.cache[u][-1],np.transpose(self.cache[u][-2]))
                gr_w = np.dot(np.transpose(self.cache[u][0][0]),self.cache[u][-1])
                self.cache[u][0] = gr_x;
                self.cache[u][1] = gr_w;
                self.delt.append(self.cache[u])
                res.append(gr_x)
            return res
            
    class ReLu:
        def __init__(self):
            self.btch_fl = False
            
        def change_param(self,nu):
            pass
        
        def null(self):
            pass
        
        def forv_pass(self,lis):
            return [np.vectorize(lambda x: x if x>0.0 else 0.0)(lis[0]),lis[1]]
        
        def back_pass(self,lis):
            return [np.vectorize(lambda x: x if x>0.0 else 0.0)(l) for l in lis]
    
    class BatchNorm:
        def __init__(self):
            self.eps = 10**(-8)
            self.cache = []
        
        def change_param(self,nu):
            pass
        
        def null(self):
            self.cache = []
            
        def forv_pass(self,data):
            try:
                self.p_size +=1
            except:
                self.p_size = 1
                self.ave_arr = np.zeros(np.shape(data[0][0]))
                self.st_der = np.zeros(np.shape(data[0][0]))
            ave_arr = np.zeros(np.shape(data[0][0]))
            for y in range(np.size(ave_arr)):
                ave_arr[0,y] += sum([d[0][0,y] for d in data])
            ave_arr /= len(data)
            delt_x = [data[y][0]-ave_arr for y in range(len(data))] 
            pre_st_der = [delt_x[i]**2 for i in range(len(data))] 
            var = sum(pre_st_der)/len(data)
            st_der = np.sqrt(var+self.eps) 
            if (self.p_size == 1):
                self.y = st_der
                self.b = ave_arr
            delt_shi_x =[delt_x[y]/st_der for y in range(len(data))] 
            self.cache = [np.array(delt_shi_x),np.array(delt_x),st_der,var]
            self.ave_arr += ave_arr
            self.st_der += st_der
            return [(self.y*delt_shi_x[y]+self.b,data[y][1]) for y in range(len(data))]
        
        def back_pass(self,dx):
            dx_h = np.array(dx)*self.y
            dups_der = reduce(lambda x,y:x+y,dx_h*self.cache[1])
            d_delt_x = dx_h/self.cache[2]
            d_std_der = dups_der*(-1/np.power(self.cache[2],2))
            d_var = d_std_der/(2*self.cache[2])
            d_av = np.ones(np.shape(dx))/len(dx)*d_var
            d_delt1_x = 2*self.cache[1]*d_av
            dx1 = d_delt1_x+d_delt_x
            dmu = -1*reduce(lambda x,y:x+y,dx1)
            dx2 = np.ones(np.shape(dx))/len(dx)*dmu
            dx = dx1+dx2
            self.y -= reduce(lambda x,y:x+y,np.array(dx)*np.array(self.cache[0]))
            self.b -= reduce(lambda x,y:x+y,dx)
            self.null()
            return dx
        
        def data_pass(self,data):
            ave_arr = self.ave_arr/self.p_size
            st_der = self.st_der/self.p_size
            return [(self.y*((data[y][0]-ave_arr)/st_der)+self.b,data[y][1]) for y in range(len(data))]
        
    class Optimizer:
        
        class Adam:
            def __init__(self):
                self.vw = 0
                self.sw = 0
                self.vb = 0
                self.sb = 0
            
            def change(self,w,b,delt,t):
                b1 = 0.9
                b2 = 0.999
                si = 10**(-8)
                self.vw = b1*self.vw + (1-b1)*delt[0]
                self.vb = b1*self.vb + (1-b1)*delt[1]
                self.sw = b2*self.sw + (1-b2)*np.power(delt[0],2)
                self.sb = b2*self.sb + (1-b2)*np.power(delt[1],2)
                cvw = self.vw/(1-(b1**t[0]))
                cvb = self.vb/(1-(b1**t[0]))
                csw = self.sw/(1-(b2**t[0]))
                csb = self.sb/(1-(b2**t[0]))
                w -= t[1] * cvw/(np.sqrt(csw)+si)
                b -= t[1] * cvb/(np.sqrt(csb)+si)
        
        class Gradient_dessent:
            @staticmethod
            def change(self,w,b,delt,nu):
                self.w -= nu[1]*delt[0]
                self.b -= nu[1]*delt[1]
                
class Graph:
    def __init__(self):
        self.last_pont = []
    
    def draw(self):
        plt.figure()
        while True :
            try:
                with open('poi','rb') as f:
                    s = pickle.load(f)
            except EOFError:
                sleep(1)
                continue
            ys = s[0][0]
            xs = [u for u in range(len(ys))]
            plt.cla() 
            plt.plot(xs,ys,"g-")
            v_ys = s[0][1]
            v_xs = [u for u in range(len(v_ys))]
            plt.plot(v_xs,v_ys,"g-")
            if len(self.last_pont) > 0 and (self.last_pont[0]+2 <= len(ys)+1 ): 
                plt.plot(self.last_pont[0],self.last_pont[1],"r*")
                plt.title("cur_ep = "+str(len(ys)+1)+" *ep = "+str(self.last_pont[0]+2)+str(" t: ")+str(s[1]))
            else:
                plt.title("cur_ep = "+str(len(ys))+str(" t: ")+str(s[1]))
            plt.grid(True)
            plt.pause(25)
            try:
                self.last_pont = [xs[-1],ys[-1]]
            except IndexError:
                pass
        plt.show()
        
        
if __name__ == "__main__":
    with open("poi",'wb') as fi:
        pickle.dump(([],0),fi)  
    g = Graph()
    g.draw()

