import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
def log_odds(x):
  return np.log(x/(1-x))
class Model():
  def __init__(self,mode) -> None:
    self.mode=mode
    self.x0=[0.75,0,0.1,0.02]
    self.bounds=[(0.5,1),(-1,1),(0,10),(0,0.03)]
    self.maxt=2000
    self.n_class=4
    self.dt=10
    self.b0=300

    self.t0=0
    self.tau=500

    self.delta=0.01
    self.lapse=1e-2/self.n_class
    self.conf_sigma=0

    self.v0=0

    self.base_1=100
    self.base_2=100

    self.k0=5
    self.lamb=0

  def conf2input(self,x):
    # return np.log(x/(1-x))
    return x-0.5

  def get_sub_data(self,ori_data,sub,extended=False,return_raw=False):
  
    conf=ori_data['conf'][:,0]
    conf2=ori_data['conf2'][:,0]
    if self.mode=='nonsocial':
      info=ori_data['inforate'][:,0]
    else:
      info=ori_data['oconf'][:,0]
    change=ori_data['ischange'][:,0]
    rt=ori_data['rtime2'][:,0]
    isconflict=ori_data['isconflict'][:,0]
    isc2=ori_data['isc2'][:,0]


    data=np.zeros((conf[sub].shape[0],5))
    data[:,0]=change[sub][:,0]
    data[:,1]=rt[sub][:,0]
    data[:,2]=conf2[sub][:,0]
    data[:,3]=conf[sub][:,0]
    data[:,4]=info[sub][:,0]
    data=data[(isconflict[sub]==1).flatten(),:]
    if return_raw:
      import copy
      raw=copy.deepcopy(data)
    # data[:,-2:]=log_odds(data,-2:)

    data[:,-2]+=0.8-np.mean(data[:,-2])
    data[:,-1]+=0.8-np.mean(data[:,-1])

    if return_raw:
      data=(raw,data)
    

    if extended:
      return data,isc2[sub][(isconflict[sub]==1).flatten(),0]

    else:
      return data
  def init_paras(self,paras):
    self.thr1,q,self.sigma,self.lamb=paras
    
    self.k2=self.k0*(1+q)
    self.k1=self.k0*(1-q)
    self.sigma*=self.k0

  def simulate(self,inputs,return_x=False,discrete_conf=True):
    v1,v2,b1,b2=inputs
    x=np.zeros((int(self.maxt/self.dt),2))
    x[0,0]+=self.base_1+b1*self.b0
    x[0,1]+=self.base_2+b2*self.b0
    noise=np.random.randn(x.shape[0],x.shape[1])
    bound=self.get_boundary(np.array(range(x.shape[0]))*self.dt)

    A=np.array([[1-self.delta*self.dt,-self.lamb*self.dt],[-self.lamb*self.dt,1-self.delta*self.dt]])
    b=np.array([v1*self.dt+self.v0*self.dt,v2*self.dt+self.v0*self.dt])
    dsigma=self.sigma*np.sqrt(self.dt)
    choice=-100
    rt=-100
    conf=-100
    
    for t in range(1,x.shape[0]):
      x[t]=x[t-1]@A+b+dsigma*noise[t]
      x[t][x[t]<0]=0
    
      if x[t][0]>bound[t] and x[t][0]>x[t][1]:
        choice=1
        rt=t*self.dt
        conf=self.evd2conf(x[t][0],x[t][1],discrete=discrete_conf)
        break
        
      if x[t][1]>bound[t]:
        choice=2
        rt=t*self.dt
        conf=self.evd2conf(x[t][1],x[t][0],discrete=discrete_conf)
        break

    if return_x:
      return choice,rt,conf,x
    return choice,rt,conf
  def get_input(self,data):
    raise('How to get input is not defined')

  def evd2conf(self,e1,e2,discrete=True):
    
    x1=e1/self.b0/2
    x2=e2/self.b0/2
    ret=self.conf_sigma*np.random.randn()+x1-x2+0.5
    if discrete:
      ret=self.to_discrete_conf(ret)
    return ret

  def get_boundary(self,t):
    try:
      if t<=self.t0:
        return self.b0
      return self.b0*(1-0.9*(t-self.t0)/(t-self.t0+self.tau))
    except:
      idx=t<=self.t0
      ret=self.b0*(1-0.9*(t-self.t0)/(t-self.t0+self.tau))
      ret[idx]=self.b0
      return ret

  def get_trial_score(self,data,repn=10,prob=None):
    
    change,rt,conf,sconf,info=data
    choice=change+1
    if prob is None:
      model_data=np.zeros((repn,3))
      for i in range(repn):
        pd_choice,pd_rt,pd_conf=self.simulate(self.get_input(data))
        
        model_data[i,:]=np.array((pd_choice,pd_rt,pd_conf))
        
      p=(np.sum((model_data[:,0]==choice)*(model_data[:,2]==conf))/model_data.shape[0])
      score=-np.log(p*(1-self.lapse*self.n_class)+self.lapse)
    else:
      p=(prob[int(choice-1),int(conf-1)])
      score=-np.log(p*(1-self.lapse*self.n_class)+self.lapse)
    return score
  

  def get_prob(self,data,repn=100):
    inputs,idx=np.unique(data[:,3:],1,axis=0)
    prob_list=[]
    
    for j in range(inputs.shape[0]):
      input=data[idx[j]]
      model_data=np.zeros((repn,3))
      for i in range(repn):
        pd_choice,pd_rt,pd_conf=self.simulate(self.get_input(input))
        
        model_data[i,:]=np.array((pd_choice,pd_rt,pd_conf))
        
      prob=np.zeros((2,4))
      for choice in range(2):
        for conf in range(4):
          p=(np.sum((model_data[:,0]==(choice+1))*(model_data[:,2]==(conf+1)))/model_data.shape[0])
          prob[choice,conf]=p
      prob_list.append(prob)
    return inputs,np.array(prob_list)   
  
  
  def get_sub_score(self,paras,data,max_trial=999,mode=0,repn=10,prob=None):
    self.init_paras(paras)
    score=0

    import copy
    to_fit=copy.deepcopy(data)
    
    to_fit[:,2]=self.to_discrete_conf(to_fit[:,2])


    inputs,prob=self.get_prob(to_fit)
    if mode==1:
      loop=range(0,np.minimum(max_trial,to_fit.shape[0]),2)
    elif mode==2:
      loop=range(1,np.minimum(max_trial,to_fit.shape[0]),2)
    else:
      # np.random.shuffle(to_fit)
      loop=range(np.minimum(max_trial,to_fit.shape[0]))
    cnt=0
    for i in loop:
      cnt+=1
      j=np.where(np.all(to_fit[i,3:]==inputs,axis=1))[0].item()
      score+=self.get_trial_score(to_fit[i],repn=repn,prob=prob[j])
    
    return score/cnt
  
  def to_discrete_conf(self,conf):
    try:
      ## simulation 
      ret=1
      if conf>=self.thr1:
        ret+=1
      return ret
    
    except:
      ## data array
      ret=np.ones_like(conf)
      ret[conf>=0.75]+=1

      return ret
  
  def get_trial_ll(self,data,repn=10):
    
    change,rt,conf,sconf,info=data
    choice=change+1
   
    model_data=np.zeros((repn,3))
    for i in range(repn):
      pd_choice,pd_rt,pd_conf=self.simulate(self.get_input(data))
      model_data[i,:]=np.array((pd_choice,pd_rt,pd_conf))
      
    p=(np.sum((model_data[:,0]==choice)*(model_data[:,2]==conf))/model_data.shape[0])
    score=-np.log(p*(1-self.lapse*self.n_class)+self.lapse)

    rt=model_data[:,1]
    p_choice=(np.sum((model_data[:,0]==choice))/model_data.shape[0])
    p_choice=-np.log(p_choice*(1-self.lapse*self.n_class)+self.n_class/2*self.lapse)
    p_conf=(np.sum((model_data[:,2]==conf))/model_data.shape[0])
    p_conf=-np.log(p_conf*(1-self.lapse*self.n_class)+2*self.lapse)
    return score,(p_choice,p_conf,rt)
  
  def get_sub_ll(self,paras,data,max_trial=999,mode=0,repn=10):
    self.init_paras(paras)
    score=0

    import copy
    to_fit=copy.deepcopy(data)
    to_fit[:,2]=self.to_discrete_conf(to_fit[:,2])

    if mode==1:
      loop=range(0,np.minimum(max_trial,to_fit.shape[0]),2)
    elif mode==2:
      loop=range(1,np.minimum(max_trial,to_fit.shape[0]),2)
    else:
      np.random.shuffle(to_fit)
      loop=range(np.minimum(max_trial,to_fit.shape[0]))
    rt_list=[]
    data_rt=[]
    dv_list=[]
    choice_term=0
    conf_term=0
    for i in loop:
      s,(c1,c2,rt)=self.get_trial_ll(to_fit[i],repn=repn)
      score+=s
      choice_term+=c1
      conf_term+=c2
      rt_list.append(rt)
      data_rt.append(data[i,1])
      dv_list.append(data[i,-2]-data[i,-1])
    rt_list=np.array(rt_list)
    dv_list=np.array(dv_list)
    dv=data[:,-2]-data[:,-1]
    rt=self.transform(rt_list,data[:,1],np.broadcast_to(dv_list.reshape((-1,1)),rt_list.shape),dv)
    rt=np.mean(rt,-1)
    ssr=np.sum((rt-np.array(data_rt))**2)
    n=rt.shape[0]
    rt_term=0.5*np.log(ssr/n)*n
    # score+=0.5*np.log(ssr/n)*n
      
    return score,n,(choice_term,conf_term,rt_term)
  

  def transform(self,source,target,v1=None,v2=None):

    if v1 is None or v2 is None:
      import copy
      ret=copy.deepcopy(source)
      if len(np.unique(ret))==1:
        ret=np.mean(target)
        return ret
      ret-=np.mean(ret)
      ret/=np.std(ret)
      ret*=np.std(target)
      ret+=np.mean(target)
      return ret
    from scipy.stats import linregress
    nq=4

    e=np.linspace(np.min(v2)-0.01,np.max(v2)+0.01,nq+1)
    m1= (np.histogram(v1, e, weights=source)[0] /np.histogram(v1, e)[0])
    m2= (np.histogram(v2, e, weights=target)[0] /np.histogram(v2, e)[0])
    if not len(np.unique(m1))==1:
      res=linregress(m1,m2)

      ret=source*res.slope+res.intercept
    else:
      ret=source+np.mean(m2)-np.mean(m1)
    return ret
   
   
class Drift_rate_model(Model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)

  def get_input(self,data):
    
    change,rt,conf,sconf,info=data
    v1=self.conf2input(sconf)
    v2=self.conf2input(info)
    return self.k1*v1,self.k2*v2,0,0
  

class Baseline_model(Model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)
    
  def get_input(self,data):
    
    change,rt,conf,sconf,info=data
    b1=self.conf2input(sconf)
    b2=self.conf2input(info)
    return 0,0,self.k1*b1*self.b0,self.k2*b2*self.b0
  

class No_mi_model(Model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)

    self.x0=[0.75,0,0.1]
    self.bounds=[(0.5,1),(-1,1),(0,10)]
    

  def init_paras(self,paras):
    self.thr1,q,self.sigma=paras
    
    self.k2=self.k0*(1+q)
    self.k1=self.k0*(1-q)
    self.sigma*=self.k0


class No_urg_model(Model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)
    self.x0=[0.75,0,0.1,0.02]
    self.bounds=[(0.5,1),(-1,1),(0,10),(0,0.03)]

  def init_paras(self,paras):
    self.thr1,q,self.sigma,self.lamb=paras
    
    self.k2=self.k0*(1+q)
    self.k1=self.k0*(1-q)
    self.sigma*=self.k0


class Minimal_model(Model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)
    self.x0=[0.75,0]
    self.bounds=[(0.5,1),(-1,1),(0,10),]

  def init_paras(self,paras):
    self.thr1,q,self.sigma=paras
    
    self.k2=self.k0*(1+q)
    self.k1=self.k0*(1-q)
    self.sigma*=self.k0




class Drift_rate_no_mi_model(Drift_rate_model,No_mi_model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)
    


class Drift_rate_no_urg_model(Drift_rate_model,No_urg_model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)
    

class Drift_rate_minimal_model(Drift_rate_model,Minimal_model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)
    


class Baseline_no_mi_model(Baseline_model,No_mi_model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)
    


class Baseline_no_urg_model(Baseline_model,No_urg_model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)
    

class Baseline_minimal_model(Baseline_model,Minimal_model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)
    
def eval(model_class,ori_data,mode):
  
  model=model_class(mode=mode)
  all_res=np.load(f'{model_class.__name__}_paras.npy',allow_pickle=True)
  dat=np.array([r.x for r in all_res])
  
  

  res=np.zeros((ori_data.shape[0],3))
  other_res=np.zeros((ori_data.shape[0],3))
  all_data=np.zeros((0,5))
  all_prd=np.zeros((0,3))
  prob_list=[]
  prd_list=[]
  for sub in range(ori_data.shape[0]) :
    data=model.get_sub_data(ori_data,sub)


    all_data=np.concatenate((all_data,data),0)
    paras=dat[sub]
    model.init_paras(paras)
    
    
    print(f'Sub id:{sub}')
    print(f'Paras:{paras}')

    res[sub,0],n,(other_res[sub,0],other_res[sub,1],other_res[sub,2])=model.get_sub_ll(paras,data,repn=10)
    k=len(paras)
    res[sub,1]=2*k+2*res[sub,0]
    res[sub,2]=np.log(n)*k+2*res[sub,0]
    print(f'Score:{res[sub,0]}')
    print(f'AIC:{res[sub,1]}')
    print(f'BIC:{res[sub,2]}')
    
 
    pass
  np.save(f'{model_class.__name__}_eval.npy',{'probability':prob_list,'prediction':prd_list,'score':res,'other':other_res})

  return res


def fit(model_class,ori_data,mode):
  
  
  model=model_class(mode=mode)
  

  to_save=[]
  for sub in range(ori_data.shape[0]) :

    data=model.get_sub_data(ori_data,sub)
    print(f'Start fitting subject {sub}')
    from scipy.optimize import differential_evolution
    res=differential_evolution(model.get_sub_score,model.bounds,args=(data,),maxiter=100,x0=model.x0,workers=16,disp=True)

    print(res)
    to_save.append(res)
    np.save(f'{model_class.__name__}_paras.npy',to_save)

    pass


def init(social=False):
  import argparse
  parser = argparse.ArgumentParser()
  import os

  from scipy.io import loadmat
  parser.add_argument('-s','--social', action='store_true')
  parser.add_argument('-m', '--model', default='12345678')
  args = parser.parse_args()
  
 
  file_path=os.path.dirname(os.path.abspath(__file__))
  if args.social or social:
    mode='social'
    ori_data=loadmat(f'{file_path}/ALLSOCIAL')['alldataSOCIAL']
  else:
    mode='nonsocial'
    ori_data=loadmat(f'{file_path}/ALLNONSOCIAL')['alldataNONSOCIAL']
    

  if not os.path.isdir(f'./{mode}'):
    try:
      os.mkdir(f'./{mode}')
    except:
      pass 
  os.chdir(f'./{mode}')  
  return args,mode,ori_data

if __name__=='__main__':
  args,mode,ori_data=init()
  
  models=[Drift_rate_model,Drift_rate_no_mi_model,Drift_rate_no_urg_model,Drift_rate_minimal_model]
  models+=[Baseline_model,Baseline_no_mi_model,Baseline_no_urg_model,Baseline_minimal_model]
  # names=[model.__name__ for model in models]
  names=['Drift rate model + MI + Urg','Drift rate model + Urg','Drift rate model + MI','Drift rate model']
  names+=['Baseline model + MI + Urg','Baseline model + Urg','Baseline model + MI','Baseline model']
  
  for i,model in enumerate(models):
    if args.model.find(f'{i+1}')<0:
      continue
    fit(model,ori_data,mode)
    eval(model,ori_data,mode)
  



  import os
  os.chdir('..')
