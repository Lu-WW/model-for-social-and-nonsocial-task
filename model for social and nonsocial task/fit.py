import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
def log_odds(x):
  return np.log(x/(1-x))
class Model():
  def __init__(self,mode) -> None:
    self.mode=mode
    self.x0=[0.1,0,0,0.5,1,0.0,2000,100]
    self.bounds=[(0,1),(0,300),(0,300),(0,5),(0.01,30),(0.,0.03),(0,5000),(0,5000)]
    self.maxt=5000

    self.dt=10
    self.conf0=0
    self.base_1=0
    self.base_2=0
    self.b0=300
    self.delta=0.01
    self.v0=0.1
    self.lamb=0
    self.t0=1e9
    self.tau=1e9

    self.lapse=1e-2



  def get_sub_data(self,ori_data,sub,extended=False):
  
    conf=ori_data['conf'][:,0]
    conf2=ori_data['conf2'][:,0]
    oconf=ori_data['oconf'][:,0]
    change=ori_data['ischange'][:,0]
    rt=ori_data['rtime2'][:,0]
    isconflict=ori_data['isconflict'][:,0]
    isc2=ori_data['isc2'][:,0]
    
    data=np.zeros((conf[sub].shape[0],5))
    data[:,0]=change[sub][:,0]
    data[:,1]=rt[sub][:,0]
    data[:,2]=conf2[sub][:,0]
    data[:,3]=conf[sub][:,0]
    data[:,4]=oconf[sub][:,0]
    data=data[(isconflict[sub]==1).flatten(),:]

    if extended:
      return data,isc2[sub][(isconflict[sub]==1).flatten(),0]

    else:
      return data
  def init_paras(self,paras):
    self.v0,self.base_1,self.base_2,self.k,self.sigma,self.lamb,self.tau,self.t0=paras

  def simulate(self,inputs):
    raise('Simulation details is not defined')
  
  def get_input(self,data):
    raise('How to get input is not defined')

  def evd2conf(self,e1,e2):
    # ret=(np.array(x)/self.b0)
    
    x1=e1/self.b0/2+0.5
    x2=e2/self.b0/2+0.5
    eps=self.lapse*8
    x1=np.clip(x1,eps,1-eps)
    x2=np.clip(x2,eps,1-eps)
    ret=1/(1+np.exp(log_odds(x2)-log_odds(x1)))
    
    ret=self.to_discrete_conf(ret)
    return ret
  def conf2evd(self,conf):
    ret=((np.array(conf))*self.b0)
    raise('err')
    return ret
  def evd2id(self,x):
    ret=(np.array(x)/(self.b0+1e-4)*self.n_evd).astype(int)
    return ret
  def id2evd(self,id):
    ret=np.array(id)*self.b0/self.n_evd
    return ret
  def rt2id(self,rt):
    ret=int(rt/self.dt)
    return ret

  def get_boundary(self,t):
    if t<=self.t0:
      return self.b0
    return self.b0*(1-0.9*(t-self.t0)/(t-self.t0+self.tau))

  def get_trial_score(self,data,repn=10,prob=None):
    
    change,rt,conf,sconf,oconf=data
    choice=change+1
    if prob is None:
      model_data=np.zeros((repn,3))
      for i in range(repn):
        pd_choice,pd_rt,pd_conf=self.simulate(self.get_input(data))
        
        model_data[i,:]=np.array((pd_choice,pd_rt,pd_conf))
        
      p=(np.sum((model_data[:,0]==choice)*(model_data[:,2]==conf))/model_data.shape[0])
      score=-np.log(p*(1-self.lapse*8)+self.lapse)
    else:
      score=-np.log(prob[int(choice==2),int(conf==1)])
    return score
    
  def get_sub_score(self,paras,data,max_trial=999,mode=0,repn=10,prob=None):
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
      # np.random.shuffle(to_fit)
      loop=range(np.minimum(max_trial,to_fit.shape[0]))
    for i in loop:
      if prob is None:
        score+=self.get_trial_score(to_fit[i],repn=repn)
      else:
        score+=self.get_trial_score(to_fit[i],repn=repn,prob=prob[i])
      
    return score
  def to_discrete_conf(self,conf):
    try:
      if conf<0.625:
        return 1
      if conf<0.75:
        return 2
      if conf<0.875:
        return 3
      return 4
    
    except:
      ret=np.ones_like(conf)
      ret[conf>=0.625]+=1
      ret[conf>=0.75]+=1
      ret[conf>=0.875]+=1
      return ret
  def get_trial_ll(self,data,repn=10):
    
    change,rt,conf,sconf,oconf=data
    choice=change+1
   
    model_data=np.zeros((repn,3))
    for i in range(repn):
      pd_choice,pd_rt,pd_conf=self.simulate(self.get_input(data))
      model_data[i,:]=np.array((pd_choice,pd_rt,pd_conf))
      
    p=(np.sum((model_data[:,0]==choice)*(model_data[:,2]==conf))/model_data.shape[0])
    score=-np.log(p*(1-self.lapse*8)+self.lapse)

    rt=model_data[:,1]
    p_choice=(np.sum((model_data[:,0]==choice))/model_data.shape[0])
    p_choice=-np.log(p_choice*(1-self.lapse*8)+2*self.lapse)
    p_conf=(np.sum((model_data[:,2]==conf))/model_data.shape[0])
    p_conf=-np.log(p_conf*(1-self.lapse*8)+2*self.lapse)
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

    e=np.quantile(v2,np.linspace(0,1,nq+1)) 
    m1= (np.histogram(v1, e, weights=source)[0] /np.histogram(v1, e)[0])
    m2= (np.histogram(v2, e, weights=target)[0] /np.histogram(v2, e)[0])
    if not len(np.unique(m1))==1:
      res=linregress(m1,m2)

      ret=source*res.slope+res.intercept
    else:
      ret=source+np.mean(m2)-np.mean(m1)
    return ret
   
  def compare(self,paras,data,repn=10):
    self.init_paras(paras)
    prd=np.zeros((data.shape[0],repn,3))
    ret=np.zeros((data.shape[0],3))


      
    def get_prd(data,repn=10):
      
   
      ret=np.zeros((repn,3))
      for i in range(repn):
        pd_choice=-1
        while (pd_choice<0):
          pd_choice,pd_rt,pd_conf=self.simulate(self.get_input(data))
          
        ret[i]=np.array([pd_choice,pd_rt,pd_conf])
        
      return ret
    for i in range(data.shape[0]):
      
      prd[i]=get_prd(data[i],repn=repn)


    ret[:,0]=np.mean(prd[:,:,0],1)

    dv=(data[:,-2]-data[:,-1]).reshape((-1,1))
    dv=np.broadcast_to(dv,prd[:,:,1].shape)
    prd[:,:,1]=self.transform(prd[:,:,1],data[:,1],dv,dv[:,0])
    ret[:,1]=np.mean(prd[:,:,1],1)
    # m=0.75
    
    # idx=prd[:,:,2]>self.conf0
    # prd[:,:,2][idx]=self.transform(prd[:,:,2][idx],data[:,2][data[:,2]>m])
    # prd[:,:,2][~idx]=self.transform(prd[:,:,2][~idx],data[:,2][data[:,2]<=m])


    ret[:,2]=np.mean(prd[:,:,2],1)/8-1/16+0.5

    return ret


def conf2input(x):
  # return np.log(x/(1-x))
  return x-0.5
class Drift_rate_model(Model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)

  def get_input(self,data):
    
    change,rt,conf,sconf,oconf=data

    return conf2input(sconf),conf2input(oconf),0,0
    # return sconf,oconf,0,0
  
   
  def simulate(self,inputs,return_x=False):
    v1,v2,b1,b2=inputs
    x=np.zeros((int(self.maxt/self.dt),2))
    x[0,0]+=self.base_1
    x[0,1]+=self.base_2
    A=np.array([[1-self.delta*self.dt,-self.lamb*self.dt],[-self.lamb*self.dt,1-self.delta*self.dt]])
    b=np.array([self.k*v1*self.dt+self.v0*self.dt,self.k*v2*self.dt+self.v0*self.dt])
    dsigma=self.sigma*np.sqrt(self.dt)
    choice=-100
    rt=-100
    conf=-100
    for t in range(1,x.shape[0]):
      x[t]=x[t-1]@A+b+dsigma*np.random.randn(2)
      x[t][x[t]<0]=0
    
      if x[t][0]>self.get_boundary(t*self.dt) and x[t][0]>x[t][1]:
        choice=1
        rt=t*self.dt
        conf=self.evd2conf(x[t][0],x[t][1])
        break
        
      if x[t][1]>self.get_boundary(t*self.dt):
        choice=2
        rt=t*self.dt
        conf=self.evd2conf(x[t][1],x[t][0])
        break

    if return_x:
      return choice,rt,conf,x
    
    return choice,rt,conf

class Baseline_model(Model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)
    
  def get_input(self,data):
    
    change,rt,conf,sconf,oconf=data
    return 0,0,conf2input(sconf)*self.b0,conf2input(oconf)*self.b0
    # return 0,0,sconf*self.b0,oconf*self.b0
  

  def simulate(self,inputs,return_x=False):
    v1,v2,b1,b2=inputs
    x=np.zeros((int(self.maxt/self.dt),2))
    x[0,0]+=self.base_1+b1*self.k
    x[0,1]+=self.base_2+b2*self.k
    A=np.array([[1-self.delta*self.dt,-self.lamb*self.dt],[-self.lamb*self.dt,1-self.delta*self.dt]])
    b=np.array([self.v0*self.dt,self.v0*self.dt])
    dsigma=self.sigma*np.sqrt(self.dt)
    choice=-100
    rt=-100
    conf=-100
    for t in range(1,x.shape[0]):
      x[t]=x[t-1]@A+b+dsigma*np.random.randn(2)
      x[t][x[t]<0]=0
    
      if x[t][0]>self.get_boundary(t*self.dt) and x[t][0]>x[t][1]:
        choice=1
        rt=t*self.dt
        conf=self.evd2conf(x[t][0],x[t][1])
        break
        
      if x[t][1]>self.get_boundary(t*self.dt):
        choice=2
        rt=t*self.dt
        conf=self.evd2conf(x[t][1],x[t][0])
        break

    if return_x:
      return choice,rt,conf,x
    return choice,rt,conf

class No_mi_model(Model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)

    self.x0=[0.1,0,0,3,1,2000,100]
    self.bounds=[(0,1),(0,300),(0,300),(0,10),(0.01,30),(0,5000),(0,5000)]
    

  def init_paras(self,paras):
    self.v0,self.base_1,self.base_2,self.k,self.sigma,self.tau,self.t0=paras

class No_urg_model(Model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)
    self.x0=[0.1,0,0,3,1,0]
    self.bounds=[(0,1),(0,300),(0,300),(0,10),(0.01,30),(0,0.03)]

  def init_paras(self,paras):
    self.v0,self.base_1,self.base_2,self.k,self.sigma,self.lamb=paras

class Minimal_model(Model):
  def __init__(self,**kwargs) -> None:
    super().__init__(**kwargs)
    self.x0=[0.1,0,0,3,1]
    self.bounds=[(0,1),(0,300),(0,300),(0,10),(0.01,30),]

  def init_paras(self,paras):
    self.v0,self.base_1,self.base_2,self.k,self.sigma=paras



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

    res[sub,0],n,(other_res[sub,0],other_res[sub,1],other_res[sub,2])=model.get_sub_ll(paras,data,repn=100)
    k=len(paras)
    res[sub,1]=2*k+2*res[sub,0]
    res[sub,2]=np.log(n)*k+2*res[sub,0]
    print(f'Score:{res[sub,0]}')
    print(f'AIC:{res[sub,1]}')
    print(f'BIC:{res[sub,2]}')
    
    # prd=model.compare(paras,data,repn=100)
    # prd_list.append(prd)
    # all_prd=np.concatenate((all_prd,prd),0)
 
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
    res=differential_evolution(model.get_sub_score,model.bounds,args=(data,),polish=False,maxiter=50,x0=model.x0,workers=16,disp=True)

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
  parser.add_argument('-e', '--eval', action='store_true')
  parser.add_argument('-p', '--precomputed', action='store_true')
  args = parser.parse_args()
  
 
  file_path=os.path.dirname(os.path.abspath(__file__))
  if args.social or social:
    mode='social'
    ori_data=loadmat(f'{file_path}/../../data/cdata_other')['cdata_other']
  else:
    mode='nonsocial'
    ori_data=loadmat(f'{file_path}/../../data/cdata_info')['cdata_info']
    
  result_dir=f'{file_path}/../../results/model_results'

  load_dir=f'{file_path}/../../data/precomputed_results'
  if args.precomputed:
    import shutil
    shutil.copytree(load_dir,result_dir,dirs_exist_ok=True)
  try:
    os.mkdir(result_dir)
  except:
    pass
  os.chdir(result_dir)


  try:
    os.mkdir(f'./{mode}')
  except:
    pass 
  os.chdir(f'./{mode}')  

  return args,mode,ori_data

if __name__=='__main__':
  
  import os
  ori_dir=os.getcwd()
  args,mode,ori_data=init()
  
  models=[Drift_rate_model,Drift_rate_no_mi_model,Drift_rate_no_urg_model,Drift_rate_minimal_model]
  models+=[Baseline_model,Baseline_no_mi_model,Baseline_no_urg_model,Baseline_minimal_model]
  # names=[model.__name__ for model in models]
  names=['Drift rate model + MI + Urg','Drift rate model + Urg','Drift rate model + MI','Drift rate model']
  names+=['Baseline model + MI + Urg','Baseline model + Urg','Baseline model + MI','Baseline model']
  

  idx=[i for i in range(len(models)) if args.model.find(f'{i+1}')>=0 ]
  models=[models[i] for i in idx]
  names=[names[i] for i in idx]
  for i,model in enumerate(models):
    if not args.eval:
      fit(model,ori_data,mode)
    if not args.precomputed:
      eval(model,ori_data,mode)
  



  os.chdir(ori_dir)
