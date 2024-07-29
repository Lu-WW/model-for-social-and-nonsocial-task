from fit import *
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange,tqdm
import warnings
warnings.filterwarnings('ignore')
n=100


def sim(model,n=1000,high=1,low=0.5):
  # model.maxt=50000
  dat=np.zeros((n,5))
  idx=np.ones(n).astype(bool)
  for rep in range(n):
    v1=np.random.rand()*(high-low)+low
    v2=np.random.rand()*(high-low)+low
    # data=(v1,0,v2,0,0)
    data=(0,0,0,v1,v2)
    # sconf,conf,oconf,change,rt=data
    
    pd_choice,pd_rt,pd_conf=model.simulate(model.get_input(data))
    if pd_choice<0:
      idx[rep]=False
    dat[rep]=np.array([pd_choice,pd_rt,pd_conf,v1,v2])
  # print(np.sum(~idx))
  dat=dat[idx]
  dat[:,2]-=np.min(dat[:,2])
  dat[:,2]/=np.max(dat[:,2])
  return dat
def get_beta(model,rep=100):
    

  conf_res=[]
  choice_res=[]
  for r in range(rep):
    dat=sim(model)
    import statsmodels.api as sm
    import pandas as pd
    
    sv=(dat[:,-2]+dat[:,-1])
    dv=(dat[:,-2]-dat[:,-1])
    signed_dv=(dat[:,-2]-dat[:,-1])
    signed_dv[dat[:,0]==2]*=-1
    
    try:
      x = np.stack((signed_dv, sv), -1)
      x = pd.DataFrame(x, columns=['delta V','sum V'])
      x = sm.add_constant(x)
      model_confidence = sm.GLM(dat[:,2], x,family=sm.families.Binomial())
      results_confidence = model_confidence.fit()
      


      x = np.stack((dv, sv), -1)
      x = pd.DataFrame(x, columns=['delta V','sum V'])
      x = sm.add_constant(x)
      
      model_choice = sm.GLM(2-dat[:,0], x,family=sm.families.Binomial())
      results_choice= model_choice.fit()
      
      
      conf_res.append(results_confidence.params)
      choice_res.append(results_choice.params)
    except:
      conf_res.append([np.nan for i in range(3)])
      choice_res.append([np.nan for i in range(3)])
      break
      
    
  conf_res=np.mean(np.array(conf_res),0)
  choice_res=np.mean(np.array(choice_res),0)
  return conf_res,choice_res
def draw_twin(x,data1,data2,xlabel,label1,label2,l1=False,l2=False):
  color1='#00B0F0'
  color2='#D35D3A'
  x=np.array(x)
  fig=plt.figure(figsize=(4,3))
  ax1 = plt.subplot(1,1,1)
  plt.subplots_adjust(left=0.2,right=0.8,bottom=0.15)
  valid=np.abs(data1)<200
  plt.plot(x[valid],data1[valid],color=color1)
  plt.xlabel(xlabel)
  plt.ylabel(label1,color=color1)
  ax1.yaxis.set_label_coords(-0.15, 0.5)
  if l1:
    plt.ylim(-5,5)
  else:
    m=np.max(np.abs(plt.ylim()))*1.1
    plt.ylim(-m,m)
  plt.axhline(0,linestyle='dashed',color=color1)
  ax2 = ax1.twinx()
  valid=np.abs(data2)<200
  plt.plot(x[valid],data2[valid],color=color2)
  plt.ylabel(label2,color=color2)
  ax2.yaxis.set_label_coords(1.15, 0.5)
  if l2:
    plt.ylim(-5,5)
  else:
    m=np.max(np.abs(plt.ylim()))*1.1
    plt.ylim(-m,m)
  plt.axhline(0,linestyle='dashed',color=color2)
  ax1.spines[['top','right']].set_visible(False)
  ax1.spines['left'].set_color(color1)
  ax1.tick_params(axis='y',colors=color1)
  ax2.spines[['top','left']].set_visible(False)
  ax2.spines['right'].set_color(color2)
  ax2.tick_params(axis='y',colors=color2)
  # fig.tight_layout()
  return fig



def draw_v0(model_class,name,mode,eval=False):

  model=model_class(mode=mode)

  all_res=np.load(f'{model_class.__name__}_paras.npy',allow_pickle=True)
  dat=np.array([r.x for r in all_res])

  # paras=np.mean(dat,0)
  paras=np.median(dat,0)
  paras[1:3]=np.mean(paras[1:3])
  # v0
  r=np.linspace(0,1,n)
  model.init_paras(paras)
  conf_l=[]
  choice_l=[]
  x=[]
  raw=model.v0
  if eval:
    dic=np.load(f'{name}_v0.npy',allow_pickle=True).item()
    # x=r[r*2>=raw]
    x=r
    conf_l=dic['conf']
    choice_l=dic['choice']
  else:
    for _,i in enumerate(tqdm(r,desc=f'Draw {name} v0')):
    # for i in r:
      # if i*2<raw:
      #   continue
      model.v0=i
      x.append(model.v0)
      conf_res,choice_res=get_beta(model)
      conf_l.append(conf_res)
      choice_l.append(choice_res)
    conf_l=np.array(conf_l)
    choice_l=np.array(choice_l)
    np.save(f'{name}_v0.npy',{'conf':conf_l,'choice':choice_l})


  fig=draw_twin(x,choice_l[:,-2],conf_l[:,-1],r'$v_0$',r'Choice $\Delta V$',r'Confidence $\Sigma V$',l2=True)
  fig.suptitle(name)
  fig.axes[0].set_xlim(np.min(r)-1,np.max(r)+1)
  fig.savefig(f'./figs/{name}_v0.svg')
  plt.close()

  fig=draw_twin(x,choice_l[:,-1],conf_l[:,-2],r'$v_0$',r'Choice $\Sigma V$',r'Confidence $\Delta V$',l1=True)
  fig.suptitle(name)
  fig.axes[0].set_xlim(np.min(r)-1,np.max(r)+1)
  fig.savefig(f'./figs/{name}_v0_sup.svg')
  plt.close()

def draw_k(model_class,name,mode,eval=False):

  model=model_class(mode=mode)

  all_res=np.load(f'{model_class.__name__}_paras.npy',allow_pickle=True)
  dat=np.array([r.x for r in all_res])

  # paras=np.mean(dat,0)
  paras=np.median(dat,0)
  paras[1:3]=np.mean(paras[1:3])
  # k
  r=np.linspace(1,15,n)
  model.init_paras(paras)
  conf_l=[]
  choice_l=[]
  x=[]
  raw=model.k
  if eval:
    dic=np.load(f'{name}_k.npy',allow_pickle=True).item()
    x=r
    # x=r[r*2>=raw]
    conf_l=dic['conf']
    choice_l=dic['choice']
  else:
    for _,i in enumerate(tqdm(r,desc=f'Draw {name} k')):
    # for i in r:
      # if i*2<raw:
      #   continue
      model.k=i
      x.append(model.k)
      conf_res,choice_res=get_beta(model)
      conf_l.append(conf_res)
      choice_l.append(choice_res)
    conf_l=np.array(conf_l)
    choice_l=np.array(choice_l)
    np.save(f'{name}_k.npy',{'conf':conf_l,'choice':choice_l})


  fig=draw_twin(x,choice_l[:,-2],conf_l[:,-1],r'$k$',r'Choice $\Delta V$',r'Confidence $\Sigma V$',l2=True)
  fig.suptitle(name)
  fig.axes[0].set_xlim(np.min(r)-1,np.max(r)+1)
  fig.savefig(f'./figs/{name}_k.svg')
  plt.close()

  fig=draw_twin(x,choice_l[:,-1],conf_l[:,-2],r'$k$',r'Choice $\Sigma V$',r'Confidence $\Delta V$',l1=True)
  fig.suptitle(name)
  fig.axes[0].set_xlim(np.min(r)-1,np.max(r)+1)
  fig.savefig(f'./figs/{name}_k_sup.svg')
  plt.close()


def draw_sigma(model_class,name,mode,eval=False):

  model=model_class(mode=mode)

  all_res=np.load(f'{model_class.__name__}_paras.npy',allow_pickle=True)
  dat=np.array([r.x for r in all_res])

  # paras=np.mean(dat,0)
  paras=np.median(dat,0)
  paras[1:3]=np.mean(paras[1:3])
  # sigma
  r=np.linspace(4,30,n)
  model.init_paras(paras)
  conf_l=[]
  choice_l=[]
  x=[]
  raw=model.sigma
  if eval:
    dic=np.load(f'{name}_sigma.npy',allow_pickle=True).item()
    # x=r[r*2>=raw]
    x=r
    conf_l=dic['conf']
    choice_l=dic['choice']
  else:
    for _,i in enumerate(tqdm(r,desc=f'Draw {name} sigma')):
    # for i in r:
      # if i*2<raw:
      #   continue
      model.sigma=i
      x.append(model.sigma)
      conf_res,choice_res=get_beta(model)
      conf_l.append(conf_res)
      choice_l.append(choice_res)
    conf_l=np.array(conf_l)
    choice_l=np.array(choice_l)
    np.save(f'{name}_sigma.npy',{'conf':conf_l,'choice':choice_l})


  fig=draw_twin(x,choice_l[:,-2],conf_l[:,-1],r'$\sigma$',r'Choice $\Delta V$',r'Confidence $\Sigma V$',l2=True)
  fig.suptitle(name)
  fig.axes[0].set_xlim(np.min(r)-1,np.max(r)+1)
  fig.savefig(f'./figs/{name}_sigma.svg')
  plt.close()

  fig=draw_twin(x,choice_l[:,-1],conf_l[:,-2],r'$\sigma$',r'Choice $\Sigma V$',r'Confidence $\Delta V$',l1=True)
  fig.suptitle(name)
  fig.axes[0].set_xlim(np.min(r)-1,np.max(r)+1)
  fig.savefig(f'./figs/{name}_sigma_sup.svg')
  plt.close()

def draw_lambda(model_class,name,mode,eval=False):

  model=model_class(mode=mode)

  all_res=np.load(f'{model_class.__name__}_paras.npy',allow_pickle=True)
  dat=np.array([r.x for r in all_res])

  
  # paras=np.mean(dat,0)
  paras=np.median(dat,0)
  paras[1:3]=np.mean(paras[1:3])
  # lambda
  r=np.linspace(0,0.03,n)
  model.init_paras(paras)
  conf_l=[]
  choice_l=[]
  x=[]
  raw=model.lamb
  if raw!=0:
    if eval:
      dic=np.load(f'{name}_lamb.npy',allow_pickle=True).item()
      x=r
      conf_l=dic['conf']
      choice_l=dic['choice']
    else:
      for _,i in enumerate(tqdm(r,desc=f'Draw {name} lambda')):
      # for i in r:
        model.lamb=i
        x.append(model.lamb)
        conf_res,choice_res=get_beta(model)
        conf_l.append(conf_res)
        choice_l.append(choice_res)
      conf_l=np.array(conf_l)
      choice_l=np.array(choice_l)
      np.save(f'{name}_lamb.npy',{'conf':conf_l,'choice':choice_l})

    fig=draw_twin(x,choice_l[:,-2],conf_l[:,-1],r'$\lambda$',r'Choice $\Delta V$',r'Confidence $\Sigma V$',l2=True)
    fig.suptitle(name)
    fig.axes[0].set_xticks([0,0.01,0.02,0.03])
    fig.savefig(f'./figs/{name}_lamb.svg')
    plt.close()

    fig=draw_twin(x,choice_l[:,-1],conf_l[:,-2],r'$\lambda$',r'Choice $\Sigma V$',r'Confidence $\Delta V$',l1=True)
    fig.suptitle(name)
    fig.axes[0].set_xticks([0,0.01,0.02,0.03])
    fig.savefig(f'./figs/{name}_lamb_sup.svg')
    plt.close()
 
def draw_tau(model_class,name,mode,eval=False):

  model=model_class(mode=mode)

  all_res=np.load(f'{model_class.__name__}_paras.npy',allow_pickle=True)
  dat=np.array([r.x for r in all_res])

  
  # paras=np.mean(dat,0)
  paras=np.median(dat,0)
  paras[1:3]=np.mean(paras[1:3])
  # tau
  r=np.linspace(500,30000,n)
  model.init_paras(paras)
  conf_l=[]
  choice_l=[]
  x=[]
  raw=model.tau
  if raw!=1e9:
    if eval:
      dic=np.load(f'{name}_tau.npy',allow_pickle=True).item()
      x=r
      conf_l=dic['conf']
      choice_l=dic['choice']
    else:
      
      for _,i in enumerate(tqdm(r,desc=f'Draw {name} tau')):
      # for i in r:
        model.tau=i
        x.append(model.tau)
        conf_res,choice_res=get_beta(model)
        conf_l.append(conf_res)
        choice_l.append(choice_res)
      conf_l=np.array(conf_l)
      choice_l=np.array(choice_l)
      np.save(f'{name}_tau.npy',{'conf':conf_l,'choice':choice_l})
    
    conf_res,choice_res=draw_no_urg(model_class,name,mode,eval=eval)

    fig=draw_twin(x,choice_l[:,-2],conf_l[:,-1],r'$\tau$',r'Choice $\Delta V$',r'Confidence $\Sigma V$',l2=True)
    fig.suptitle(name)

    fig.axes[0].set_xlim(r[0],r[-1])
    ticks=fig.axes[0].get_xticks().tolist()
    # labs=fig.axes[0].get_xticklabels()
    xl=fig.axes[0].get_xlim()[1]
    while(ticks[-1]>xl):
      ticks.pop()
      # labs.pop()
    labs=[f'{int(i)}' for i in ticks]
    x_inf=ticks[-1]+ticks[-1]-ticks[-2]
    ticks.append(x_inf)
    labs.append('Inf')
    fig.axes[0].set_xticks(ticks,labs)
    fig.axes[0].scatter(x_inf,choice_res[-2],color='#00B0F0')
    fig.axes[1].scatter(x_inf,conf_res[-1],color='#D35D3A')
    d = 0.8 
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,linestyle='none', mec='black', mew=2, clip_on=False,in_layout=False)
    xm=(ticks[-1]+ticks[-2])/2
    y0,y1=fig.axes[0].get_ylim()
    m=1.1*np.maximum(np.abs(choice_res[-2]),np.abs(conf_res[-1]))
    if m>y1:
      y0=-m
      y1=m
    fig.axes[0].plot([xm*0.99,xm*1.01], [y0,y0], **kwargs)
    fig.axes[0].set_ylim(y0,y1)
    fig.axes[0].set_xlim(r[0]-2000,x_inf*1.1)
    fig.savefig(f'./figs/{name}_tau.svg')
    plt.close()

    fig=draw_twin(x,choice_l[:,-1],conf_l[:,-2],r'$\tau$',r'Choice $\Sigma V$',r'Confidence $\Delta V$',l1=True)
    fig.suptitle(name)

    fig.axes[0].set_xlim(r[0],r[-1])
    ticks=fig.axes[0].get_xticks().tolist()
    # labs=fig.axes[0].get_xticklabels()
    xl=fig.axes[0].get_xlim()[1]
    while(ticks[-1]>xl):
      ticks.pop()
      # labs.pop()
    labs=[f'{int(i)}' for i in ticks]
    x_inf=ticks[-1]+ticks[-1]-ticks[-2]
    ticks.append(x_inf)
    labs.append('Inf')
    fig.axes[0].set_xticks(ticks,labs)
    fig.axes[0].scatter(x_inf,choice_res[-1],color='#00B0F0')
    fig.axes[1].scatter(x_inf,conf_res[-2],color='#D35D3A')
    d = 0.8 
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,linestyle='none', mec='black', mew=2, clip_on=False,in_layout=False)
    xm=(ticks[-1]+ticks[-2])/2
    y0,y1=fig.axes[0].get_ylim()
    m=1.1*np.maximum(np.abs(choice_res[-1]),np.abs(conf_res[-2]))
    if m>y1:
      y0=-m
      y1=m
    fig.axes[0].plot([xm*0.99,xm*1.01], [y0,y0], **kwargs)
    fig.axes[0].set_ylim(y0,y1)
    fig.axes[0].set_xlim(r[0]-2000,x_inf*1.1)
    fig.savefig(f'./figs/{name}_tau_sup.svg')
    plt.close()

def draw_no_urg(model_class,name,mode,eval=False):
  model=model_class(mode=mode)

  all_res=np.load(f'{model_class.__name__}_paras.npy',allow_pickle=True)
  dat=np.array([r.x for r in all_res])

  
  # paras=np.mean(dat,0)
  paras=np.median(dat,0)
  paras[1:3]=np.mean(paras[1:3])
  model.init_paras(paras)
  if eval:
    dic=np.load(f'{name}_no_urg.npy',allow_pickle=True).item()
    conf_res=dic['conf']
    choice_res=dic['choice']
    
  else:
    model.tau=1e9
    conf_res,choice_res=get_beta(model)
    np.save(f'{name}_no_urg.npy',{'conf':conf_res,'choice':choice_res})

  return conf_res,choice_res
if __name__=='__main__':
 
  import os
  ori_dir=os.getcwd()
  args,mode,ori_data=init(social=True)
  
  models=[Drift_rate_model,Drift_rate_no_mi_model,Drift_rate_no_urg_model,Drift_rate_minimal_model]
  models+=[Baseline_model,Baseline_no_mi_model,Baseline_no_urg_model,Baseline_minimal_model]
  # names=[model.__name__ for model in models]
  
  names=['Drift rate model + MI + Urg','Drift rate model + Urg','Drift rate model + MI','Drift rate model']
  names+=['Baseline model + MI + Urg','Baseline model + Urg','Baseline model + MI','Baseline model']
  
  if not args.eval:
      
    import multiprocessing as mp
    pool = mp.Pool(8)
    results=[]
    for j,model_class in enumerate(models):
      results.append(pool.apply_async(draw_sigma, args=(model_class,names[j],mode)))
      results.append(pool.apply_async(draw_lambda, args=(model_class,names[j],mode)))
      results.append(pool.apply_async(draw_tau, args=(model_class,names[j],mode)))
      results.append(pool.apply_async(draw_k, args=(model_class,names[j],mode)))

    results = [p.get() for p in results]
      

  for j,model_class in enumerate(models):
    
    draw_sigma(model_class,names[j],mode,True)
    draw_lambda(model_class,names[j],mode,True)
    draw_tau(model_class,names[j],mode,True)
    draw_k(model_class,names[j],mode,True)

  import os
  os.chdir(ori_dir)

