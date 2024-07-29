from fit import *
import numpy as np
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')
def get_data_prior(model_class):
  model=model_class(mode=mode)
  all_res=np.load(f'{model_class.__name__}_paras.npy',allow_pickle=True)
  dat=np.array([r.x for r in all_res])
  
  datas=[]
  for sub in range(ori_data.shape[0]) :    
    data,isc,dir,cuedir=model.get_sub_data(ori_data,sub,extended=True)


    paras=dat[sub]
    model.init_paras(paras)

    prd=model.compare(paras,data,repn=1)
    data[:,0]-=1
    prd[:,0]-=1

    no_cue=np.sum(data[:,-2:],1)==0
    conflict_cue=(~no_cue)&(dir!=cuedir)
    consistent_cue=(~no_cue)&(dir==cuedir)

    data[:,3:5]*=2/np.max((data[:,4]-data[:,3]))
    
    dv=np.around(data[:,4]-data[:,3],2)
    dvs=np.unique(dv)
    
    data_logodds=log_odds(data[:,2])
    model_logodds=log_odds(prd[:,2])
    data_base_logodds=[]
    model_base_logodds=[]

    
    for v in dvs:
      idx=(dv==v)
      data_base_logodds.append(log_odds(np.mean(data[:,2][idx&no_cue])))
      model_base_logodds.append(log_odds(np.mean(prd[:,2][idx&no_cue])))
      data_logodds[idx]-=data_base_logodds[-1]
      model_logodds[idx]-=model_base_logodds[-1]
      condition_sub=(conflict_cue)&(cuedir==data[:,0])
      condition_model=(conflict_cue)&(cuedir==prd[:,0])
      data_logodds[idx&condition_sub]+=2*data_base_logodds[-1]
      model_logodds[idx&condition_model]+=2*model_base_logodds[-1]
      
    data=np.concatenate((data_logodds.reshape(-1,1),data),1)
    prd=np.concatenate((model_logodds.reshape(-1,1),prd),1)
    datas.append({'data':data,'prd':prd,'cuedir':cuedir,'dir':dir})
  return datas

def get_data_post(model_class):
  model=model_class(mode=mode)
  all_res=np.load(f'{model_class.__name__}_paras.npy',allow_pickle=True)
  dat=np.array([r.x for r in all_res])
  datas=[]
  for sub in range(dat.shape[0]) :
    
    data,ans,ans2,dir,cuedir,coh=model.get_sub_data(ori_data,sub,extended=True)
    coh*=2/np.max(coh)
    coh=coh*(dir*2.0-1)

    paras=dat[sub]
    model.init_paras(paras)
    
    prd=model.compare(paras,data,repn=1)
    data[:,0]-=1
    prd[:,0]-=1
    
    consistent_cue=(ans==cuedir)

    prd[:,0][consistent_cue]=0
    prd[:,2][consistent_cue]=1

    
    data_logodds=log_odds(data[:,2])
    model_logodds=log_odds(prd[:,2])
    data_logodds-=log_odds(data[:,3])
    model_logodds-=log_odds(data[:,3])
    condition_sub=(data[:,0]==1)
    condition_model=(prd[:,0]==1)
    data_logodds[condition_sub]+=2*log_odds(data[:,3][condition_sub])
    model_logodds[condition_model]+=2*log_odds(data[:,3][condition_model])

    data=np.concatenate((data_logodds.reshape(-1,1),data),1)
    prd=np.concatenate((model_logodds.reshape(-1,1),prd),1)
    datas.append({'data':data,'prd':prd,'cuedir':cuedir,'dir':dir,'ans':ans,'coh':coh})
  return datas



def draw(data,prd,dv,ax,condition_sub,condition_model,p=0,subonly=False,**kwargs):
  
  l=['Delta log odds','ratio of choosing right','reaction time (ms)','confidence']
  dvs=np.unique(dv)
  ave_sub=np.zeros_like(dvs)
  ave_mod=np.zeros_like(dvs)
  bar_sub=np.zeros_like(dvs)
  bar_mod=np.zeros_like(dvs)
  for i in range((dvs.shape[0])):
    idx=(dv==dvs[i])
    ave_sub[i]=np.mean(data[idx&condition_sub,p])
    ave_mod[i]=np.mean(prd[idx&condition_model,p])
        
    bar_sub[i]=np.std(data[idx&condition_sub,p])/np.sqrt(np.sum(idx&condition_sub))
    bar_mod[i]=np.std(prd[idx&condition_model,p])/np.sqrt(np.sum(idx&condition_sub))
        
  ax.set_xticks([-2,-1,-0.5,0,0.5,1,2],[-2,-1,'',0,'',1,2])
  if p==0:
    ax.axhline(log_odds(0.7),color='black')
    ax.axhline(0,color='black',linestyle='dashed')

  if subonly:
    lines=ax.plot(dvs,ave_sub,linestyle='dashed',**kwargs)
  else:
    lines=ax.plot(dvs,ave_mod,**kwargs)
  ax.errorbar(dvs,ave_sub,yerr=bar_sub,c=lines[0].get_color(),fmt='o')

  ax.spines[['top','right']].set_visible(False)
  ax.set_ylabel(l[p],size=15)
  if p==1:
    ax.axhline(0.5,color='black',linestyle='dashed')
    def glm(x,y):
      import statsmodels.api as sm
      import pandas as pd
      x=pd.DataFrame(x,columns=['x'])
      x = sm.add_constant(x)
      glm=sm.GLM(y,x,family=sm.families.Binomial())
      res=glm.fit()
      return res.params.iloc[1],res.params.iloc[0]
    
    model_k,model_b=glm(dv[condition_model],prd[condition_model,1])
    sub_k,sub_b=glm(dv[condition_sub],data[condition_sub,1])


    return (sub_k,sub_b),(model_k,model_b)
def draw_fig(prior_data,post_data,choice_ax,prior_ax,post_ax,legend=False):
  

  cuedir=prior_data['cuedir']
  dir=prior_data['dir']
  data=prior_data['data']
  prd=prior_data['prd']
          
  no_cue=np.sum(data[:,-2:],1)==0
  conflict_cue=(~no_cue)&(dir!=cuedir)
  consistent_cue=(~no_cue)&(dir==cuedir)


  dv=np.around(data[:,5]-data[:,4],2)
  
  condition_sub=no_cue
  condition_model=no_cue
  subres0,modelres0=draw(data,prd,dv,choice_ax,condition_sub,condition_model,p=1,label='neutral',color='grey')
  condition_sub=(~no_cue)&(cuedir==0)
  condition_model=(~no_cue)&(cuedir==0)
  subres1,modelres1=draw(data,prd,dv,choice_ax,condition_sub,condition_model,p=1,label='left-prior',color='deepskyblue')
  condition_sub=(~no_cue)&(cuedir==1)
  condition_model=(~no_cue)&(cuedir==1)
  subres2,modelres2=draw(data,prd,dv,choice_ax,condition_sub,condition_model,p=1,label='right-prior',color='steelblue')

  condition_sub=consistent_cue
  condition_model=consistent_cue
  draw(data,prd,dv,prior_ax,condition_sub,condition_model,label='congruent')
  condition_sub=conflict_cue
  condition_model=conflict_cue
  draw(data,prd,dv,prior_ax,condition_sub,condition_model,label='incongruent')
  condition_sub=conflict_cue&(data[:,1]==cuedir)
  condition_model=conflict_cue&(prd[:,1]==cuedir)
  draw(data,prd,dv,prior_ax,condition_sub,condition_model,label='same-with-info')
  condition_sub=conflict_cue&(data[:,1]!=cuedir)
  condition_model=conflict_cue&(prd[:,1]!=cuedir)
  draw(data,prd,dv,prior_ax,condition_sub,condition_model,label='diff-with-info')


  cuedir=post_data['cuedir']
  dir=post_data['dir']
  data=post_data['data']
  prd=post_data['prd']
  ans=post_data['ans']
  coh=post_data['coh']
          
  dv=np.around(coh,2)
  data_change=data[:,1]==1
  model_change=prd[:,1]==1
  data[:,1]=data[:,1].astype(bool)^ans
  prd[:,1]=prd[:,1].astype(bool)^ans



  conflict_cue=(ans!=cuedir)
  condition_sub=cuedir==0
  condition_model=cuedir==0
  subres3,modelres3=draw(data,prd,dv,choice_ax,condition_sub,condition_model,p=1,label='left-post',color='orangered')
  condition_sub=cuedir==1
  condition_model=cuedir==1
  subres4,modelres4=draw(data,prd,dv,choice_ax,condition_sub,condition_model,p=1,label='right-post',color='maroon')

  consistent_cue=(dir==cuedir)
  condition_sub=consistent_cue
  condition_model=consistent_cue
  draw(data,prd,dv,post_ax,condition_sub,condition_model,label='congruent',subonly=True)
  condition_sub=~consistent_cue
  condition_model=~consistent_cue
  draw(data,prd,dv,post_ax,condition_sub,condition_model,label='incongruent',subonly=True)
  condition_sub=conflict_cue&(data_change)
  condition_model=conflict_cue&(model_change)
  draw(data,prd,dv,post_ax,condition_sub,condition_model,label='change')
  condition_sub=conflict_cue&(~data_change)
  condition_model=conflict_cue&(~model_change)
  draw(data,prd,dv,post_ax,condition_sub,condition_model,label='stay')

  if legend:
    choice_ax.legend(frameon=False,handlelength=0.8,handletextpad=0.4,columnspacing=0.6)
    prior_ax.legend(frameon=False,ncol=2,loc=9,handlelength=0.8,handletextpad=0.4,columnspacing=0.6)
    post_ax.legend(frameon=False,ncol=2,loc=9,handlelength=0.8,handletextpad=0.4,columnspacing=0.6)
  prior_ax.set_title('prior',size=17)
  post_ax.set_title('post',size=17)
  choice_ax.set_xlabel('coherence',size=15)
  prior_ax.set_xlabel('coherence',size=15)
  post_ax.set_xlabel('coherence',size=15)

  subres=[subres0,subres1,subres2,subres3,subres4]
  modelres=[modelres0,modelres1,modelres2,modelres3,modelres4]
  return subres,modelres

  
def significance(p):
  if p>=0.05:
      sig='n.s.'
  elif p>=0.01:
      sig='*'
  elif p>=0.001:
      sig='**'
  else:
      sig='***'
  return sig
def add_sig(ax,x1,x2,p,h=0.8):
  sig=significance(p)
  ax.hlines(h,x1,x2,transform=ax.get_xaxis_transform(),color='black')
  if sig=='n.s.':
    ax.text((x1+x2)/2,h+0.04,sig,size=12,transform=ax.get_xaxis_transform(),ha='center',va='center')
  else:
    ax.text((x1+x2)/2,h+0.02,sig,size=12,transform=ax.get_xaxis_transform(),ha='center',va='center')

def J(df):
  from scipy.special import gamma
  return gamma(df/2)/np.sqrt(df/2)/gamma((df-1)/2)

def draw_bar(ax,data,datalabs=[],xlabs=[],ylab='',legendlabs=[],legend=False,output_pref=''):
  colors=['dodgerblue','indianred']

  l=[]
  ncols=len(colors)
  x=[i+int(i/ncols) for i in range(data.shape[1])]
  for j in range(data.shape[0]):
    for i in range(0,data.shape[1],ncols):
      ax.plot(x[i:i+ncols],data[j,i:i+ncols],color='grey',marker='o',markersize=3)
      

  from scipy.stats import ttest_rel
  from scipy.stats import ttest_1samp
  for i in range(data.shape[1]):
    try:
      lab=datalabs[i]
    except:
      lab=''
    bar=ax.bar(x[i],np.mean(data[:,i]),width=1,label='',color=colors[i%ncols])
    l.append(bar)
    if i%2==0:
      res=ttest_rel(data[:,i],data[:,i+1])
      t,p,df=res.statistic,res.pvalue,res.df
      add_sig(ax,x[i],x[i+1],p)
      # print('',file=output_file)

      a=data[:,i]-data[:,i+1]
      d=np.mean(a)/np.std(a,ddof=1)*J(res.df)
      if p>=1e-3:
        paired_s=(f'paired t-test: t({df}) = {t:.2f}, p = {p:.3f}, d = {d:.2f};')
      else:
        paired_s=(f'paired t-test: t({df}) = {t:.2f}, p = {p:.2e}, d = {d:.2f};')


    res=ttest_1samp(data[:,i].flatten(),0)
    t,p,df=res.statistic,res.pvalue,res.df
    a=data[:,i].flatten()
    d=np.mean(a)/np.std(a,ddof=1)*J(res.df)

    add_sig(ax,x[i]-0.4,x[i]+0.4,p,h=0.7)
    

    labs2=['prior','post']
    if p>=1e-3:
      print(f'{output_pref}{xlabs[int(i/2)]} {labs2[i%2]}: t({df}) = {t:.2f}, p = {p:.3f}, d = {d:.2f};',file=output_file)
    else:
      print(f'{output_pref}{xlabs[int(i/2)]} {labs2[i%2]}: t({df}) = {t:.2f}, p = {p:.2e}, d = {d:.2f};',file=output_file)

    
    if i%2==1:
      print(paired_s,file=output_file)
  try:
    ax.set_xticks([0.5+3*i for i in range(int(data.shape[1]/2))],xlabs)
  except:
    pass
  ax.axhline(0,color='black',linewidth=1)
  l=l[:ncols]

  ax.set_ylabel(ylab,size=15)
  ax.spines[['top','right']].set_visible(False)
  
  if legend:
    
    try:
      ax.legend(handles=l,labels=legendlabs,frameon=False,handlelength=0.8,handletextpad=0.4,loc=(0.95, 0.9))
    except:
      pass


def draw_all_sub():
  
  all_sub_fig=plt.figure(figsize=(30,(6*int((2+len(prior_datas))/3))))
  figs=all_sub_fig.subfigures(int((2+len(prior_datas))/3),3).flatten()
  sub_fit_res=[]
  model_fit_res=[]
  from copy import deepcopy
  for sub in range(len(prior_datas)):
    prior_data=deepcopy(prior_datas[sub])
    post_data=deepcopy(post_datas[sub])
    fig=figs[sub]
    choice_ax=fig.add_subplot(1,2,1)
    prior_ax=fig.add_subplot(2,2,2)
    post_ax=fig.add_subplot(2,2,4)
    subres,modelres=draw_fig(prior_data,post_data,choice_ax,prior_ax,post_ax)
    sub_fit_res.append(subres)
    model_fit_res.append(modelres)
    prior_ax.set_xlabel('')
    prior_ax.set_xticklabels([])
    choice_ax.set_title(f'Subject {sub+1}')
    fig.subplots_adjust(left=0.1,right=0.9)
  all_sub_fig.savefig('prior_post_all_sub.svg')


  import copy 
  prior_data=copy.deepcopy(prior_datas[0])
  for k in prior_data.keys():
    for sub in range(1,prior_datas.shape[0]):
      dat=prior_datas[sub][k]
      prior_data[k]=np.concatenate((prior_data[k],dat),0)

  post_data=copy.deepcopy(post_datas[0])
  for k in post_data.keys():
    for sub in range(1,post_datas.shape[0]):
      dat=post_datas[sub][k]
      post_data[k]=np.concatenate((post_data[k],dat),0)

  fig=plt.figure(figsize=(15,6))
  all_figs=fig.subfigures(1,2,width_ratios=[4,5]).flatten()
  choice_ax=all_figs[0].add_subplot(1,1,1)
  choice_ax.set_box_aspect(1)

  right_figs=all_figs[1].subfigures(2,1).flatten()
  prior_ax=right_figs[1].add_subplot(1,2,1)
  post_ax=right_figs[1].add_subplot(1,2,2)
  
  subres,modelres=draw_fig(prior_data,post_data,choice_ax,prior_ax,post_ax,legend=True)

  sub_fit_res=np.array(sub_fit_res)
  model_fit_res=np.array(model_fit_res)
  
  figs=right_figs[0].subfigures(1,2).flatten()
  sub_bias_ax=figs[0].add_subplot(1,2,1)
  model_bias_ax=figs[0].add_subplot(1,2,2)
  sub_slope_ax=figs[1].add_subplot(1,2,1)
  model_slope_ax=figs[1].add_subplot(1,2,2)
  # sub_bias_ax=fig.add_subplot(2,6,3)
  # model_bias_ax=fig.add_subplot(2,6,4)
  # sub_slope_ax=fig.add_subplot(2,6,5)
  # model_slope_ax=fig.add_subplot(2,6,6)
  



  sub_fit_res[:,(1,3,2,4),0]-=sub_fit_res[:,:1,0]
  model_fit_res[:,(1,3,2,4),0]-=model_fit_res[:,:1,0]
  
  sub_fit_res[:,(1,3,2,4),1]-=sub_fit_res[:,:1,1]
  model_fit_res[:,(1,3,2,4),1]-=model_fit_res[:,:1,1]
  sub_fit_res[:,(1,3),1]*=-1
  model_fit_res[:,(1,3),1]*=-1
  print('Data bias: ',file=output_file)
  draw_bar(sub_bias_ax,sub_fit_res[:,(1,3,2,4),1],xlabs=['left','right'],ylab='Delta bias')
  print('Model bias: ',file=output_file)
  draw_bar(model_bias_ax,model_fit_res[:,(1,3,2,4),1],xlabs=['left','right'],legendlabs=['prior','post'],legend=True)
  print('Data slope: ',file=output_file)
  draw_bar(sub_slope_ax,sub_fit_res[:,(1,3,2,4),0],xlabs=['left','right'],ylab='Delta slope')
  print('Model slope: ',file=output_file)
  draw_bar(model_slope_ax,model_fit_res[:,(1,3,2,4),0],xlabs=['left','right'])

  sub_bias_ax.set_title('data')
  model_bias_ax.set_title('model')
  sub_slope_ax.set_title('data')
  model_slope_ax.set_title('model')

  sub_bias_ax.set_ylim(-0.5,5)
  model_bias_ax.set_ylim(-0.5,5)
  sub_slope_ax.set_ylim(-1,2)
  model_slope_ax.set_ylim(-1,2)
  model_bias_ax.set_yticklabels([])
  model_slope_ax.set_yticklabels([])
  

  prior_ax.set_ylim(-0.8,4)
  post_ax.set_ylim(-0.8,4)
  # post_ax.set_yticklabels([])
  # post_ax.set_ylabel('')
  
  figs[0].subplots_adjust(left=0,right=0.85)
  figs[1].subplots_adjust(left=0.15,right=1)
  right_figs[0].subplots_adjust(left=0,right=1,wspace=0.3)
  right_figs[1].subplots_adjust(left=0,right=1,wspace=0.3,bottom=0.2)
  fig.savefig('prior_post_all_sub_ave.svg')
  plt.show(block=False)
  pass




if __name__=='__main__':
 
  plt.rcParams['font.size']=12
  import os
  ori_dir=os.getcwd()

  args,mode,ori_data=init()
  # if not args.eval:
  prior_datas=get_data_prior(Baseline_model)
  np.save('prior_datas.npy',prior_datas)
  prior_datas=np.load('prior_datas.npy',allow_pickle=True)

  os.chdir(ori_dir)

  args,mode,ori_data=init(post=True,keepconsistent=True)
  # if not args.eval:
  post_datas=get_data_post(Drift_rate_model)
  np.save('post_datas.npy',post_datas)
  post_datas=np.load('post_datas.npy',allow_pickle=True)

  os.chdir('..')
  with open('./prior_post_stat_output.txt','w') as output_file:
    draw_all_sub()


  os.chdir(ori_dir)
