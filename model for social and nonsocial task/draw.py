from fit import *
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd



def draw_all_sub(model_i):
  model_class=models[model_i]
  all_sub_fig=plt.figure(figsize=(35,40))
  figs=all_sub_fig.subfigures(int((2+ori_data.shape[0])/3),3).flatten()

  model=model_class(mode=mode)
  all_res=np.load(f'{model_class.__name__}_paras.npy',allow_pickle=True)
  dat=np.array([r.x for r in all_res])
  

  res=np.zeros((ori_data.shape[0],3))
  all_data=np.zeros((0,5))
  all_prd=np.zeros((0,3))
  
  for sub in range(dat.shape[0]) :
    
    data=model.get_sub_data(ori_data,sub)


    all_data=np.concatenate((all_data,data),0)
    paras=dat[sub]
    model.init_paras(paras)
    
    
    prd=model.compare(paras,data,repn=10)
    all_prd=np.concatenate((all_prd,prd),0)

    dv=np.around(data[:,4]-data[:,3],2)
    dvs=np.unique(dv)
    sv=np.around(data[:,4]+data[:,3],2)
    svs=np.unique(sv)
    l=['Change','Reaction time (ms)','Confidence']
    

    figs[sub].subplots(1,3)
    f=0
    for p in [0,2,1]:
      f+=1
      ax=figs[sub].axes[f-1]
      ave_sub=np.zeros_like(dvs)
      ave_mod=np.zeros_like(dvs)
      for i in range((dvs.shape[0])):
        idx=dv==dvs[i]
        ave_sub[i]=np.mean(data[idx,p])
        ave_mod[i]=np.mean(prd[idx,p])
        
      if p==0:
        ave_mod=ave_mod-1
        
        
      ax.scatter(dvs,ave_sub,label='Data')
      ax.plot(dvs,ave_mod,label='Model')
      if sub>=ori_data.shape[0]-3:
        ax.set_xlabel('Confidence difference')
      ax.spines[['top','right']].set_visible(False)
      ax.set_ylabel(l[p])
      
    figs[sub].axes[0].text(-0.4,0.5,f'Sub {sub+1}',transform=figs[sub].axes[0].transAxes)
    figs[sub].subplots_adjust(left=0.1)
    
    pass

  all_sub_fig.savefig(f'all_sub_{names[model_i]}.svg')
  plt.close()


  data=np.array(all_data)
  prd=np.array(all_prd)
  dv=np.around(data[:,4]-data[:,3],2)
  dvs=np.unique(dv)
  

  h,e=np.histogram(dv,15)
  el,er=e[0],e[-1]
  j=0
  while h[j]<30:
    j+=1
  e=e[j+1:]
  h=h[j+1:]

  j=len(h)-1
  while h[j]<30:
    j-=1
  e=e[:j+1]

  e=np.insert(e,0,el)  
  e=np.append(e,er)  
      

  digitized = np.digitize(dv, e)
    

  dvs=(e[1:]+e[:-1])/2
  for i in range(1, len(e)):
    dv[digitized == i]=dvs[i-1]

  sv=np.around(data[:,3]+data[:,4],2)
  svs=np.unique(sv)
  l=['Change','Reaction time (ms)','Confidence']
  plt.figure(figsize=(12,4))
  
  f=0
  for p in [0,2,1]:
    f+=1
    ave_sub=np.zeros_like(dvs)
    ave_mod=np.zeros_like(dvs)
    bar_sub=np.zeros_like(dvs)
    bar_mod=np.zeros_like(dvs)
    for i in range((dvs.shape[0])):
      idx=dv==dvs[i]
      
      ave_sub[i]=np.mean(data[idx,p])
      ave_mod[i]=np.mean(prd[idx,p])
      bar_sub[i]=np.std(data[idx,p])/np.sqrt(np.sum(idx))
      bar_mod[i]=np.std(prd[idx,p])/np.sqrt(np.sum(idx))
      
    if p==0:
      ave_mod=ave_mod-1


    idx=np.array([np.sum(dv==dvs[i])>0 for i in range(dvs.shape[0])])
    ave_sub=ave_sub[idx]
    ave_mod=ave_mod[idx]
    bar_sub=bar_sub[idx]
    bar_mod=bar_mod[idx]
    dvs=dvs[idx]
    
    plt.subplot(1,3,f)
    from matplotlib.ticker import MaxNLocator
    plt.gca().yaxis.set_major_locator(MaxNLocator(5)) 
    plt.errorbar(dvs,ave_sub,yerr=bar_sub,fmt='o',label='Data')
    line=plt.plot(dvs,ave_mod,label='Model')
    
    plt.xlabel('Confidence difference')
    plt.ylabel(l[p])
    m=plt.xlim()[1]
    m=np.ceil(m*10)/10+0.1
    plt.xlim(-m,m)
    plt.gca().spines[['top','right']].set_visible(False)

  

  plt.savefig(f'all_sub_ave_{names[model_i]}.svg')
  plt.show(block=False)


def draw_paras(model_i,para_names):
  model_class=models[model_i]
  model=model_class(mode=mode)
  all_res=np.load(f'{model_class.__name__}_paras.npy',allow_pickle=True)
  dat=np.array([r.x for r in all_res])
  dat[:,1:3]/=model.b0
  
  fig=plt.figure(figsize=(12,4))
  fig.subplots(1,dat.shape[1])
  import seaborn as sns
  import pandas as pd
  for j in range(dat.shape[1]):
      
    sns.violinplot(data=pd.DataFrame(dat[:,j]),ax=fig.axes[j],inner='points',color='#94C5FD')
    fig.axes[j].set_title(para_names[j])
  fig.suptitle(names[model_i])

  fig.savefig(f'./figs/{names[model_i]}_paras.svg')
  pass
  

  
def get_sub_beta(ori_data):
  conf_beta_list=[[]]
  choice_beta_list=[[]]
  acc_beta_list=[[]]
  isc_beta_list=[[]]
  model_class=models[0]
  
  model=model_class(mode=mode)
    
  j=0
  for sub in range(ori_data.shape[0]) :
    data,isc=model.get_sub_data(ori_data,sub,extended=True)



    import statsmodels.api as sm
    import pandas as pd
    
    sv=np.around(log_odds(data[:,3])+log_odds(data[:,4]),2)
    dv=np.around(log_odds(data[:,3])-log_odds(data[:,4]),2)
    signed_dv=np.around(log_odds(data[:,3])-log_odds(data[:,4]),2)
    signed_dv[data[:,0]+1==2]*=-1
    ans=(data[:,4]>data[:,3]).astype(int)

    x = np.stack((signed_dv, sv), -1)
    x = pd.DataFrame(x, columns=['delta V','sum V'])
    x = sm.add_constant(x)
    model_confidence = sm.GLM(log_odds(data[:,2]), x)
    results_confidence = model_confidence.fit()
    


    x = np.stack((dv, sv), -1)
    x = pd.DataFrame(x, columns=['delta V','sum V'])
    x = sm.add_constant(x)
    model_choice = sm.GLM(1-data[:,0], x,family=sm.families.Binomial())
    results_choice= model_choice.fit()
    
    


    x = np.stack((np.abs(dv), sv), -1)
    x = pd.DataFrame(x, columns=['delta V','sum V'])
    x = sm.add_constant(x)
    model_acc = sm.GLM(1-np.abs(data[:,0]-ans), x,family=sm.families.Binomial())
    results_acc = model_acc.fit()

    x = np.stack((np.abs(dv), sv), -1)
    x = pd.DataFrame(x, columns=['delta V','sum V'])
    x = sm.add_constant(x)
    model_isc = sm.GLM(isc, x,family=sm.families.Binomial())
    results_isc = model_isc.fit()
    
    conf_beta_list[j].append(results_confidence.params)
    choice_beta_list[j].append(results_choice.params)
    acc_beta_list[j].append(results_acc.params)
    isc_beta_list[j].append(results_isc.params)
    
  np.save('sub_conf_beta_list.npy',conf_beta_list)
  np.save('sub_choice_beta_list.npy',choice_beta_list)
  np.save('sub_acc_beta_list.npy',acc_beta_list)
  np.save('sub_isc_beta_list.npy',isc_beta_list)

def draw_sub_beta():
  
  conf_beta_list=np.load('sub_conf_beta_list.npy',allow_pickle=True)
  choice_beta_list=np.load('sub_choice_beta_list.npy',allow_pickle=True)
  acc_beta_list=np.load('sub_acc_beta_list.npy',allow_pickle=True)
  isc_beta_list=np.load('sub_isc_beta_list.npy',allow_pickle=True)
  ls=[choice_beta_list,conf_beta_list,acc_beta_list,isc_beta_list]
  j=0
  dat=[np.array(l[j])[:,-1].tolist() for l in ls]
  labs=['Choice','Confidence','Accuracy\n(stay or change)','Accuracy\n(left or right)']
  
  sns.violinplot()
  fig=plt.figure(figsize=(12,4))
  fig.subplots(1,4)
  for i in range(len(labs)):
    sns.violinplot(data=pd.DataFrame(dat[i]),ax=fig.axes[i],inner='points',color='#94C5FD')
    fig.axes[i].set_title(labs[i])
  fig.savefig('./figs/sub_beta.svg')

def draw_pie():
  res={}
  for i in range(len(models)):
    name=f'{models[i].__name__}_eval.npy'
    res[f'{models[i].__name__}']=np.load(name,allow_pickle=True)

  score_data={model:res[model].item()['score'] for model in res.keys()} 

  import pandas as pd
  data=[]
  
  for i in range(len(models)):
    model=models[i].__name__
    res=score_data[model]
    for sub in range(ori_data.shape[0]):
      data.append({'sub':sub,'model':model,'score':res[sub,0],'AIC':res[sub,1],'BIC':res[sub,2]})
      
  data=pd.DataFrame(data)



  cnt=np.zeros((len(models),2))
  for sub in range(ori_data.shape[0]):
    l=[]
    for i in range(len(models)):
      model=models[i].__name__
      l.append(score_data[model][sub,1])
      
    best=np.argmin(l)
    cnt[best,0]+=1
    l=[]
    for i in range(len(models)):
      model=models[i].__name__
      l.append(score_data[model][sub,2])
    best=np.argmin(l)
    cnt[best,1]+=1
    

  def f(s):
    if s==0:
      return ''
    return f'{s:.1f}%'
  plt.figure(figsize=(6,3))
  _,txt,_=plt.pie(cnt[:,1],labels=names,autopct=f,radius=1.1,textprops={'color':'white'},colors=plt.cm.tab20b(range(len(models))))
  for t in txt:
    t.set_visible(False)
  plt.title('Best model',y=0.93)
  plt.legend(frameon=False,bbox_to_anchor=(0.95, 0.9),handlelength=0.8)
  plt.subplots_adjust(hspace=0)
  plt.savefig('./figs/best_model_pie.svg', bbox_inches='tight', pad_inches=0)
  plt.show(block=False)
  
def get_beta():
  conf_beta_list=[[] for i in range(len(models))]
  choice_beta_list=[[] for i in range(len(models))]
  acc_beta_list=[[] for i in range(len(models))]
  for j,model_class in enumerate(models):
    model=model_class(mode=mode)
    all_res=np.load(f'{model_class.__name__}_paras.npy',allow_pickle=True)
    dat=np.array([r.x for r in all_res])
    

    for sub in range(ori_data.shape[0]) :
      data=model.get_sub_data(ori_data,sub)

      paras=dat[sub]
      
      model.init_paras(paras)

      prd=model.compare(paras,data,repn=10)
      

      dv=np.around(data[:,3]-data[:,4],2)
      sv=np.around(data[:,3]+data[:,4],2)
        
      import statsmodels.api as sm
      import pandas as pd
      
      signed_dv=np.around(data[:,3]-data[:,4],2)
      signed_dv[data[:,0]+1==2]*=-1
      ans=1+(data[:,4]>data[:,3]).astype(int)
      try:
        x = np.stack((signed_dv, sv), -1)
        x = pd.DataFrame(x, columns=['delta V','sum V'])
        x = sm.add_constant(x)
        model_confidence = sm.GLM(prd[:,2], x,family=sm.families.Binomial())
        results_confidence = model_confidence.fit()
        
        conf_beta_list[j].append(results_confidence.params)


        x = np.stack((dv, sv), -1)
        x = pd.DataFrame(x, columns=['delta V','sum V'])
        x = sm.add_constant(x)
        model_choice = sm.GLM(2-prd[:,0], x,family=sm.families.Binomial())
        results_choice= model_choice.fit()
        
        choice_beta_list[j].append(results_choice.params)


        x = np.stack((np.abs(dv), sv), -1)
        x = pd.DataFrame(x, columns=['delta V','sum V'])
        x = sm.add_constant(x)
        model_acc = sm.GLM(1-np.abs(prd[:,0]-ans), x,family=sm.families.Binomial())
        results_acc = model_acc.fit()
        
        acc_beta_list[j].append(results_acc.params)
      except:
        pass
    
  np.save('conf_beta_list.npy',conf_beta_list)
  np.save('choice_beta_list.npy',choice_beta_list)
  np.save('acc_beta_list.npy',acc_beta_list)
def draw_beta():
  

  conf_beta_list=np.load('conf_beta_list.npy',allow_pickle=True)
  choice_beta_list=np.load('choice_beta_list.npy',allow_pickle=True)
  acc_beta_list=np.load('acc_beta_list.npy',allow_pickle=True)
  conf_beta_fig=plt.figure(figsize=(10,10))
  conf_beta_fig.subplots(2,4)
  choice_beta_fig=plt.figure(figsize=(10,10))
  choice_beta_fig.subplots(2,4)
  acc_beta_fig=plt.figure(figsize=(10,10))
  acc_beta_fig.subplots(2,4)

  for l,fig in [(conf_beta_list,conf_beta_fig),(choice_beta_list,choice_beta_fig),(acc_beta_list,acc_beta_fig)]:
    for j,model_class in enumerate(models):
      dat=np.array(l[j])[:,-1]

      from scipy.stats import ttest_1samp
      if np.mean(dat)>0:
        t,p=ttest_1samp(dat,0,alternative='greater')
      else:
        t,p=ttest_1samp(dat,0,alternative='less')
      # t,p=ttest_1samp(dat,0)

      if p>=0.05:
        sig='p={:.3f} ns'.format(p)
        # sig='ns'.format(p)
      elif p>=0.01:
        sig='p={:.3f} *'.format(p)
        # sig='*'.format(p)
      elif p>=0.001:
        sig='p={:.3f} **'.format(p)
        # sig='**'.format(p)
      else:
        sig='p<0.001 ***'
        # sig='***'
      
      
      sns.violinplot(data=pd.DataFrame(dat),ax=fig.axes[j],inner='points',color='#94C5FD')
      ax=fig.axes[j]
      
      if np.mean(dat)>0:
        ax.text(0.2,0.8,f'{sig}',ha='center',transform=ax.transAxes)
      else:
        ax.text(0.2,0.2,f'{sig}',ha='center',transform=ax.transAxes)
      # ax.set_ylim(top=ax.get_ylim()[1]*1.2)
      # ax.text(0.5,0.8,f'{sig}',ha='center',transform=ax.transAxes)
      
      ax.set_xticks([])
      ax.set_ylabel('Beta')
      ax.spines[['top','right']].set_visible(False)
      ax.set_title(names[j])
      ax.axhline(0,linestyle='dashed',color='black')

  conf_beta_fig.suptitle('Conf beta')
  choice_beta_fig.suptitle('Choice beta')
  acc_beta_fig.suptitle('Accuracy beta')

  conf_beta_fig.savefig('./figs/conf_beta.svg')
  choice_beta_fig.savefig('./figs/choice_beta.svg')
  acc_beta_fig.savefig('./figs/acc_beta.svg')
  plt.show(block=False)

  pass


def J(df):
  from scipy.special import gamma
  return gamma(df/2)/np.sqrt(df/2)/gamma((df-1)/2)

def print_beta(output_file):
  
  def cal_res(l):
    q1=np.quantile(l,0.25)
    q3=np.quantile(l,0.75)
    ext=1.5*(q3-q1)
    a=np.array(l)
    idx=(a<q3+ext)&(a>q1-ext)
    a=a[idx]
    res=ttest_1samp(a,0)

    d=np.mean(a)/np.std(a,ddof=1)*J(res.df)
    return res,d
  
  conf_beta_list=np.load('sub_conf_beta_list.npy',allow_pickle=True)
  choice_beta_list=np.load('sub_choice_beta_list.npy',allow_pickle=True)
  acc_beta_list=np.load('sub_acc_beta_list.npy',allow_pickle=True)
  isc_beta_list=np.load('sub_isc_beta_list.npy',allow_pickle=True)

  ls=[choice_beta_list,conf_beta_list,acc_beta_list,isc_beta_list]
  labs=['Choice','Confidence','Accuracy (stay or change)','Accuracy (left or right)']
  j=0
  dat=[np.array(l[j])[:,-1].tolist() for l in ls]

  from scipy.stats import ttest_1samp
  for i,l in enumerate(dat):
      res,d=cal_res(l)
      t,p,df=res.statistic,res.pvalue,res.df
      # print(f'Data {labs[i]}:t({df})={t:.2f},p={p:.2e}',file=output_file)

      if p>=1e-3:
        
        print(f'Data {labs[i]}: t({df}) = {t:.2f}, p = {p:.3f}, d = {d:.2f};',file=output_file)
      else:  
        print(f'Data {labs[i]}: t({df}) = {t:.2f}, p = {p:.2e}, d = {d:.2f};',file=output_file)


  conf_beta_list=np.load('conf_beta_list.npy',allow_pickle=True)
  choice_beta_list=np.load('choice_beta_list.npy',allow_pickle=True)
  acc_beta_list=np.load('acc_beta_list.npy',allow_pickle=True)
  
  ls=[choice_beta_list,conf_beta_list,acc_beta_list]
  labs=['Choice','Confidence','Accuracy']
  for j,model in enumerate(models):
    dat=[np.array(l[j])[:,-1].tolist() for l in ls]

    
    from scipy.stats import ttest_1samp
    for i,l in enumerate(dat):
      res,d=cal_res(l)
      t,p,df=res.statistic,res.pvalue,res.df
      if p>=1e-3:
        
        print(f'{names[j]} {labs[i]}: t({df}) = {t:.2f}, p = {p:.3f}, d = {d:.2f};',file=output_file)
      else:  
        print(f'{names[j]} {labs[i]}: t({df}) = {t:.2f}, p = {p:.2e}, d = {d:.2f};',file=output_file)

  
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from fit import *



def draw_cor_no_sub(model_i,paras_datas,model_datas,sub_datas,labs):
  
  plt.rcParams['font.size']=17
  n=len(paras_datas)
  m=len(para_names)+1
  fig=plt.figure(figsize=(m*5,5*n))
  fig.subplots(n,m,sharey=True)  
  for j in range(n):
    paras_data=paras_datas[j]
    model_data=model_datas[j]
    sub_data=sub_datas[j]
    lab=labs[j]
    all_res=np.load(paras_data,allow_pickle=True)
    dat=np.array([r.x for r in all_res])
    dat[:,1:3]/=300

    conf_beta_list=np.load(model_data,allow_pickle=True)
    model_beta=np.array(conf_beta_list[model_i])[:,-1]
    sub_conf_beta_list=np.load(sub_data,allow_pickle=True)
    beta=np.array(sub_conf_beta_list[model_i])[:,-1]

    from scipy.stats import linregress
    import seaborn as sns
    ax=fig.axes[j*m]
    if j>0:
      ax.sharex(fig.axes[0])
    res=linregress(beta,model_beta)
    # print(linregress(beta,model_beta))
    sns.regplot(x=beta,y=model_beta,ax=ax)
    if j==0:
      ax.set_title(f'Subject-Model',fontdict={'size':20},pad=20)
      
    if j==n-1:
      ax.set_xlabel('Subject confirmation bias')
    ax.set_ylabel('Model confirmation bias')
    ax.set_ylim(top=3)
    ax.text(0.5,0.8,f'R={res.rvalue:.3f}\np={res.pvalue:.3f}',ha='center',transform=ax.transAxes)

    ax.set_box_aspect(1)
    ax.text(-0.3,0.5,lab,va='center',ha='center',rotation=90,fontsize=20,transform=ax.transAxes)
    ax.spines[['top','right']].set_visible(False)

    
    print(f'Subject-Model:',file=output_file)
    print(f'r({beta.shape[0]-2})={res.rvalue:.3f},p={res.pvalue:.3f}',file=output_file)
    for i in range(m-1):
      ax=fig.axes[j*m+1+i]

      if j>0:
        ax.sharex(fig.axes[1+i])
      res=linregress(dat[:,i],model_beta.reshape(-1,1).repeat(dat.shape[1],1)[:,i])
      # print(para_names[i],linregress(dat[:,i],model_beta.reshape(-1,1).repeat(dat.shape[1],1)[:,i]))
      sns.regplot(x=dat[:,i],y=model_beta.reshape(-1,1).repeat(dat.shape[1],1)[:,i],ax=ax)
      if j==0:
        ax.set_title(f'{para_names[i]}',fontdict={'size':20},pad=20)
      ax.set_ylim(top=3)
      ax.text(0.5,0.8,f'R={res.rvalue:.3f}\np={res.pvalue:.3f}',ha='center',transform=ax.transAxes)
      if j==n-1:
        ax.set_xlabel('Value')
      # ax.set_ylabel('Model confirmation bias')

      ax.set_box_aspect(1)
      ax.spines[['top','right']].set_visible(False)
      print(f'{para_names[i]}:',file=output_file)
      print(f'r({model_beta.shape[0]-2})={res.rvalue:.3f},p={res.pvalue:.3f}',file=output_file)
  print('',file=output_file)
  fig.subplots_adjust(wspace=0.05,hspace=0.15)
  plt.savefig(f'{names[model_i]}_social_nonsocial_cor.svg')
  plt.show(block=False)

  pass



if __name__=='__main__':
  import os
  ori_dir=os.getcwd()
  args,mode,ori_data=init()
  
  try:
    os.mkdir('./figs')
  except:
    pass
  
  models=[Drift_rate_model,Drift_rate_no_mi_model,Drift_rate_no_urg_model,Drift_rate_minimal_model]
  models+=[Baseline_model,Baseline_no_mi_model,Baseline_no_urg_model,Baseline_minimal_model]
  
  
  names=['Drift rate model + MI + Urg','Drift rate model + Urg','Drift rate model + MI','Drift rate model']
  names+=['Baseline model + MI + Urg','Baseline model + Urg','Baseline model + MI','Baseline model']

  idx=[i for i in range(len(models)) if args.model.find(f'{i+1}')>=0 ]
  models=[models[i] for i in idx]
  names=[names[i] for i in idx]

  for i in range (len(models)):
    labs=['v0','b1/b0','b2/b0','k','sigma']
    if names[i].find('MI')>=0:
      labs.append('lambda')
    if names[i].find('Urg')>=0:
      labs.append('tau')
      labs.append('t0')
    draw_paras(i,labs)

  #if not args.eval:
  draw_all_sub(0)
  get_sub_beta()
  get_beta()

  draw_sub_beta()
  draw_pie()
  draw_beta()


  with open(ori_dir+'/beta.txt','w') as f:
    print_beta(f)

  try:

    os.chdir('..')
      
    names=['Drift rate model + MI + Urg','Drift rate model + Urg','Drift rate model + MI','Drift rate model']
    names+=['Baseline model + MI + Urg','Baseline model + Urg','Baseline model + MI','Baseline model']
    para_names=['v0','b1/b0','b2/b0','k','sigma','lambda','tau','t0']
    paras_datas=['./nonsocial/Drift_rate_model_paras.npy','./social/Drift_rate_model_paras.npy']
    model_datas=['./nonsocial/conf_beta_list.npy','./social/conf_beta_list.npy']
    sub_datas=['./nonsocial/sub_conf_beta_list.npy','./social/sub_conf_beta_list.npy']
    labs=['Non-social','Social']
    
    with open('./social_nonsocial_cor_output.txt','w') as output_file:
      draw_cor_no_sub(0,paras_datas,model_datas,sub_datas,labs)
  except:
    pass
  os.chdir(ori_dir)
