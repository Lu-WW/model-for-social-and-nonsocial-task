
import subprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--social',action='store_true')
parser.add_argument('-n', '--nonsocial',action='store_true')
parser.add_argument('-i', '--start_id',default=0,type=int)
args = parser.parse_args()

import os
file_path=os.path.dirname(os.path.abspath(__file__))
def run_one(id):
  os.chdir(f'{file_path}')
  if not os.path.isdir(f'./{id}'):
    try:
      os.mkdir(f'./{id}')
    except:
      pass 
  os.chdir(f'./{id}')  
  ms=['1']
  
  if args.social:
    ps=[]
    for m in ms:
      cmd=f'python -u {file_path}/fit.py -s -m {m} >>./social_model_{m}.log'
      ps.append(subprocess.Popen(cmd,shell=True))
    for p in ps:
      p.wait()
    subprocess.run(f'python -u {file_path}/draw.py -s',shell=True)

  elif args.nonsocial:
    ps=[]
    for m in ms:
      cmd=f'python -u {file_path}/fit.py -m {m} >>./nonsocial_model_{m}.log'
      ps.append(subprocess.Popen(cmd,shell=True))
    for p in ps:
      p.wait()
    subprocess.run(f'python -u {file_path}/draw.py',shell=True)
  else:
    
    ps=[]
    for m in ms:
      cmd=f'python -u {file_path}/fit.py -s -m {m} >>./social_model_{m}.log'
      ps.append(subprocess.Popen(cmd,shell=True))
      
    for m in ms:
      cmd=f'python -u {file_path}/fit.py -m {m} >>./nonsocial_model_{m}.log'
      ps.append(subprocess.Popen(cmd,shell=True))
    for p in ps:
      p.wait()
      
    subprocess.run(f'python -u {file_path}/draw.py -s',shell=True)
    subprocess.run(f'python -u {file_path}/draw.py',shell=True)
    
  os.chdir('..')


n=10
from tqdm import trange
for i in trange(args.start_id,args.start_id+n):
  run_one(i)
