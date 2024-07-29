
import subprocess
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--social',action='store_true')
parser.add_argument('-n', '--nonsocial',action='store_true')
parser.add_argument('-e', '--eval', action='store_true')
parser.add_argument('-p', '--precomputed', action='store_true')
args = parser.parse_args()
ms=['1','2','3','4','5','6','7','8']

e=''
if args.eval:
  e=' -e'
pre=''
if args.precomputed:
  pre=' -p'

file_path=os.path.dirname(os.path.abspath(__file__))


if args.social:
  ps=[]
  for m in ms:
    cmd=f'python -u "{file_path}/fit.py"{e}{pre} -s -m {m} >>"{file_path}/../../results/social_model_{m}.log"'
    ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()
    
  ps=[]
  cmd=f'python -u "{file_path}/draw.py"{e}{pre} -s'
  ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()


elif args.nonsocial:
  ps=[]
  for m in ms:
    cmd=f'python -u "{file_path}/fit.py"{e}{pre} -m {m} >>"{file_path}/../../results/nonsocial_model_{m}.log"'
    ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()
    
  ps=[]
  cmd=f'python -u "{file_path}/draw.py"{e}{pre}'
  ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()
    
    
else:
  
  ps=[]
  for m in ms:
    cmd=f'python -u "{file_path}/fit.py"{e}{pre} -s -m {m} >>"{file_path}/../../results/social_model_{m}.log"'
    ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()
    

  ps=[]
  for m in ms:
    cmd=f'python -u "{file_path}/fit.py"{e}{pre} -m {m} >>"{file_path}/../../results/nonsocial_model_{m}.log"'
    ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()
    
  ps=[]
  cmd=f'python -u "{file_path}/draw.py"{e}{pre}'
  ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()
    
  ps=[]
  cmd=f'python -u "{file_path}/draw.py"{e}{pre} -s'
  ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()
