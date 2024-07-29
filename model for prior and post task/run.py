
import subprocess
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--post',action='store_true')
parser.add_argument('-r', '--prior',action='store_true')
parser.add_argument('-e', '--eval', action='store_true')
parser.add_argument('-p', '--precomputed', action='store_true')
args = parser.parse_args()
# ms=['1','2','3','4','5','6','7','8']

# ms=['5','6','7','8']
e=''
if args.eval:
  e=' -e'
pre=''
if args.precomputed:
  pre=' -p'

file_path=os.path.dirname(os.path.abspath(__file__))

if args.post:
  ms=['1','2','3','4','5','6','7','8']
  ps=[]
  for m in ms:
    cmd=f'python -u "{file_path}/fit.py"{e}{pre} -o -m {m} >>"{file_path}/../../results/post_model_{m}.log"'
    ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()
    

elif args.prior:
  ms=['5','6','7','8']
  ps=[]
  for m in ms:
    cmd=f'python -u "{file_path}/fit.py"{e}{pre} -m {m} >>"{file_path}/../../results/prior_model_{m}.log"'
    ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()
    
    
else:
  
  ms=['5','6','7','8']
  ps=[]
  for m in ms:
    cmd=f'python -u "{file_path}/fit.py"{e}{pre} -m {m} >>"{file_path}/../../results/prior_model_{m}.log"'
    ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()
    
    

  ms=['1','2','3','4','5','6','7','8']
  ps=[]
  for m in ms:
    cmd=f'python -u "{file_path}/fit.py"{e}{pre} -o -m {m} >>"{file_path}/../../results/post_model_{m}.log"'
    ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()


  ps=[]
  cmd=f'python -u "{file_path}/draw_prior.py"{e}{pre}'
  ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()

    

