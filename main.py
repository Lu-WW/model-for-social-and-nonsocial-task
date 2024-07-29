
import subprocess
if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--eval', action='store_true')
  parser.add_argument('-p', '--precomputed', action='store_true')
  args = parser.parse_args()

    
  e=''
  if args.eval:
    e=' -e'
  pre=''
  if args.precomputed:
    pre=' -p'

  print('Running social/nonsocial task')
  ps=[]
  cmd=f'python "./model for social and nonsocial task/run.py"{e}{pre}'
  ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()

  print('Running social/nonsocial task simulation')
  ps=[]
  cmd=f'python "./model for social and nonsocial task/sim.py"{e}{pre}'
  ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()

    
  print('Running prior/post task')
  ps=[]
  cmd=f'python "./model for prior and post task/run.py"{e}{pre}'
  ps.append(subprocess.Popen(cmd,shell=True))
  for p in ps:
    p.wait()




    