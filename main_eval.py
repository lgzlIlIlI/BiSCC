from __future__ import print_function
import argparse
import os
import torch
import model_eval
import multiprocessing as mp
import wsad_dataset
import random
from test_eval import test
import options
import numpy as np
torch.set_default_tensor_type('torch.cuda.FloatTensor')
def setup_seed(seed):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

import torch.optim as optim

if __name__ == '__main__':
   pool = mp.Pool(5)
   os.environ["CUDA_VISIBLE_DEVICES"]  = '0'
   args = options.parser.parse_args()
   seed=args.seed
   setup_seed(seed)
   device = torch.device("cuda")
   
   dataset = getattr(wsad_dataset,args.dataset)(args)
   max_map=[0]*9
   model1 = model_eval.Base0(dataset.feature_size, dataset.num_class,opt=args).to(device)
   model1_dict = model1.state_dict()
   checkpoint = torch.load('/media/lgz/Scipio/lgz/Bi-SCC/our_biscc2/ckpt_eval/alphatest_base_withDrop_new .pkl')
   model1.load_state_dict(checkpoint)
   itr=0
   iou,dmap = test(itr, dataset, args, model1,device,pool)
   #torch.save(model1.state_dict(), './ckpt_eval/alphatest_base_withDrop_new .pkl')

    
