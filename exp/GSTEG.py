#!/usr/bin/env python
import sys
import pdb
import traceback
#sys.path.insert(0, '..')
sys.path.insert(0, '.')
from main import main
from bdb import BdbQuit
import subprocess
subprocess.Popen('find ./exp/.. -iname "*.pyc" -delete'.split())

args = [
    '--name', __file__.split('/')[-1].split('.')[0],  # name is filename
    '--cache-dir', './cr_caches',
    '--rgb-data', '/home/ubuntu/10618-Project/dataset/Charades_v1_rgb',
    '--rgb-pretrained-weights', './rgb_i3d_pretrained.pt',
    '--resume', './cr_caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar',
    '--train-file', 'Charades_v1_train_500.csv',
    # '--val-file', '/home/ubuntu/10618-Project/Charades_v1_test.csv',
    '--val-file', 'Charades_v1_test_200.csv',
    '--groundtruth-lookup', './utils/groundtruth_200.p'    
#'--evaluate',
]
sys.argv.extend(args)
try:
    main()
except BdbQuit:
    sys.exit(1)
except Exception:
    traceback.print_exc()
    print('')
    pdb.post_mortem()
    sys.exit(1)


