'''
python test.py \
    --name t2net_unreal2nyu --model test \
    --img_source_file ../datasets/nyu_data/test_color \
    --img_target_file ../datasets/nyu_data/test_color \
    --gpu_ids 1 --ntest 654 --norm instance

python test.py \
    --name t2net_simgan_vkitti2kitti --model test \
    --img_source_dir ../datasets/vkitti_data/train_color \
    --img_target_dir ./datasplit/eigen_test_files.txt \
    --lab_source_dir ../datasets/vkitti_data/train_depth \
    --lab_target_dir ./datasplit/eigen_test_files.txt \
    --txt_data_path ../datasets/kitti_data \
    --load_size 640 192 \
    --gpu_ids 1 --ntest 697 --norm instance

'''


import os
from options.test_options import TestOptions
from dataloader.data_loader import dataloader
from model.models import create_model
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
opt.dataset_mode = 'paired'

dataset = dataloader(opt)
dataset_size = len(dataset) * opt.batch_size
print ('testing images = %d ' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir,opt.name, '%s_%s' %(opt.phase, opt.which_epoch))
web_page = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# testing
for i,data in enumerate(dataset):
    model.set_input(data)
    model.test()
    model.save_results(visualizer, web_page)

    if i == opt.ntest:
        break