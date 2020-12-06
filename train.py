"""
python train.py --name unreal2nyu --model wsupervised \
    --img_source_file ../../data/akada/datasets/unreal2nyu/trainA \
    --img_target_file ../../data/akada/datasets/unreal2nyu/trainB \
    --lab_source_file ../../data/akada/datasets/unreal2nyu/trainA_depth \
    --lab_target_file ../../data/akada/datasets/unreal2nyu/trainB_depth \
    --gpu_ids 1 --shuffle --flip --rotation --no_html --display_id -1 --norm instance

"""

import time
from options.train_options import TrainOptions
from dataloader.data_loader import dataloader
from model.models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse()

dataset = dataloader(opt)
dataset_size = len(dataset) * opt.batch_size
print('training images = %d' % dataset_size)

# create datasets for Gaussian Process
labeled_dataset = None
unlabeled_dataset = None
if opt.gp:
    labeled_dataset, unlabeled_dataset = dataloader(opt, gp=True)
    print('The number of labeled training images for GP = %d' % len(labeled_dataset))
    print('The number of unlabeled training images for GP = %d' % len(unlabeled_dataset))


model = create_model(opt, labeled_dataset, unlabeled_dataset)
visualizer = Visualizer(opt)
total_steps=0

for epoch in range(opt.epoch_count, opt.niter+opt.niter_decay+1):
    epoch_start_time = time.time()
    epoch_iter = 0

    # training
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)
        model.optimize_parameters(i)

        if total_steps % opt.display_freq == 0:
            if epoch >= opt.transform_epoch:
                model.validation_target()
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save_networks('latest')


    # training for Task network with Gaussian Process
    if opt.gp:
        
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d for I2I and Task' % (epoch, total_steps))
            model.save_networks('{}_middle'.format(epoch))

        print('End of epoch for I2I and Task %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))   


        start_time_gp = time.time()
        iter_time_gp = 0
        epoch_iters_gp = 0
        print('Storing latent vectors ...')
        model.generate_fmaps_GP()
        print('Time spent on storing latent vectors: %d sec' % (time.time() - start_time_gp))

        for i, data in enumerate(unlabeled_dataset):
            iter_start_time_gp = time.time()
            epoch_iters_gp += opt.batch_size

            model.optimize_parameters_GP(i, data)

            iter_time_gp += time.time() - iter_start_time_gp 
            if epoch_iters_gp % opt.print_freq == 0:    # print training losses and save logging information to the disk
                print('Gaussian Process: epoch %d, itar %d, time %d sec, loss_gp %.8f' % (epoch, epoch_iters_gp, iter_time_gp, model.loss_gp.item()/opt.batch_size))
                iter_time_gp = 0

            if epoch_iters_gp % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model with GP (epoch %d, total_iters %d)' % (epoch, epoch_iters_gp))
                save_suffix = 'latest'
                model.save_networks(save_suffix)

        print('=== End of epoch for GP %d / %d \t Time Taken: %d sec ===' % (epoch, opt.niter + opt.niter_decay, time.time() - start_time_gp))


    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch (epoch %d, iters %d)' % (epoch, total_steps))
        model.save_networks('latest')
        model.save_networks(epoch)

    print ('End of the epoch for all networks %d / %d \t Time Take: %d sec' %
           (epoch, opt.niter + opt.niter_decay, time.time()-epoch_start_time))

    model.update_learning_rate()
