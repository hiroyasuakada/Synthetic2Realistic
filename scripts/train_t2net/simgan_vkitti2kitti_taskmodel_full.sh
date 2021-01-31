set -ex
python train.py \
--name t2net_simgan_vkitti2kitti_taskmodel_full \
--model supervised_full \
--img_source_dir ../datasets/vkitti_data/train_color \
--img_target_dir ./datasplit/eigen_train_files.txt \
--lab_source_dir ../datasets/vkitti_data/train_depth \
--lab_target_dir ../datasets/kitti_data/eigen_train_depth_gt \
--txt_data_path ../datasets/kitti_data \
--norm batch \
--batch_size 6 --niter 10 --niter_decay 10 \
--shuffle --flip --rotation \
--gpu_ids 0,1 --no_html --display_id -1 \
--load_size 640 192 \
--lambda_rec_img 100 \
--lambda_rec_lab 100 \
--lr_task 0.0001 \
--lr_trans 0.00005 \
