set -ex
python train.py \
--name t2net_simgan_vkitti2kitti \
--model wsupervised \
--img_source_dir ../../data/akada/datasets/vkitti_data/train_color \
--img_target_dir ./datasplit/eigen_train_files.txt \
--lab_source_dir ../../data/akada/datasets/vkitti_data/train_depth \
--lab_target_dir ./datasplit/eigen_train_files.txt \
--txt_data_path ../../data/akada/datasets/kitti_data \
--norm instance \
--batch_size 6 --niter 10 --niter_decay 10 \
--shuffle --flip --rotation \
--gpu_ids 2 --no_html --display_id -1 \
--load_size 640 192 \
--lambda_rec_img 100 \
--lambda_rec_lab 100 \
--lr_trans 5e-5