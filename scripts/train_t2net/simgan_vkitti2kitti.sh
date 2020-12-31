set -ex
python train.py \
--name t2net_simgan_vkitti2kitti \
--model wsupervised \
--img_source_dir ../datasets/vkitti_data/train_color \
--img_target_dir ./datasplit/eigen_train_files.txt \
--lab_source_dir ../datasets/vkitti_data/train_depth \
--lab_target_dir ./datasplit/eigen_train_files.txt \
--txt_data_path ../datasets/kitti_data \
--norm instance \
--batch_size 1 --niter 10 --niter_decay 10 \
--shuffle --flip --rotation \
--gpu_ids 1 --no_html --display_id -1 \
--load_size 640 192 \
--lambda_rec_img 100 \
--lambda_rec_lab 100