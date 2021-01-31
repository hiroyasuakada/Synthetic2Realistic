set -ex
python train.py \
--name t2net_simgan_unreal2nyu_batch_B6 \
--model wsupervised \
--img_source_dir ../datasets/unreal2nyu/trainA \
--img_target_dir ../datasets/unreal2nyu/trainB \
--lab_source_dir ../datasets/unreal2nyu/trainA_depth \
--lab_target_dir ../datasets/unreal2nyu/trainB_depth \
--norm batch \
--batch_size 6 --niter 10 --niter_decay 10 \
--shuffle --flip --rotation \
--gpu_ids 3,2 --no_html --display_id -1 \
--lr_task 1e-4 \
--lr_trans 2e-5 \