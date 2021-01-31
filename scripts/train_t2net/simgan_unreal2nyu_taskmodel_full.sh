set -ex
python train.py \
--name t2net_simgan_unreal2nyu_taskmodel_full \
--model supervised_full \
--img_source_dir ../datasets/unreal2nyu/trainA \
--img_target_dir ../datasets/unreal2nyu/trainB \
--lab_source_dir ../datasets/unreal2nyu/trainA_depth \
--lab_target_dir ../datasets/unreal2nyu/trainB_depth \
--norm batch \
--batch_size 6 --niter 10 --niter_decay 10 \
--shuffle --flip --rotation \
--gpu_ids 0,1 --no_html --display_id -1 \
--load_size 256 192 \
--lambda_rec_img 40 \
--lambda_rec_lab 20 \
--lr_task 0.0001 \
--lr_trans 0.00005 \
--mixed