set -ex
python train.py \
--name t2net_simgan_unreal2nyu_gp_wofarth \
--model wsupervised \
--img_source_dir ../datasets/unreal2nyu/trainA \
--img_target_dir ../datasets/unreal2nyu/trainB \
--lab_source_dir ../datasets/unreal2nyu/trainA_depth \
--lab_target_dir ../datasets/unreal2nyu/trainB_depth \
--norm instance \
--batch_size 1 --niter 10 --niter_decay 10 --shuffle --flip --rotation \
--gpu_ids 0 --no_html --display_id -1 \
--gp \
--experiment
