set -ex
python train.py \
--name t2net_simgan_unreal2nyu_batch_B12 \
--model wsupervised \
--img_source_dir ../../data/akada/datasets/unreal2nyu/trainA \
--img_target_dir ../../data/akada/datasets/unreal2nyu/trainB \
--lab_source_dir ../../data/akada/datasets/unreal2nyu/trainA_depth \
--lab_target_dir ../../data/akada/datasets/unreal2nyu/trainB_depth \
--norm batch \
--batch_size 12 --niter 10 --niter_decay 10 \
--shuffle --flip --rotation \
--gpu_ids 0,1 --no_html --display_id -1 \
--lr_task 2e-4 \
--lr_trans 4e-5 \