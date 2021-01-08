set -ex
python train.py \
--name test \
--model wsupervised \
--img_source_dir ../../data/akada/datasets/unreal2nyu/trainA \
--img_target_dir ../../data/akada/datasets/unreal2nyu/trainB \
--lab_source_dir ../../data/akada/datasets/unreal2nyu/trainA_depth \
--lab_target_dir ../../data/akada/datasets/unreal2nyu/trainB_depth \
--batch_size 1 --niter 10 --niter_decay 10 \
--shuffle --flip --rotation \
--gpu_ids 1 --no_html --display_id -1 \