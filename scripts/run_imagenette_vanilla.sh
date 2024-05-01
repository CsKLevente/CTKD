# Use this script to train your own teacher model.

python3 train_teacher.py --model vgg16 \
                --dataset 'imagenette' \
                --batch_size 256 \
                --epochs 200 \
                --learning_rate 0.1 \
                --lr_decay_epochs '40,80,120,160' \
                --lr_decay_rate 0.1 \
                --weight_decay 1e-4 \
                --experiments_dir 'baseline/imagenette-half_size/vgg16' \
                --experiments_name 'vgg16' \
                --save_model --print-freq 15 \
                --half_size_img
