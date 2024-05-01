# Use this script to train your own teacher model.

python3 train_teacher.py --model ResNet34 \
                --dataset 'imagewoof' \
                --batch_size 128 \
                --epochs 120 \
                --learning_rate 0.1 \
                --lr_decay_epochs '30,60,90' \
                --lr_decay_rate 0.1 \
                --weight_decay 1e-4 \
                --experiments_dir 'baseline/imagewoof/ResNet34' \
                --experiments_name 'ResNet34' \
                --save_model --print-freq 15
