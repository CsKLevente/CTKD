# Use this script to train your own student model.
model_path=/home/koppany/TUDelft/thesis/git/Torch-Pruning/benchmarks/run/ctkd_models/resnet20

# KD
python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd \
        --model_s resnet20 -r 0.1 -a 0.9 -b 0 --kd_T 4 \
        --path-s $model_path/su_3.0/cifar100/prune/cifar100-global-group_sl-resnet20/cifar100_resnet20_group_sl.pth \
        --learning_rate 0.001 --epochs 300 --lr_decay_epochs '150,180,210, 240, 270' \
        --save_model --experiments_dir 'tea-res56-stu-res20/pruning+KD' --experiments_name 'torch_pruning/su_3.0/run_4'
python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd \
        --model_s resnet20 -r 0.1 -a 0.9 -b 0 --kd_T 4 \
        --path-s $model_path/su_2.0/cifar100/prune/cifar100-global-group_sl-resnet20/cifar100_resnet20_group_sl.pth \
        --learning_rate 0.001 --epochs 300 --lr_decay_epochs '150,180,210, 240, 270' \
        --save_model --experiments_dir 'tea-res56-stu-res20/pruning+KD' --experiments_name 'torch_pruning/su_2.0/run_2'
python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd \
        --model_s resnet20 -r 0.1 -a 0.9 -b 0 --kd_T 4 \
        --path-s $model_path/su_1.5/cifar100/prune/cifar100-global-group_sl-resnet20/cifar100_resnet20_group_sl.pth \
        --learning_rate 0.001 --epochs 300 --lr_decay_epochs '150,180,210, 240, 270' \
        --save_model --experiments_dir 'tea-res56-stu-res20/pruning+KD' --experiments_name 'torch_pruning/su_1.5/run_2'
python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd \
        --model_s resnet20 -r 0.1 -a 0.9 -b 0 --kd_T 4 \
        --path-s $model_path/su_1.11/cifar100/prune/cifar100-global-group_sl-resnet20/cifar100_resnet20_group_sl.pth \
        --learning_rate 0.001 --epochs 300 --lr_decay_epochs '150,180,210, 240, 270' \
        --save_model --experiments_dir 'tea-res56-stu-res20/pruning+KD' --experiments_name 'torch_pruning/su_1.1/run_2'

## KD+CTKD
#python3 train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd \
#        --model_s resnet20 -r 0.1 -a 0.9 -b 0 --kd_T 4 \
#        --have_mlp 1 --mlp_name 'global' \
#        --cosine_decay 1 --decay_max 1 --decay_min -1 --decay_loops 10 \
#        --save_model --experiments_dir 'tea-res56-stu-res18/KD/CTKD' --experiments_name 'fold-1'

## PKT
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill pkt \
#        --model_s resnet20 -r 0.1 -a 0.9 -b 30000 \
#        --save_model --experiments_dir 'tea-res56-stu-res20/PKT'
## PKT+CTKD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill pkt \
#        --model_s resnet20 -r 0.1 -a 0.9 -b 30000 \
#        --have_mlp 1 --mlp_name 'global' \
#        --cosine_decay 1 --decay_max 1 --decay_min -1 --decay_loops 10 \
#        --save_model --experiments_dir 'tea-res56-stu-res20/PKT/CTKD' --experiments_name 'fold-1'

## SP
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill similarity \
#        --model_s resnet20 -r 0.1 -a 0.9 -b 3000 \
#        --save_model --experiments_dir 'tea-res56-stu-res20/SP'
## SP+CTKD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill similarity \
#        --model_s resnet20 -r 0.1 -a 0.9 -b 3000 \
#        --have_mlp 1 --mlp_name 'global' \
#        --cosine_decay 1 --decay_max 1 --decay_min -1 --decay_loops 10 \
#        --save_model --experiments_dir 'tea-res56-stu-res20/SP/CTKD' --experiments_name 'fold-1'

## VID
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill vid \
#        --model_s resnet20 -r 0.1 -a 0.9 -b 1 \
#        --save_model --experiments_dir 'tea-res56-stu-res20/VID'
## VID+CTKD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill vid \
#        --model_s resnet20 -r 0.1 -a 0.9 -b 1 \
#        --have_mlp 1 --mlp_name 'global' \
#        --cosine_decay 1 --decay_max 1 --decay_min -1 --decay_loops 10 \
#        --save_model --experiments_dir 'tea-res56-stu-res20/VID/CTKD' --experiments_name 'fold-1'

## CRD
## python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill crd \
##        --model_s resnet20 -r 0.1 -a 0.9 -b 0.8 \
##        --save_model --experiments_dir 'tea-res56-stu-res20/CRD'
## CRD+CTKD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill crd \
#        --model_s resnet20 -r 0.1 -a 0.9 -b 0.8 \
#        --have_mlp 1 --mlp_name 'global' \
#        --cosine_decay 1 --decay_max 1 --decay_min -1 --decay_loops 10 \
#        --save_model --experiments_dir 'tea-res56-stu-res20/CRD/CTKD' --experiments_name 'fold-1'

## SRRL
## python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill srrl \
##        --model_s resnet20 -r 0.1 -a 0.9 -b 1 \
##        --save_model --experiments_dir 'tea-res56-stu-res20/SRRL'
## SRRL+CTKD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill srrl \
#        --model_s resnet20 -r 0.1 -a 0.9 -b 1 \
#        --have_mlp 1 --mlp_name 'global' \
#        --cosine_decay 1 --decay_max 1 --decay_min -1 --decay_loops 10 \
#        --save_model --experiments_dir 'tea-res56-stu-res20/SRRL/CTKD' --experiments_name 'fold-2'

# DKD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill dkd \
#        --model_s resnet20 -r 1 -a 0 -b 1 --dkd_alpha 1 --dkd_beta 2 \
#        --save_model --experiments_dir 'tea-res56-stu-res20/DKD'
# DKD+CTKD TODO: Figure out why it fails with CTKD enabled (temp becomes nan)
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill dkd \
#        --model_s resnet20 -r 1 -a 0 -b 1 --dkd_alpha 1 --dkd_beta 2 \
#        --have_mlp 1 --mlp_name 'global' \
#        --cosine_decay 1 --decay_max 0 --decay_min -1 --decay_loops 10 \
#        --learning_rate 0.05 \
#        --save_model --experiments_dir 'tea-res56-stu-res20/DKD/CTKD' --experiments_name 'fold-2'


#python3 train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth \
#        --distill kd \
#        --model_s resnet20 -r 0.1 -a 0.9 -b 0 --kd_T 4 \
#        --batch_size 64 --learning_rate 0.05 \
#        --have_mlp 1 --mlp_name 'global' \
#        --cosine_decay 1 --decay_max 0 --decay_min -1 --decay_loops 10 \
#        --save_model \
#        --experiments_dir 'tea-res56-stu-res20/KD/CTKD' \
#        --experiments_name 'fold-1'
        
