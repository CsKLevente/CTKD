# Use this script to train your own student model.

# PKT
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet20 -r 1 -a 1 -b 30000
# SP
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill similarity --model_s resnet20 -r 1 -a 1 -b 3000
# VID
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill vid --model_s resnet20 -r 1 -a 1 -b 1
# CRD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet20 -r 1 -a 1 -b 0.8
# SRRL
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill srrl --model_s resnet20 -r 1 -a 1 -b 1
# DKD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill srrl --model_s resnet20 -r 1 -a 0 -b 1 --dkd_alpha 1 --dkd_beta 1


## KD ResNet18
#for i in {1..5}
#do
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet18 -r 1 -a 1 -b 0 --kd_T 4 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model \
#        --experiments_dir 'imagenette-tea-res34-stu-res18/KD/baseline' \
#        --experiments_name fold-$i
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet18 -r 1 -a 1 -b 0 --kd_T 4 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model --sparsity_learning \
#        --experiments_dir 'imagenette-tea-res34-stu-res18/KD/sparsity_learning/kd_T_4' \
#        --experiments_name fold-$i
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet18 -r 1 -a 1 -b 0 --kd_T 1.5 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model --sparsity_learning \
#        --experiments_dir 'imagenette-tea-res34-stu-res18/KD/sparsity_learning/kd_T_1.5' \
#        --experiments_name fold-$i
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet18 -r 1 -a 1 -b 0 --kd_T 2 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model --sparsity_learning \
#        --experiments_dir 'imagenette-tea-res34-stu-res18/KD/sparsity_learning/kd_T_2' \
#        --experiments_name fold-$i
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet18 -r 1 -a 1 -b 0 --kd_T 8 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model --sparsity_learning \
#        --experiments_dir 'imagenette-tea-res34-stu-res18/KD/sparsity_learning/kd_T_8' \
#        --experiments_name fold-$i
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet18 -r 1 -a 1 -b 0 --kd_T 16 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model --sparsity_learning \
#        --experiments_dir 'imagenette-tea-res34-stu-res18/KD/sparsity_learning/kd_T_16' \
#        --experiments_name fold-$i
#
#
## with CTKD
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet18 -r 1 -a 1 -b 0 --kd_T 4 \
#        --have_mlp 1 --mlp_name 'global' \
#        --t_start 1 --t_end 20 --cosine_decay 1 \
#        --decay_max 0 --decay_min -1 --decay_loops 5 \
#        --save_model \
#        --experiments_dir 'imagenette-tea-res34-stu-res18/KD/ctkd_120/baseline' \
#        --experiments_name fold-$i
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet18 -r 1 -a 1 -b 0 --kd_T 4 \
#        --have_mlp 1 --mlp_name 'global' \
#        --t_start 1 --t_end 20 --cosine_decay 1 \
#        --decay_max 0 --decay_min -1 --decay_loops 5 \
#        --save_model --sparsity_learning \
#        --experiments_dir 'imagenette-tea-res34-stu-res18/KD/ctkd_120/sparsity_learning/kd_T_4' \
#        --experiments_name fold-$i
#done




## KD ResNet10
#for i in {2..5}
#do
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 4 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model \
#        --experiments_dir 'imagenette-tea-res34-stu-res10/KD/baseline' \
#        --experiments_name fold-$i
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 4 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model --sparsity_learning \
#        --experiments_dir 'imagenette-tea-res34-stu-res10/KD/sparsity_learning/kd_T_4' \
#        --experiments_name fold-$i
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 1.5 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model --sparsity_learning \
#        --experiments_dir 'imagenette-tea-res34-stu-res10/KD/sparsity_learning/kd_T_1.5' \
#        --experiments_name fold-$i
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 2 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model --sparsity_learning \
#        --experiments_dir 'imagenette-tea-res34-stu-res10/KD/sparsity_learning/kd_T_2' \
#        --experiments_name fold-$i
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 8 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model --sparsity_learning \
#        --experiments_dir 'imagenette-tea-res34-stu-res10/KD/sparsity_learning/kd_T_8' \
#        --experiments_name fold-$i
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 16 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model --sparsity_learning \
#        --experiments_dir 'imagenette-tea-res34-stu-res10/KD/sparsity_learning/kd_T_16' \
#        --experiments_name fold-$i
#
## with CTKD
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 4 \
#        --have_mlp 1 --mlp_name 'global' \
#        --t_start 1 --t_end 20 --cosine_decay 1 \
#        --decay_max 0 --decay_min -1 --decay_loops 5 \
#        --save_model \
#        --experiments_dir 'imagenette-tea-res34-stu-res10/KD/ctkd_120/baseline' \
#        --experiments_name fold-$i
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 120 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 4 \
#        --have_mlp 1 --mlp_name 'global' \
#        --t_start 1 --t_end 20 --cosine_decay 1 \
#        --decay_max 0 --decay_min -1 --decay_loops 5 \
#        --save_model --sparsity_learning \
#        --experiments_dir 'imagenette-tea-res34-stu-res10/KD/ctkd_120/sparsity_learning/kd_T_4' \
#        --experiments_name fold-$i
#done

# KD ResNet8 -> ResNet10
for alpha in {1,2,4,8,16,32,64,128,256,512,1024}
do
  for i in {2..4}
  do
#python3 train_student.py --path-t ./save/models/baseline/imagenette-half_size/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 200 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 4 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model \
#        --experiments_dir 'dynamic_training/imagenette-half_size-tea-vgg16-stu-res10/KD/baseline' \
#        --experiments_name fold-$i \
#        --half_size_img --early_stopping --decay_on_plateau
#
python3 train_student.py --path-t ./save/models/baseline/imagenette-half_size/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
        --distill kd \
        --batch_size 64 --epochs 200 --dataset imagenette \
        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 4 \
        --have_mlp 0 --mlp_name 'global' \
        --save_model --sparsity_learning \
        --experiments_dir 'shrinkage_strength/dynamic_training/imagenette-half_size-tea-vgg16-stu-res10/KD/sparsity_learning/kd_T_4' \
        --experiments_name $alpha/fold-$i \
        --half_size_img --early_stopping --decay_on_plateau \
        --shrinkage_strength $alpha
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette-half_size/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 200 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 1.5 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model --sparsity_learning \
#        --experiments_dir 'dynamic_training/imagenette-half_size-tea-vgg16-stu-res10/KD/sparsity_learning/kd_T_1.5' \
#        --experiments_name fold-$i \
#        --half_size_img --early_stopping --decay_on_plateau
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette-half_size/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 200 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 2 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model --sparsity_learning \
#        --experiments_dir 'dynamic_training/imagenette-half_size-tea-vgg16-stu-res10/KD/sparsity_learning/kd_T_2' \
#        --experiments_name fold-$i \
#        --half_size_img --early_stopping --decay_on_plateau
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette-half_size/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 200 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 8 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model --sparsity_learning \
#        --experiments_dir 'dynamic_training/imagenette-half_size-tea-vgg16-stu-res10/KD/sparsity_learning/kd_T_8' \
#        --experiments_name fold-$i \
#        --half_size_img --early_stopping --decay_on_plateau
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette-half_size/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 200 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 16 \
#        --have_mlp 0 --mlp_name 'global' \
#        --save_model --sparsity_learning \
#        --experiments_dir 'dynamic_training/imagenette-half_size-tea-vgg16-stu-res10/KD/sparsity_learning/kd_T_16' \
#        --experiments_name fold-$i \
#        --half_size_img --early_stopping --decay_on_plateau
#
#
## with CTKD
#
#python3 train_student.py --path-t ./save/models/baseline/imagenette-half_size/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
#        --distill kd \
#        --batch_size 64 --epochs 200 --dataset imagenette \
#        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
#        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 4 \
#        --have_mlp 1 --mlp_name 'global' \
#        --t_start 1 --t_end 20 --cosine_decay 1 \
#        --decay_max 0 --decay_min -1 --decay_loops 5 \
#        --save_model \
#        --experiments_dir 'dynamic_training/imagenette-half_size-tea-vgg16-stu-res10/KD/ctkd/baseline' \
#        --experiments_name fold-$i \
#        --half_size_img --early_stopping --decay_on_plateau
#
python3 train_student.py --path-t ./save/models/baseline/imagenette-half_size/vgg16/vgg16/vgg16_ckpt_epoch_200.pth \
        --distill kd \
        --batch_size 64 --epochs 200 --dataset imagenette \
        --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
        --model_s ResNet10 -r 1 -a 1 -b 0 --kd_T 4 \
        --have_mlp 1 --mlp_name 'global' \
        --t_start 1 --t_end 20 --cosine_decay 1 \
        --decay_max 0 --decay_min -1 --decay_loops 5 \
        --save_model --sparsity_learning \
        --experiments_dir 'shrinkage_strength/dynamic_training/imagenette-half_size-tea-vgg16-stu-res10/KD/ctkd/sparsity_learning/kd_T_4' \
        --experiments_name $alpha/fold-$i \
        --half_size_img --early_stopping --decay_on_plateau \
        --shrinkage_strength $alpha

  done  # i
done  #alpha