# !/bin/bash
target_prop="homo"    #Available options:homo lumo
device_no=0         #gpu number
expt_no=3
data_set="qm9_pyg_3.pt" #processed dataset.  Should be saved in dataset/qm9/processed folder
selection_method="gradient_conflicts" #, random, coreset, unc_div, mqr_ud, gradient_conflicts
epochs=200 #number of epochs to train
backbone="spherenet" #dimenet, spherenet
addendum=1500 #number of samples to label in each AL iteration
init_size=5000 #number of initially labeled samples
train_size=28000 #total number of samples in data pool (labeled + unlabeled)
valid_size=2000 #number of samples in valid size
for cycle_no in {0..7} #No of AL iterations
do
    python3 train.py --init_size $init_size --train_size $train_size --valid_size $valid_size \
        --ADDENDUM $addendum --backbone $backbone \
        --epochs $epochs --device $device_no --target $target_prop --dataset $data_set \
        --selection_method $selection_method --cycle $cycle_no --expt $expt_no
done