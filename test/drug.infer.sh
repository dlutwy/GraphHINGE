python run.py \
-d drug \
-p ../data/DDdataset \
-cuda cuda:0 \
-epochs 70 \
-b 128 \
-inference \
-n 13 \
-train
# -s 1
# -model_state_file NotFFTGraphHINGE_FFT_drug_train_lr_0.001_bs_128_wd_0.0009.pth
# -usersid 1 \
# -usersid 2 \
# -usersid 3 \
