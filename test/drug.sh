python run.py \
-d drug \
-p ../data/DDdataset \
-cuda cuda:0 \
-epochs 300 \
-b 128 \
-train \
-n 13
# -s 1 \

# GraphHINGE_FFT -> 原本为FFT实现，但是改成了 conv1d 实现
# GraphHINGE_CNN -> interaction 模块改为可训练的CNN， 以证明 interaction 的有效性。 Test Loss 1.0100 | Test AUC 0.8398 | Test ACC 0.7687 | Test F1 0.7563 | Test Logloss 0.9976 |

# auc 差不多 ACC CNN 的低
# TODO 评估指标的选择

# best epoch 70