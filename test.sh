set -x

python test.py \
    --dt_weights yolov7/weights/yolov7-tiny-best.pt \
    --pe_weights A2J/checkpoint/epoch#29_lr_0.00035_wetD_0.00010_stepSize_10_gamma_0.1.pt \
    --source datasets/test/depth_data \
    --annotations datasets/test/res/ICU_test_labels.h5 \
    --save-json \
    --classes 0 \
    --nosave