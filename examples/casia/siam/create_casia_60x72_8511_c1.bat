
echo off
set EXAMPLE=.
set DATA=../../data/casia

set TRAIN_DATA_ROOT=D:/data/face/CASIA-WebFacec-60x72-check/
set VAL_DATA_ROOT=D:/data/face/CASIA-WebFacec-60x72-check/
set RESIZE_WIDTH=0
set RESIZE_HEIGHT=0
set GLOG_logtostderr=1 

echo on
echo "Creating train lmdb..."

"../../../bin-gpu-siam/convert_imageset_siamese" --resize_height=%RESIZE_HEIGHT% --resize_width=%RESIZE_WIDTH% --gray=true %TRAIN_DATA_ROOT% casia_list_60x72_check_above20_5914_all.txt %EXAMPLE%/casia_train_lmdb_60x72_c1_ex

echo "Creating val lmdb..."

"../../../bin-gpu-siam/convert_imageset_siamese" --resize_height=%RESIZE_HEIGHT% --resize_width=%RESIZE_WIDTH% --gray=true %VAL_DATA_ROOT% casia_list_60x72_check_below20_all.txt %EXAMPLE%/casia_val_lmdb_60x72_c1_ex

echo "Done."

pause