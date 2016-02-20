
echo off
set EXAMPLE=.
set DATA=../../data/casia

set TRAIN_DATA_ROOT=D:/data/CASIA-WebFacec-60x72-check/
set VAL_DATA_ROOT=D:/data/CASIA-WebFacec-60x72-check/
set RESIZE_WIDTH=0
set RESIZE_HEIGHT=0
set GLOG_logtostderr=1 

echo on
echo "Creating train lmdb..."

"../../../bin-gpu-release/convert_imageset" --resize_height=%RESIZE_HEIGHT% --resize_width=%RESIZE_WIDTH% --gray=true %TRAIN_DATA_ROOT% %EXAMPLE%/pair_data1.txt %EXAMPLE%/casia_train_lmdb_60x72_c1_pair1

"../../../bin-gpu-release/convert_imageset" --resize_height=%RESIZE_HEIGHT% --resize_width=%RESIZE_WIDTH% --gray=true %TRAIN_DATA_ROOT% %EXAMPLE%/pair_data2.txt %EXAMPLE%/casia_train_lmdb_60x72_c1_pair2

echo "Creating val lmdb..."

"../../../bin-gpu-release/convert_imageset" --resize_height=%RESIZE_HEIGHT% --resize_width=%RESIZE_WIDTH% --gray=true %VAL_DATA_ROOT% %EXAMPLE%/pair_data1_val.txt %EXAMPLE%/casia_val_lmdb_60x72_c1_pair1

"../../../bin-gpu-release/convert_imageset" --resize_height=%RESIZE_HEIGHT% --resize_width=%RESIZE_WIDTH% --gray=true %VAL_DATA_ROOT% %EXAMPLE%/pair_data2_val.txt %EXAMPLE%/casia_val_lmdb_60x72_c1_pair2


echo "Done."

pause