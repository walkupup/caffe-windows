

set DATA=..\..\data\mnist

echo "Creating leveldb..."

del .\mnist_siamese_train_leveldb
del .\mnist_siamese_test_leveldb

"..\..\bin-gpu-release\convert_mnist_siamese_data" %DATA%\train-images.idx3-ubyte %DATA%\train-labels.idx1-ubyte .\mnist_siamese_train_leveldb
"..\..\bin-gpu-release\convert_mnist_siamese_data" %DATA%\t10k-images.idx3-ubyte %DATA%\t10k-labels.idx1-ubyte .\mnist_siamese_test_leveldb

echo "Done."

pause
