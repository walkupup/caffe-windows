set TOOLS=../../../bin

"%TOOLS%/caffe" train --solver=./mnist_siamese_solver.prototxt --gpu=all 
"%TOOLS%/caffe" train --solver=./mnist_siamese_solver2.prototxt --gpu=all --snapshot=./siamese_iter_1000000.solverstate

pause