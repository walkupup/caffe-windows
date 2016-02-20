set TOOLS=../../../bin

"%TOOLS%/caffe" train --solver=./lenet_solver.prototxt --gpu=all 
"%TOOLS%/caffe" train --solver=./lenet_solver2.prototxt --gpu=all --snapshot=./MyDeepID_gray_iter_2000000.solverstate

pause