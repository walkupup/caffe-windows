set TOOLS=../../../bin-gpu-release

"%TOOLS%/caffe" train --solver=./mysiamese_v3_solver.prototxt --gpu=all 
"%TOOLS%/caffe" train --solver=./mysiamese_v3_solver2.prototxt --gpu=all --snapshot=./mysiamese_iter_1000000.solverstate

pause