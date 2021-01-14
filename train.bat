set BUILD=D:\tgc\workspace\deeplab\caffe-deeplab\build_caffe_win64\tools\Release\
set EXAMPLE=.\

%BUILD%caffe.exe  train --solver=%EXAMPLE%solver.prototxt
::%BUILD%caffe.exe  train --solver=%EXAMPLE%solver.prototxt --snapshot=%EXAMPLE%model/stn_pt-fnn_new_iter_50000.solverstate

::python solve.py

pause
