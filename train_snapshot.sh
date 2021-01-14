export LD_LIBRARY_PATH=/usr/local/lib:./caffe/ubuntu-lib:/home/ps/jd_workspace/cuda-10.0/lib64::$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
cur_date=`date +%Y-%m-%d-%H-%M-%S`
log_file_name=./log/${cur_date}
/home/ps/jd_workspace/dl/caffe_ws/build/tools/caffe train \
    -solver ./solver.prototxt \
	  -snapshot ./model/stnpt-fnn-vgg-stnfeature-breakthrough_iter_420000.solverstate \
    -gpu 1
