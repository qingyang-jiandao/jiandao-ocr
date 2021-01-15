# jiandao-ocr
OCR with the caffe  framework on windows and linux
## Training:

1. Download The MSCOCO dataset [MSCOCO dataset](https://cocodataset.org/#download) as background image:
    * http://images.cocodataset.org/zips/train2014.zip
    * http://images.cocodataset.org/zips/val2014.zip
    * http://images.cocodataset.org/zips/test2014.zip
    
    1.1  unpack zip to `coco-dataset-dir`

2. Download and install Python for Windows: https://www.python.org/ftp/python/3.5.2/python-3.5.2-amd64.exe

3. Run command: `python synthetic_chinese_dataset\script\synthesize_data_6862.py` (to synthesize train/test data: images, annotations, segObjects).
    - Reset `args['background_image_dir']` to `coco-dataset-dir`
    - If required reset the args of `class OCRData`

4. Run command: `python generate_data_list.py` (to generate files: images.list, annotations.list, segObjects.list, samples.list).
    - If required reset these paths: `image_dir=images dir, label_dir=annotations dir, mask_dir=segObjects dir`

5. Run command: `generate_dataset_h5.py` (to generate files: training.list, testing.list).
    - If required change paths: `imgPath, labelPath, maskPath`
    - Set `--dataPath`. like this: `python generate_dataset_h5.py --dataPath h5_space`

6. Start training by using `train.sh` or `train_snapshot.sh` or `python train_net.py` on linux: 

7. Start training by using `train.bat` or `python train_net.py` on windows: 

8. Reset `net:` path in the solver.prototxt: 
    8.1  To fuzzy text: train-stnpt-fnn-vgg-pro-stnfeature-64x256.prototxt has better results.
    8.2  To text with perspective transformation: train-stnpt-fnn-lossweight-vgg-64x256.prototxt has better results.	
    8.2  Train-stnpt-segnet-vgg-attention-lstmnode-64x256.prototxt use attention mechanism to decode sequence.
	
 **Note:** In order to achieve better results, in addition to the synthetic data, part of the labeled data is required.

## Testing 

    The model will be tested on the testing set at the end of training.

## Demo

1.  Run `demo_stn.py` to recognize the text by train-stnpt-fnn-lossweight-vgg-64x256.prototxt.
    Note:
    - If required change paths: `wight_file, deploy_file`
    - Set `test_img` to input image path

2.  Run `demo_jiandao.py` to recognize the text by train-stnpt-fnn-vgg-pro-stnfeature-64x256.prototxt.
    Note:
    - If required change paths: `wight_file, deploy_file`
    - Set `test_img` to input image path

## c++ for model deployment on windows and linux

    the source is in the cpp/uselib directory.

## Jiandao Caffe

Jiandao Caffe is a modified version of the popular [Caffe Deep Learning framework](http://caffe.berkeleyvision.org/) adapted for use with DesignWare EV6x Processors.
It combines multiple customized branches and includes a large range of patches to support diverse models. 

### Installation
Please check out the prerequisites and read the detailed notes at the [BVLC Caffe Installation](http://caffe.berkeleyvision.org/installation.html) if this is your first time to install Caffe.

#### Linux
A simple guide:
1. Ensure that you have all the dependencies mentioned at the [BVLC Caffe Installation](http://caffe.berkeleyvision.org/installation.html) for your OS installed (protobuf, leveldb, snappy, opencv, hdf5-serial, protobuf-compiler, BLAS, Python, CUDA etc.)
2. Checkout the Jiandao Caffe **master** branch. Configure the build by copying and modifying the example Makefile.config for your setup.
```Shell
git clone https://github.com/qingyang-jiandao/jiandao-caffe.git
cd jiandao-caffe
cp Makefile.config.example Makefile.config
# Modify Makefile.config to suit your needs, e.g. enable/disable the CPU-ONLY, CUDNN, NCCL and set the path for CUDA, Python and BLAS.
# If needed, add [your installed matio path]/include to INCLUDE_DIRS and [your installed matio path]/lib to LIBRARY_DIRS.
```
3. Build Caffe and run the tests.
```Shell
make -j4 && make pycaffe
```

#### Windows
A simple guide:
1. Download the **Visual Studio 2015** (or VS 2017). Choose to install the support for visual C++ instead of applying the default settings.
2. Install the CMake 3.4 or higher. Install Python 2.7 or 3.5/3.6. Add cmake.exe and python.exe to your PATH.
3. After installing the Python, please open a `cmd` prompt and use `pip install numpy` to install the **numpy** package.
4. Checkout the Jiandao Caffe **master** branch for build. The windows branch is deprecated, please do not use it. We use `C:\Projects` as the current folder for the following instructions.
5. Edit any of the options inside **jiandao-caffe\scripts\build_win.cmd** to suit your needs, such as settings for Python version, CUDA/CuDNN enabling etc.   
```cmd
C:\Projects> git clone https://github.com/qingyang-jiandao/jiandao-caffe.git
C:\Projects> cd jiandao-caffe
C:\Projects\jiandao-caffe> build_win.cmd or build_win_vs2015.cmd
:: If no error occurs, the caffe.exe will be created at C:\Projects\jiandao-caffe\build\tools\Release after a successful build.
```
Other detailed installation instructions can be found [here](https://github.com/BVLC/caffe/blob/windows/README.md).
