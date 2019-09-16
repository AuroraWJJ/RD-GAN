# RD-GAN
##Recursive expression animation synthesis based on dual GAN

### Requirement
-Hardware environment£¨Lenovo server£©£º
[li]Inter(R)Xeon(TM) E5-1650 3.50 GHzs¡Á 2
[li]NVIDIA  1080Ti Graphics Card
[li]32GB DDR4 SDRAM


-Software environment£º
[li]Ubuntu16.04£¨CUDA9.0 ¡¢ CUDNN £©
[li]Pycharm2016.3
[li]Anaconda2£¨Python2.7£©¡¢Tensorflow1.8¡¢OpenCV


### train
Run `train_ck_faceGAN.py` in the pycharm to train the model.
Run `config.py` to change dataset.

### test, predict
Run `real_time_predict.py` to make a predict, and set configuration in `config.py`.

### Description of other documents
1. The path of the files is in `tree.txt`
Different file names correspond to different functional files, you can know the function of the file by the name of the file. For example,`0_MMI_preprocess.py` represents the preprocessing of MMI dataset. The words with 'video' or 'image' represent the processing of video or image.

2. Prime files
We define parameters in `config.py`, such as training times, paths, etc. See the file or the following information for details.
`datasets.py` is a data feeder program file that provides the datasets used for tensorflow training and constructs out-of-order inputs.
`model.py` is a model file that is used to construct the model of the entire network.
`train_ck_facegan.py` is the main program file for training CK+ data.
We make a predict of the input face in `real_time_predict.py`.
All the functions that operate on the files path are in the `path_prcess.py`.
`image_process.py` contains functions related to image processing, such as zooming in and out of images, linear interpolation, etc.

3. Configuration
`config.py` is the configuration file, the specific configuration is as follows:
    SSIM_RATIO = 1           # define parameters for SSIM
    PREDICT_BATCH_SIZE = 64  # the batch size of the predicted data
    BATCH_SIZE = 32          # the batch size of the training data
    REAL_TIME_REENACT_BATCH_SIZE = 1
    NAME_DATA_SET = 'ck+'   # FRGC
    # NAME_DATA_SET = 'MMI'   # FRGC
    TIMES_TO_SAVE = 500
    TRAIN_FLAG = ''          # TRAIN_FLAG = 'train' or 'test'
    TRAIN_ON_SMALL_DATA = False
    SMALL_DATA_SIZE = 64000  # integer multiple of batch_size
    DROP_OUT = 0.6
    CONV_KERNEL_SIZE = [3, 3] 
    TIMES_EPOCH = 500        # number of iterations of training
    CAPACITY = 1000 * BATCH_SIZE
    SHUFFLE = False   
    MAX_ITER = 200000        # maximum number of iterations
    WEIGHT_DECAY = 5e-4
    VALIDE_RATIO = 0.5

    -Hardware setting
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    GPU = '0'
    PER_PROCESS_GPU_MEMORY_FRACTION = 0.9

    -image setting
    EDGE_THICK = 2
    NUM_THREADS = 10
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    CHANNEL = 3