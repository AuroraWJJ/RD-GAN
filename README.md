# RD-GAN
## Recursive expression animation synthesis based on dual GAN

### Requirement
#### Hardware environment(Lenovo server)£º  
[li] Inter(R)Xeon(TM) E5-1650 3.50 GHzs¡Á 2  
[li] NVIDIA  1080Ti Graphics Card  
[li] 2GB DDR4 SDRAM  


#### Software environment:  
[li] Ubuntu16.04(CUDA9.0, CUDNN)  
[li] Pycharm2016.3  
[li] Anaconda2(Python2.7), Tensorflow1.8, OpenCV  


### train
Run `train_ck_faceGAN.py` in the pycharm to train the model.  
Run `config.py` to change dataset.

### test, predict
Run `real_time_predict.py` to make a predict, and set configuration in `config.py`.

### Description of other documents
1. The path of the files is at the end.  
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
    PREDICT_BATCH_SIZE = 64       # the batch size of the predicted data  
    BATCH_SIZE = 32          # the batch size of the training data  
    REAL_TIME_REENACT_BATCH_SIZE = 1  
    NAME_DATA_SET = 'ck+'      # FRGC£»or NAME_DATA_SET = 'MMI' , FRGC  
    TIMES_TO_SAVE = 500  
    TRAIN_FLAG = ''          # TRAIN_FLAG = 'train' or 'test'  
    TRAIN_ON_SMALL_DATA = False  
    SMALL_DATA_SIZE = 64000       # integer multiple of batch_size  
    DROP_OUT = 0.6  
    CONV_KERNEL_SIZE = [3, 3]   
    TIMES_EPOCH = 500          # number of iterations of training  
    CAPACITY = 1000 * BATCH_SIZE  
    SHUFFLE = False  
    MAX_ITER = 200000          # maximum number of iterations  
    WEIGHT_DECAY = 5e-4  
    VALIDE_RATIO = 0.5  

    *Hardware setting*  
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
    GPU = '0'  
    PER_PROCESS_GPU_MEMORY_FRACTION = 0.9  

    *image setting*  
    EDGE_THICK = 2  
    NUM_THREADS = 10  
    IMAGE_WIDTH = 128  
    IMAGE_HEIGHT = 128  
    CHANNEL = 3  

4. The directory tree structure is as follows:  
D:\0-RYAN\4-PYTHON\FACEGAN
```
©À©¤0-target-of-me
©À©¤datasets //the path of datasets
©¦  ©À©¤bos
©¦  ©¦  ©À©¤bos_img
©¦  ©¦  ©¸©¤bos_lm2
©¦  ©À©¤ck+
©¦  ©¦  ©À©¤img_FaceGAN_reenact //the path of generation
©¦  ©¦  ©À©¤img_FaceGAN_reenact-old
©¦  ©¦  ©À©¤img_FaceGAN_reenact-s014
©¦  ©¦  ©À©¤img_FaceGAN_reenact_without_chin
©¦  ©¦  ©À©¤img_pair //preprocessed paired images
©¦  ©¦  ©À©¤img_pair_without_chin
©¦  ©¦  ©¸©¤original //the original images of dataset
©¦  ©À©¤face
©¦  ©À©¤face2
©¦  ©À©¤frgc
©¦  ©¦  ©À©¤img_concat
©¦  ©¦  ©À©¤img_pair
©¦  ©¦  ©À©¤small_128_128_2
©¦  ©¦  ©À©¤small_64_64_1
©¦  ©¦  ©¸©¤small_64_64_2
©¦  ©À©¤MMI
©¦  ©¦  ©À©¤img_pair
©¦  ©¦  ©¸©¤original
©¦  ©¸©¤source
©À©¤dlib_param
©À©¤faceGAN_bck
©À©¤fineGAN_train_log
©¦  ©À©¤logs
©¦  ©À©¤model
©¦  ©¸©¤sample_img
©À©¤test_code
©¦  ©À©¤mind1_lines
©¦  ©À©¤pts
©¦  ©À©¤test
©¦  ©À©¤tf_data
©¦  ©¸©¤video
©À©¤test_output
©¦  ©À©¤datasets
©¦  ©¸©¤model
©À©¤train_log //train
©¦  ©À©¤logs
©¦  ©À©¤model //the path for the model
©¦  ©¸©¤sample_img  //randomly selected result images
©À©¤train_log-
©¦  ©À©¤blur
©¦  ©¦  ©À©¤model
©¦  ©¦  ©¸©¤sample_img
©¦  ©¸©¤old version
©¦      ©À©¤logs
©¦      ©À©¤logs-
©¦      ©À©¤logs---ck+
©¦      ©À©¤logs-64_64_1-frgc
©¦      ©À©¤model
©¦      ©À©¤model-
©¦      ©À©¤model---ck+
©¦      ©À©¤model-64_64_1-frgc
©¦      ©À©¤sample_img
©¦      ©À©¤sample_img-
©¦      ©¦  ©À©¤old
©¦      ©¦  ©À©¤old-encoder --> vgg
©¦      ©¦  ©¸©¤thin line-the original network-64_64_1-frgc
©¦      ©¸©¤sample_img---ck+
©À©¤train_log_ck+_withchin
©¦  ©À©¤logs
©¦  ©À©¤model
©¦  ©À©¤sample_img
©¦  ©¸©¤result image
©À©¤train_log_FineGAN_without_chin
©¦  ©À©¤logs
©¦  ©À©¤model
©¦  ©À©¤sample_img
©¦  ©À©¤sample_img--
©¦  ©¸©¤sample_img---
©À©¤train_log_without_chin
©¦  ©À©¤logs
©¦  ©À©¤model
©¦  ©À©¤sample_img
©¦  ©À©¤sample_img-200000
©¦  ©¸©¤code
©À©¤__pycache__
©À©¤code
©À©¤new folder
©¦  ©¸©¤__pycache__
©¸©¤model and results
    ©À©¤one-way model--connection--monochrome image
    ©¦  ©À©¤img_FaceGAN_reenact
    ©¦  ©À©¤model
    ©¦  ©¸©¤sample_img
    ©¸©¤two-way model--connection
        ©À©¤img_FaceGAN_reenact--bad results2
        ©À©¤img_FaceGAN_reenact--bad results
        ©À©¤img_FaceGAN_reenact-old
        ©À©¤img_FaceGAN_reenact-out1
        ©À©¤model
        ©¸©¤sample_img
```