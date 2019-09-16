# RD-GAN
Recursive expression animation synthesis based on dual GAN

### 配置要求
####硬件环境（联想工作站）：
[li]英特尔至强系列处理器 Inter(R)Xeon(TM) E5-1650 3.50 GHzs× 2
[li]NVIDIA  1080Ti 显卡
[li]32GB 的四通道内存


####软件环境：
[li]Ubuntu16.04 系统 （CUDA9.0 、 CUDNN ）
[li]Pycharm2016.3
[li]Anaconda2（Python2.7）、Tensorflow1.8、OpenCV


### 训练
在pycharm中运行
train_ck_faceGAN.py 更换数据集使用config.py里面的代码

### 测试、预测
做预测使用real_time_predict.py的代码，同样在config里面进行配置

### 其他文件解释
1、文件结构在最后
不同的文件名称对应不同的功能文件，但都可以见名知意。
如‘0_MMI_preprocess.py’表示对MMI数据集的预处理
带有‘video’‘image’表示对视频和图像的处理
SSIM的为结构相似性损失构建过程中的测试文件，如果对ssim了解不深，可以借鉴学习

2、主要文件
config.py表示所有的参数定义的地方，定义了诸如训练次数、路径等参数，详细内容见文件或者下述信息
datasets.py 表示整个程序的运行主要数据供给端的程序文件，用于提供tensorflow训练所
使用的数据集、构造无序的输入流
model.py为模型文件，在这个文件中用于构造整个函数的模型
train_ck_faceGAN.py 表示主程序文件，用于训练ck+的数据
real_time_predict.py 表示对输入的人脸进行预测
path_prcess.py 表示路径处理的所有函数，即对路径操作的所有函数全在这里
image_process.py 表示图像处理的所有函数，即对图像的所有处理如放大缩小、线性插值等全部在这里

3、配置
config.py 表示配置文件，具体配置如下：
	SSIM_RATIO = 1           # 定义SSIM的参数
    PREDICT_BATCH_SIZE = 64  # 预测数据的批图像个数
    BATCH_SIZE = 32          # 每次训练的批图像个数
    REAL_TIME_REENACT_BATCH_SIZE = 1
    NAME_DATA_SET = 'ck+'   # FRGC
    # NAME_DATA_SET = 'MMI'   # FRGC
    TIMES_TO_SAVE = 500
    TRAIN_FLAG = ''          # 这个地方更改训练（train）和测试（test）
    TRAIN_ON_SMALL_DATA = False
    SMALL_DATA_SIZE = 64000  # 需要时batch_size的整数倍
    DROP_OUT = 0.6           # 保留的比例
    CONV_KERNEL_SIZE = [3, 3]  # 卷积核的大小
    TIMES_EPOCH = 500        # 训练迭代的次数
    CAPACITY = 1000 * BATCH_SIZE
    SHUFFLE = False        # 是否乱序
    MAX_ITER = 200000        # 迭代次数
    WEIGHT_DECAY = 5e-4     # 权重下降率
    VALIDE_RATIO = 0.5      # 验证集大小

    # 硬件设置部分
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    GPU = '0'
    PER_PROCESS_GPU_MEMORY_FRACTION = 0.9  # 使用GPU的比例

    # 图像部分
    EDGE_THICK = 2          # 生成边缘图像的线条粗细
    NUM_THREADS = 10        # 线程数

    IMAGE_WIDTH = 128       # 图片宽度
    IMAGE_HEIGHT = 128      # 图片高度
    CHANNEL = 3             # 图像的通道数


4、系统目录树结构
当前文件夹目录结构及部分有用的文件注释
├─0-target-of-me
├─datasets  // 数据集路径
│  ├─bos 
│  │  ├─bos_img
│  │  └─bos_lm2
│  ├─ck+
│  │  ├─img_FaceGAN_reenact  // 生成的表演路径
│  │  ├─img_FaceGAN_reenact-old
│  │  ├─img_FaceGAN_reenact-s014
│  │  ├─img_FaceGAN_reenact_without_chin
│  │  ├─img_pair  // 预处理过后的图像对
│  │  ├─img_pair_without_chin 
│  │  └─original  // 数据集原始图像
│  ├─face
│  ├─face2
│  ├─frgc
│  │  ├─img_concat
│  │  ├─img_pair
│  │  ├─small_128_128_2
│  │  ├─small_64_64_1
│  │  └─small_64_64_2
│  ├─MMI
│  │  ├─img_pair
│  │  └─original
│  └─source
├─dlib_param
├─faceGAN_bck
├─fineGAN_train_log
│  ├─logs
│  ├─model
│  └─sample_img
├─test_code
│  ├─mind1_lines
│  ├─pts
│  ├─test
│  ├─tf_data
│  └─video
├─test_output
│  ├─datasets
│  └─model
├─train_log  // 训练记录，
│  ├─logs    // 训练日志
│  ├─model   //模型存储位置
│  └─sample_img  // 训练随机效果图
├─train_log-
│  ├─模糊不清
│  │  ├─model
│  │  └─sample_img
│  └─老版本
│      ├─logs
│      ├─logs-
│      ├─logs---ck+
│      ├─logs-64_64_1-frgc
│      ├─model
│      ├─model-
│      ├─model---ck+
│      ├─model-64_64_1-frgc
│      ├─sample_img
│      ├─sample_img-
│      │  ├─old
│      │  ├─old-encoder改为vgg
│      │  └─细线条-原网络-64_64_1-frgc
│      └─sample_img---ck+
├─train_log_ck+_withchin
│  ├─logs
│  ├─model
│  ├─sample_img
│  └─效果图
├─train_log_FineGAN_without_chin
│  ├─logs
│  ├─model
│  ├─sample_img
│  ├─sample_img--
│  └─sample_img---
├─train_log_without_chin
│  ├─logs
│  ├─model
│  ├─sample_img
│  ├─sample_img-200000
│  └─代码
├─__pycache__
├─代码
├─新建文件夹
│  └─__pycache__
└─模型文件和预测效果图像
    ├─单向模型--connection--黑白
    │  ├─img_FaceGAN_reenact
    │  ├─model
    │  └─sample_img
    └─双向模型--connection
        ├─img_FaceGAN_reenact
        ├─img_FaceGAN_reenact
        ├─img_FaceGAN_reenact-old
        ├─img_FaceGAN_reenact-out1
        ├─model
        └─sample_img
