# RD-GAN
Recursive expression animation synthesis based on dual GAN

### ����Ҫ��
####Ӳ�����������빤��վ����
[li]Ӣ�ض���ǿϵ�д����� Inter(R)Xeon(TM) E5-1650 3.50 GHzs�� 2
[li]NVIDIA  1080Ti �Կ�
[li]32GB ����ͨ���ڴ�


####���������
[li]Ubuntu16.04 ϵͳ ��CUDA9.0 �� CUDNN ��
[li]Pycharm2016.3
[li]Anaconda2��Python2.7����Tensorflow1.8��OpenCV


### ѵ��
��pycharm������
train_ck_faceGAN.py �������ݼ�ʹ��config.py����Ĵ���

### ���ԡ�Ԥ��
��Ԥ��ʹ��real_time_predict.py�Ĵ��룬ͬ����config�����������

### �����ļ�����
1���ļ��ṹ�����
��ͬ���ļ����ƶ�Ӧ��ͬ�Ĺ����ļ����������Լ���֪�⡣
�确0_MMI_preprocess.py����ʾ��MMI���ݼ���Ԥ����
���С�video����image����ʾ����Ƶ��ͼ��Ĵ���
SSIM��Ϊ�ṹ��������ʧ���������еĲ����ļ��������ssim�˽ⲻ����Խ��ѧϰ

2����Ҫ�ļ�
config.py��ʾ���еĲ�������ĵط�������������ѵ��������·���Ȳ�������ϸ���ݼ��ļ�����������Ϣ
datasets.py ��ʾ���������������Ҫ���ݹ����˵ĳ����ļ��������ṩtensorflowѵ����
ʹ�õ����ݼ������������������
model.pyΪģ���ļ���������ļ������ڹ�������������ģ��
train_ck_faceGAN.py ��ʾ�������ļ�������ѵ��ck+������
real_time_predict.py ��ʾ���������������Ԥ��
path_prcess.py ��ʾ·����������к���������·�����������к���ȫ������
image_process.py ��ʾͼ��������к���������ͼ������д�����Ŵ���С�����Բ�ֵ��ȫ��������

3������
config.py ��ʾ�����ļ��������������£�
	SSIM_RATIO = 1           # ����SSIM�Ĳ���
    PREDICT_BATCH_SIZE = 64  # Ԥ�����ݵ���ͼ�����
    BATCH_SIZE = 32          # ÿ��ѵ������ͼ�����
    REAL_TIME_REENACT_BATCH_SIZE = 1
    NAME_DATA_SET = 'ck+'   # FRGC
    # NAME_DATA_SET = 'MMI'   # FRGC
    TIMES_TO_SAVE = 500
    TRAIN_FLAG = ''          # ����ط�����ѵ����train���Ͳ��ԣ�test��
    TRAIN_ON_SMALL_DATA = False
    SMALL_DATA_SIZE = 64000  # ��Ҫʱbatch_size��������
    DROP_OUT = 0.6           # �����ı���
    CONV_KERNEL_SIZE = [3, 3]  # ����˵Ĵ�С
    TIMES_EPOCH = 500        # ѵ�������Ĵ���
    CAPACITY = 1000 * BATCH_SIZE
    SHUFFLE = False        # �Ƿ�����
    MAX_ITER = 200000        # ��������
    WEIGHT_DECAY = 5e-4     # Ȩ���½���
    VALIDE_RATIO = 0.5      # ��֤����С

    # Ӳ�����ò���
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    GPU = '0'
    PER_PROCESS_GPU_MEMORY_FRACTION = 0.9  # ʹ��GPU�ı���

    # ͼ�񲿷�
    EDGE_THICK = 2          # ���ɱ�Եͼ���������ϸ
    NUM_THREADS = 10        # �߳���

    IMAGE_WIDTH = 128       # ͼƬ���
    IMAGE_HEIGHT = 128      # ͼƬ�߶�
    CHANNEL = 3             # ͼ���ͨ����


4��ϵͳĿ¼���ṹ
��ǰ�ļ���Ŀ¼�ṹ���������õ��ļ�ע��
����0-target-of-me
����datasets  // ���ݼ�·��
��  ����bos 
��  ��  ����bos_img
��  ��  ����bos_lm2
��  ����ck+
��  ��  ����img_FaceGAN_reenact  // ���ɵı���·��
��  ��  ����img_FaceGAN_reenact-old
��  ��  ����img_FaceGAN_reenact-s014
��  ��  ����img_FaceGAN_reenact_without_chin
��  ��  ����img_pair  // Ԥ��������ͼ���
��  ��  ����img_pair_without_chin 
��  ��  ����original  // ���ݼ�ԭʼͼ��
��  ����face
��  ����face2
��  ����frgc
��  ��  ����img_concat
��  ��  ����img_pair
��  ��  ����small_128_128_2
��  ��  ����small_64_64_1
��  ��  ����small_64_64_2
��  ����MMI
��  ��  ����img_pair
��  ��  ����original
��  ����source
����dlib_param
����faceGAN_bck
����fineGAN_train_log
��  ����logs
��  ����model
��  ����sample_img
����test_code
��  ����mind1_lines
��  ����pts
��  ����test
��  ����tf_data
��  ����video
����test_output
��  ����datasets
��  ����model
����train_log  // ѵ����¼��
��  ����logs    // ѵ����־
��  ����model   //ģ�ʹ洢λ��
��  ����sample_img  // ѵ�����Ч��ͼ
����train_log-
��  ����ģ������
��  ��  ����model
��  ��  ����sample_img
��  �����ϰ汾
��      ����logs
��      ����logs-
��      ����logs---ck+
��      ����logs-64_64_1-frgc
��      ����model
��      ����model-
��      ����model---ck+
��      ����model-64_64_1-frgc
��      ����sample_img
��      ����sample_img-
��      ��  ����old
��      ��  ����old-encoder��Ϊvgg
��      ��  ����ϸ����-ԭ����-64_64_1-frgc
��      ����sample_img---ck+
����train_log_ck+_withchin
��  ����logs
��  ����model
��  ����sample_img
��  ����Ч��ͼ
����train_log_FineGAN_without_chin
��  ����logs
��  ����model
��  ����sample_img
��  ����sample_img--
��  ����sample_img---
����train_log_without_chin
��  ����logs
��  ����model
��  ����sample_img
��  ����sample_img-200000
��  ��������
����__pycache__
��������
�����½��ļ���
��  ����__pycache__
����ģ���ļ���Ԥ��Ч��ͼ��
    ��������ģ��--connection--�ڰ�
    ��  ����img_FaceGAN_reenact
    ��  ����model
    ��  ����sample_img
    ����˫��ģ��--connection
        ����img_FaceGAN_reenact
        ����img_FaceGAN_reenact
        ����img_FaceGAN_reenact-old
        ����img_FaceGAN_reenact-out1
        ����model
        ����sample_img
