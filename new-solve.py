from paddlex import transforms as T
import paddlex as pdx

train_transforms = T.Compose([
    T.MixupImage(mixup_epoch=-1),
    T.RandomDistort(),
    T.RandomExpand(im_padding_value=[123.675, 116.28, 103.53]),
    T.RandomCrop(),
    T.RandomDistort(),  # 以一定的概率对图像进行随机像素内容变换，可包括亮度、对比度、饱和度、色相角度、通道顺序调整等
    T.RandomHorizontalFlip(),
    T.BatchRandomResize(
        target_sizes=[192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512],
        interp='RANDOM'),
    T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.Resize(
        target_size=320, interp='CUBIC'),
    T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = pdx.datasets.VOCDetection(
    data_dir=r'C:\Users\boyif\Desktop\paddle\phone\train\0_phone',                    # 数据集目录
    file_list=r"C:\Users\boyif\Desktop\paddle\phone\train\0_phone\train_list.txt",    # 训练集文件路径
    label_list=r"C:\Users\boyif\Desktop\paddle\phone\train\0_phone\labels.txt",       # 数据标签
    transforms=train_transforms,           # 训练集的数据增强策略
    shuffle=True)                          # 打乱数据集的顺序
eval_dataset = pdx.datasets.VOCDetection(
    data_dir=r'C:\Users\boyif\Desktop\paddle\phone\train\0_phone',
    file_list=r'C:\Users\boyif\Desktop\paddle\phone\train\0_phone\val_list.txt',
    label_list=r"C:\Users\boyif\Desktop\paddle\phone\train\0_phone\labels.txt",
    transforms=eval_transforms)

# 获取标签总数
num_classes = len(train_dataset.labels)


# 初始化模型
model = pdx.det.FasterRCNN(num_classes=num_classes, backbone='ResNet50')
###  模型的参数说明   ###
# num_classes (int): 类别数。默认为80。
# backbone (str): YOLOv3的backbone网络，取值范围为['MobileNetV3']。默认为'MobileNetV3'。
# anchors (list|tuple): anchor框的宽度和高度。默认为[[10, 15], [24, 36], [72, 42], [35, 87], [102, 96], [60, 170], [220, 125], [128, 222], [264, 266]]。
# anchor_masks (list|tuple): 在计算YOLOv3损失时，使用anchor的mask索引。默认为[[6, 7, 8], [3, 4, 5], [0, 1, 2]]。
# use_iou_aware (bool): 是否使用IoU Aware分支。默认为False。
# use_spp (bool): 是否使用Spatial Pyramid Pooling结构。默认为True。
# use_drop_block (bool): 是否使用Drop Block。默认为True。
# scale_x_y (float): 调整中心点位置时的系数因子。默认为1.05。
# ignore_threshold (float): 在计算PPYOLOv2损失时，IoU大于ignore_threshold的预测框的置信度被忽略。默认为0.5。
# label_smooth (bool): 是否使用label smooth。默认为False。
# use_iou_loss (bool): 是否使用IoU loss。默认为True。
# use_matrix_nms (bool): 是否使用Matrix NMS。默认为False。
# nms_score_threshold (float): 检测框的置信度得分阈值，置信度得分低于阈值的框应该被忽略。默认为0.005。
# nms_topk (int): 进行NMS时，根据置信度保留的最大检测框数。如果为-1则全部保留。默认为1000。
# nms_keep_topk (int): 进行NMS后，每个图像要保留的总检测框数。默认为100。
# nms_iou_threshold (float): 进行NMS时，用于剔除检测框IOU的阈值。默认为0.45。

# 启动模型训练
model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=6,
    eval_dataset=eval_dataset,
    pretrain_weights=None,
    #pretrain_weights="COCO",
    learning_rate=0.005 / 12,
    warmup_steps=1000,
    warmup_start_lr=0.0,
    lr_decay_epochs=[105, 135, 150],
    save_interval_epochs=5,
    early_stop=True,
    early_stop_patience=10,
    save_dir='output/FasterRCNN',
    resume_checkpoint=r"C:\Users\boyif\Desktop\paddle\phone\output\FasterRCNN\best_model"
    )

####  模型训练参数说明  #####
# num_epochs (int): 训练迭代轮数。
# train_dataset (paddlex.dataset): 训练数据集。
# train_batch_size (int): 训练数据batch大小，默认为64。目前检测仅支持单卡batch大小为1进行评估，train_batch_size参数不影响评估时的batch大小。
# eval_dataset (paddlex.dataset or None): 评估数据集。当该参数为None时，训练过程中不会进行模型评估。默认为None。
# optimizer (paddle.optimizer.Optimizer): 优化器。当该参数为None时，使用默认优化器：paddle.optimizer.lr.PiecewiseDecay衰减策略，paddle.optimizer.Momentum优化方法。
# save_interval_epochs (int): 模型保存间隔（单位：迭代轮数）。默认为1。
# log_interval_steps (int): 训练日志输出间隔（单位：迭代次数）。默认为10。
# save_dir (str): 模型保存路径。默认为'output'。
# pretrain_weights (str ort None): 若指定为'.pdparams'文件时，则从文件加载模型权重；若为字符串’IMAGENET’，则自动下载在ImageNet图片数据上预训练的模型权重（仅包含backbone网络）；若为字符串’COCO’，则自动下载在COCO数据集上预训练的模型权重；若为None，则不使用预训练模型。默认为'IMAGENET'。
# learning_rate (float): 默认优化器的学习率。默认为0.001。
# warmup_steps (int): 默认优化器进行warmup过程的步数。默认为0。
# warmup_start_lr (int): 默认优化器warmup的起始学习率。默认为0.0。
# lr_decay_epochs (list): 默认优化器的学习率衰减轮数。默认为[216, 243]。
# lr_decay_gamma (float): 默认优化器的学习率衰减率。默认为0.1。
# metric ({'COCO', 'VOC', None}): 训练过程中评估的方式。默认为None，根据用户传入的Dataset自动选择，如为VOCDetection，则metric为'VOC'；如为COCODetection，则metric为'COCO'。
# use_ema (bool): 是否使用指数衰减计算参数的滑动平均值。默认为False。
# early_stop (bool): 是否使用提前终止训练策略。默认为False。
# early_stop_patience (int): 当使用提前终止训练策略时，如果验证集精度在early_stop_patience个epoch内连续下降或持平，则终止训练。默认为5。
# use_vdl (bool): 是否使用VisualDL进行可视化。默认为True。
# resume_checkpoint (str): 恢复训练时指定上次训练保存的模型路径，例如output/ppyolov2/best_model。若为None，则不会恢复训练。默认值为None。