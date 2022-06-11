import paddlex as pdx
from paddlex import transforms as T
import os
model = pdx.load_model(r'C:\Users\boyif\Desktop\paddle\Phone_det\RCNN\best_model')  # 加载训练好的模型

eval_transforms = T.Compose([
    T.Resize(
        target_size=320, interp='CUBIC'),
    T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
eval_dataset = pdx.datasets.VOCDetection(
    data_dir=r'C:\Users\boyif\Desktop\paddle\Phone_det',
    file_list=r'C:\Users\boyif\Desktop\paddle\Phone_det\\test_list.txt',
    label_list=r"C:\Users\boyif\Desktop\paddle\Phone_det\\labels.txt",
    transforms=eval_transforms)
print(model.evaluate(eval_dataset))

pathDir = os.listdir(r"C:\Users\boyif\Desktop\paddle\Phone_det")
count = 0
for path in pathDir:
    count = count+1
    image_name = os.path.join(r"C:\Users\boyif\Desktop\paddle\Phone_det", path)
    result = model.predict(image_name)
    print(result)
    pdx.det.visualize(image_name, result, threshold=0.2, save_dir='./output/predict')  # 将预测好的图片标记并输出到指定目录，threhold代表置信度低于0.6的不进行输出
    if count >= 100:
        break
