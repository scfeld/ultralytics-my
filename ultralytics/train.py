import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
 
# 模型配置文件，以相对路径调用，也可以使用绝对路径
model_yaml_path = './ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml'
#数据集配置文件，以相对路径调用，也可以使用绝对路径
data_yaml_path = './datasets/data.yaml'
if __name__ == '__main__':
    model = RTDETR(model_yaml_path)
    # model.load('rtdetr-l.pt') # 加载预训练权重
    #训练模型
    results = model.train(data=data_yaml_path,
                          imgsz=640,
                          epochs=200,
                          batch=8,
                          workers=0,
                          project='runs/RT-DETR/train',
                          name='exp',
                          mpdiou_alpha=0.7
                          )