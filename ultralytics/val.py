import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
 
# 训练好的模型配置文件，以相对路径调用，也可以使用绝对路径
model_yaml_path = './runs/RT-DETR/train/exp/weights/best.pt'  #这行使用训练和的best.pt文件
 
#数据集配置文件，以相对路径调用，也可以使用绝对路径
data_yaml_path = './datasets/data.yaml'
if __name__ == '__main__':
    model = RTDETR(model_yaml_path)
    model.val(data=data_yaml_path,
              split='val',
              imgsz=640,
              batch=4,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/RT-DETR/val',
              name='exp',
              )