import time

from models.big_lama.model.lama import LaMa
from models.big_lama.model.config import Config,HDStrategy
import cv2
import numpy as np


config = Config(
        hd_strategy=HDStrategy.RESIZE,
    hd_strategy_crop_margin = 100,
    hd_strategy_crop_trigger_size = 1000,
    hd_strategy_resize_limit = 500

    # prompt: str = ""
    # negative_prompt: str = ""
    # # 始终是在原图尺度上的值
    # use_croper: bool = False
    # croper_x: int = None
    # croper_y: int = None
    # croper_height: int = None
    # croper_width:
    #  int = None
)

model = LaMa('cpu')

img_path = r"C:\Users\xyfuture\Pictures\origin.png"
mask_path = r"C:\Users\xyfuture\Pictures\mask.png"

img = cv2.imread(img_path)
mask = cv2.imread(mask_path)

first_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
first_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

second_img = first_img[100:,100:]
second_mask = first_mask[100:,100:]

infer_images = [first_img for i in range(16)]
infer_masks = [first_mask for j in range(16)]


st = time.time()
# for i in range(5):
out_img = model(infer_images,infer_masks,config)
print("latency:{}".format(time.time()-st))



# for i,out in enumerate(out_img):
#     out = out.astype(np.uint8)
#     # out = cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
#     cv2.imwrite(f'{i}.jpg', out)


