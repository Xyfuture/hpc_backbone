import time

from models.big_lama.model.lama import LaMa
from models.big_lama.model.config import LaMaConfig
import cv2
import numpy as np


config = LaMaConfig()

model = LaMa(config)

img_path = r"C:\Users\xyfuture\Pictures\origin.png"
mask_path = r"C:\Users\xyfuture\Pictures\mask.png"

img = cv2.imread(img_path)
mask = cv2.imread(mask_path)

first_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
first_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

second_img = first_img[100:,100:]
second_mask = first_mask[100:,100:]

infer_images = [first_img for i in range(2)]
infer_masks = [first_mask for j in range(2)]


st = time.time()
# for i in range(5):
out_img = model(infer_images,infer_masks)
print("latency:{}".format(time.time()-st))



# for i,out in enumerate(out_img):
#     out = out.astype(np.uint8)
#     # out = cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
#     cv2.imwrite(f'{i}.jpg', out)


