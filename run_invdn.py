import numpy as np

from models.inv_dn.interface import load_invdn_model
import cv2


img_path = r"D:\code\swdesign\hpc_backbone\test\samples\origin.png"
# img_path = r"D:\code\swdesign\InvDN_paddlepaddle\SIDD_mini\val_mini\GT\0_1.PNG"
model = load_invdn_model()

img = cv2.imread(img_path)
# img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
first_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

out = model(first_img)

cv2.imwrite("out.jpg",out)

