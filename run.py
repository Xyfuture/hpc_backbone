from models.big_lama.model.lama import LaMa
from models.big_lama.model.config import Config,HDStrategy
import cv2
import numpy as np

model = LaMa('cpu')

config = Config(
        hd_strategy=HDStrategy.CROP,
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

img_path = r"C:\Users\xyfuture\Pictures\origin.png"
mask_path = r"C:\Users\xyfuture\Pictures\mask.png"


img = cv2.imread(img_path)
mask = cv2.imread(mask_path)


img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
# mask = 255 - mask

# mask = np.zeros((1080,1920),dtype=np.uint8)
# mask[100:300,100:300] = 255
# plt.imshow(mask)
# plt.show()



# cv2.imshow('test',mask)
# cv2.waitKey()

out = model(img,mask,config)

out = out.astype(np.uint8)

# out = cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
cv2.imwrite('out.jpg',out)

