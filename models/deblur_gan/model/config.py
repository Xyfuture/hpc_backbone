from pydantic import BaseModel


class DeBlurConfig(BaseModel):
    norm_layer:str = 'instance'
    weight_path = r'D:\code\swdesign\hpc_backbone\models\deblur_gan\weights\fpn_inception.h5'
