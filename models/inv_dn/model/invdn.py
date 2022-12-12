import os
from paddle import inference
from loguru import logger
import cv2
import numpy as np
from models.inv_dn.model.config import InvDNConfig


class InvDN:
    def __init__(self, device: str = 'cpu', config: InvDNConfig = InvDNConfig()):
        self.device = device
        self.config = config

        self.model,self.infer_config,self.input_tensor,self.output_tensor = self.load_model(
            self.config.model_info_path,self.config.model_weight_path
        )

    def load_model(self, model_info_path, model_weight_path):

        infer_config = inference.Config(model_info_path, model_weight_path)
        if self.device == 'cpu':
            infer_config.disable_gpu()
        else:
            infer_config.enable_use_gpu(1000)

        # optimization config
        infer_config.enable_memory_optim()
        infer_config.disable_glog_info()

        infer_config.switch_use_feed_fetch_ops(False)
        infer_config.switch_ir_optim(True)

        # create predictor
        model = inference.create_predictor(infer_config)

        # static graph for input and output

        input_names = model.get_input_names()
        input_tensor = model.get_input_handle(input_names[0])

        output_names = model.get_output_names()
        output_tensor = model.get_output_handle(output_names[0])

        return model,infer_config,input_tensor,output_tensor

    def pre_process(self,img:np.ndarray):
        """
        Args:
            img: np.ndarray uint8 RGB [H,W,C] imag

        Returns: img np.ndarray float32 [C,H,W] norm
        """
        img = img.astype('float32') / 255.0
        img = img.transpose([2,0,1])

        img = img[np.newaxis,:,:,:] # up dimension

        # 引入随机噪声
        random_noise = np.random.randn(1, 45, img.shape[2], img.shape[3]).astype(np.float32)
        img = np.concatenate([img, random_noise], 1)
        return img

    def post_process(self,img:np.ndarray):
        if len(img.shape) == 4:
            img = img[0]
        img = img.transpose([1,2,0])
        img = (img*255).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return  img

    def run(self,img):
        """
        Args:
            img: np.ndarray [N,C,H,W] float32 norm
        """
        self.input_tensor.copy_from_cpu(img)
        self.model.run()
        output = self.output_tensor.copy_to_cpu()
        return output

    def sigment_run(self,img:np.ndarray):
        n,c,h,w = img.shape
        tmp_input = np.zeros([n,c,256,256],dtype='float32')
        tmp_output = np.zeros([n,3,h,w],dtype='float32')

        for i in range(0,h,256):
            for j in range(0,w,256):
                i_e = 256 if i+256<h else h
                j_e = 256 if j+256<w else w

                tmp_input[:,:,:i_e-i,:j_e-j] = img[:,:,i:i_e,j:j_e]
                tmp_output[:,:,i:i_e,j:j_e] = self.run(tmp_input)[:,:,:i_e-i,:j_e-j]
        return tmp_output

    def __call__(self, img):
        img = self.pre_process(img)
        img = self.sigment_run(img)
        img = self.post_process(img)
        return img