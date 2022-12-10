import asyncio
import pickle

import numpy as np
from loguru import logger
import redis.asyncio as redis
from pydantic import BaseModel
from models.big_lama.interface import load_model
from models.big_lama.model.config import LaMaConfig
import cv2

class LaMaWorkerConfig(BaseModel):
    batch_size: int = 4
    batch_timeout_limit: int = 0

    worker_stream_key:str = "Inpainting_worker"
    worker_group_name:str = "worker"

    ack_stream_key:str = "Inpainting_finish_ack"
    ack_group_name:str = 'master'


def read_image_from_binary(byte_obj,color=cv2.IMREAD_COLOR):
    img = np.frombuffer(byte_obj,dtype='uint8')
    img = cv2.imdecode(img,color)
    return img


def write_image_to_binary(img):
    img = cv2.imencode('.jpg',img)
    assert img[0]
    byte_obj = img[1].tobytes()
    return byte_obj


class LaMaWorker:
    def __init__(self, worker_config: LaMaWorkerConfig=LaMaWorkerConfig(), model_config:LaMaConfig=LaMaConfig()):
        self.worker_config: LaMaWorkerConfig = worker_config
        self.model_config = model_config

        self.model = load_model(self.model_config)

    def test(self,receiver):
        receive_group_name = self.worker_config.worker_group_name
        receive_stream_key = self.worker_config.worker_stream_key
        send_stream_key = self.worker_config.ack_stream_key
        batch_size = self.worker_config.batch_size

        payload =  receiver.xreadgroup(groupname=receive_group_name, consumername='c2', block=0,
                                            count=batch_size, streams={receive_stream_key: '>'})

    async def process(self, receiver: redis.Redis,sender:redis.Redis ):
        receive_group_name = self.worker_config.worker_group_name
        receive_stream_key = self.worker_config.worker_stream_key
        send_stream_key = self.worker_config.ack_stream_key
        batch_size = self.worker_config.batch_size

        logger.info("lama worker running")

        while True:
            payload = await receiver.xreadgroup(groupname=receive_group_name, consumername='c2', block=0,
                                                count=batch_size, streams={receive_stream_key: '>'})
            logger.info("lama worker process new input")
            tags,images,masks,results = [],[],[],[]
            for i,cur in enumerate(payload[0][1]):
                tags.append(cur[0])
                images.append(pickle.loads(cur[1][b'image']))
                masks.append(pickle.loads(cur[1][b'mask']))
            # decode binary file
            images = [read_image_from_binary(image, cv2.IMREAD_COLOR) for image in images]
            images = [cv2.cvtColor(image,cv2.COLOR_BGR2RGB) for image in images]

            masks = [read_image_from_binary(mask, cv2.IMREAD_GRAYSCALE) for mask in masks]

            # test output
            # for image in images:
            #     print('here')
            #     cv2.imwrite('color.jpg',image)


            logger.info("lama worker decode binary images")
            num = len(tags)

            results = self.model(images,masks)
            results = [write_image_to_binary(result) for result in results]
            logger.info('lama worker finish process')

            tasks = [
                sender.xadd(send_stream_key , {'tag': tags[i],'result':pickle.dumps(results[i]) } )
                for i in range(num)
            ]

            await asyncio.gather(*tasks)
            logger.info("lama work send ack back")
            # ack








