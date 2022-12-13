import asyncio
import pickle

import numpy as np
from loguru import logger
import redis.asyncio as redis
from pydantic import BaseModel
from models.deblur_gan.interface import load_deblur_model
from models.deblur_gan.model.config import DeBlurConfig
import cv2

from serve.worker.utils import read_image_from_binary


class DeBlurWorkerConfig(BaseModel):
    device:str = 'cpu'

    # batch_size: int = 4
    # batch_timeout_limit: int = 0

    worker_stream_key: str = "Deblur_worker"
    worker_group_name: str = "worker"

    ack_stream_key: str = "Deblur_finish_ack"
    ack_group_name: str = 'master'


class DeBlurWorker:
    def __init__(self,worker_config:DeBlurWorkerConfig=DeBlurWorkerConfig(),
                 model_config:DeBlurConfig=DeBlurConfig()):
        self.worker_config = worker_config
        self.model_config = model_config

        self.model = load_deblur_model(self.worker_config.device,self.model_config)

    async def process(self, receiver: redis.Redis, sender: redis.Redis):
        worker_group_name = self.worker_config.worker_group_name
        worker_stream_key = self.worker_config.worker_stream_key
        ack_stream_key = self.worker_config.ack_stream_key

        logger.info("deblur worker running")

        while True:
            payload = await receiver.xreadgroup(groupname=worker_group_name,consumername='c2',block=0,
                                                count=1,streams={worker_stream_key:'>'})
            logger.info("deblur worker process new input")

            tag =payload[0][1][0]
            image = pickle.loads(payload[0][1][1][b'image'])

            image = read_image_from_binary(image,cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            logger.info("deblur worker decode binary images")

            result = self.model(image)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            tasks = [
                sender.xadd(ack_stream_key,{'tag':tag,'status':'ok','result':pickle.dumps(result)}),
                sender.xack(worker_stream_key, worker_stream_key, tag)
            ]
            await asyncio.gather(*tasks)

            logger.info("deblur worker send ack back")
