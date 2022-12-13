import asyncio
import pickle

import numpy as np
from loguru import logger
import redis.asyncio as redis
from pydantic import BaseModel
from models.big_lama.interface import load_lama_model
from models.big_lama.model.config import LaMaConfig
import cv2

from serve.worker.utils import read_image_from_binary, write_image_to_binary


class LaMaWorkerConfig(BaseModel):
    device: str = 'cpu'

    batch_size: int = 4
    batch_timeout_limit: int = 0

    worker_stream_key: str = "Inpainting_worker"
    worker_group_name: str = "worker"

    ack_stream_key: str = "Inpainting_finish_ack"
    ack_group_name: str = 'master'


class LaMaWorker:
    def __init__(self, worker_config: LaMaWorkerConfig = LaMaWorkerConfig(), model_config: LaMaConfig = LaMaConfig()):
        self.worker_config: LaMaWorkerConfig = worker_config
        self.model_config = model_config

        self.model = load_lama_model(self.worker_config.device, self.model_config)

    def test(self, receiver):
        receive_group_name = self.worker_config.worker_group_name
        receive_stream_key = self.worker_config.worker_stream_key
        send_stream_key = self.worker_config.ack_stream_key
        batch_size = self.worker_config.batch_size

        payload = receiver.xreadgroup(groupname=receive_group_name, consumername='c2', block=0,
                                      count=batch_size, streams={receive_stream_key: '>'})

    def input_check(self, images, masks) -> bool:

        for img, msk in zip(images, masks):
            if img.shape[:2] != msk.shape:
                return False
        return True

    async def process(self, receiver: redis.Redis, sender: redis.Redis):
        receive_group_name = self.worker_config.worker_group_name
        receive_stream_key = self.worker_config.worker_stream_key
        send_stream_key = self.worker_config.ack_stream_key
        batch_size = self.worker_config.batch_size

        logger.info("lama worker running")

        while True:
            logger.info("lama worker ready")
            payload = await receiver.xreadgroup(groupname=receive_group_name, consumername='c2', block=0,
                                                count=batch_size, streams={receive_stream_key: '>'})
            logger.info("lama worker process new input")
            tags, images, masks, results = [], [], [], []
            for i, cur in enumerate(payload[0][1]):
                tags.append(cur[0])
                images.append(pickle.loads(cur[1][b'image']))
                masks.append(pickle.loads(cur[1][b'mask']))

            num = len(tags)

            # decode binary file
            images = [read_image_from_binary(image, cv2.IMREAD_COLOR) for image in images]

            # for i, img in enumerate(images):
            #     cv2.imwrite(f'test/output/origin_{i}.jpg', img)

            images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

            masks = [read_image_from_binary(mask, cv2.IMREAD_GRAYSCALE) for mask in masks]

            # for i, msk in enumerate(masks):
            #     cv2.imwrite(f'test/output/mask_{i}.jpg', msk)

            logger.info("lama worker decode binary images")

            if not self.input_check(images, masks):
                logger.error("lama worker input not valid")
                tasks = []
                for i in range(num):
                    tasks.append(sender.xadd(send_stream_key,
                                    {'tag': tags[i], 'status': 'invalid', 'result': 'error'}))
                    tasks.append(receiver.xack(receive_stream_key, receive_group_name, tags[i]))
                await asyncio.gather(*tasks)
                continue

            results = self.model(images, masks)

            # for i, r in enumerate(results):
            #     cv2.imwrite(f'test/output/result_{i}.jpg', r)

            results = [write_image_to_binary(result) for result in results]
            logger.info('lama worker finish process')

            tasks = []
            for i in range(num):
                tasks.append(
                    sender.xadd(send_stream_key, {'tag': tags[i], 'status': 'ok', 'result': pickle.dumps(results[i])}))
                tasks.append(receiver.xack(receive_stream_key, receive_group_name, tags[i]))

            await asyncio.gather(*tasks)

            logger.info("lama worker send ack back")
            # ack
