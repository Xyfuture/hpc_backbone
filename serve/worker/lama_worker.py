import asyncio
import pickle

from models.big_lama.interface import load_model
import redis.asyncio as redis
from pydantic import BaseModel

from models.big_lama.model.config import LaMaConfig


class LaMaWorkerConfig(BaseModel):
    batch_size: int = 4
    batch_timeout_limit: int = 0

    receive_stream_key:str = "Inpainting"
    receive_group_name:str = "worker"

    send_stream_key:str = "finish_ack"

class LaMaWorker:
    def __init__(self, worker_config: LaMaWorkerConfig=LaMaWorkerConfig(), model_config:LaMaConfig=LaMaConfig()):
        self.worker_config: LaMaWorkerConfig = worker_config
        self.model_config = model_config

        self.model = load_model(self.model_config)

    def test(self,receiver):
        receive_group_name = self.worker_config.receive_group_name
        receive_stream_key = self.worker_config.receive_stream_key
        send_stream_key = self.worker_config.send_stream_key
        batch_size = self.worker_config.batch_size

        payload =  receiver.xreadgroup(groupname=receive_group_name, consumername='c2', block=0,
                                            count=batch_size, streams={receive_stream_key: '>'})

    async def process(self, receiver: redis.Redis,sender:redis.Redis ):
        receive_group_name = self.worker_config.receive_group_name
        receive_stream_key = self.worker_config.receive_stream_key
        send_stream_key = self.worker_config.send_stream_key
        batch_size = self.worker_config.batch_size

        print('here')

        while True:
            payload = await receiver.xreadgroup(groupname=receive_group_name, consumername='c2', block=0,
                                                count=batch_size, streams={receive_stream_key: '>'})
            print('get data')
            tags,images,masks,results = [],[],[],[]
            for i,cur in enumerate(payload[0][1]):
                tags.append(cur[0])
                images.append(pickle.loads(cur[1][b'image']))
                masks.append(pickle.loads(cur[1][b'mask']))

            num = len(tags)

            results = self.model(images,masks)
            print('get results')

            tasks = [
                sender.xadd(send_stream_key , {'tag': tags[i],'result':pickle.dumps(results[i]) } )
                for i in range(num)
            ]

            await asyncio.gather(*tasks)
            print('send ack')
            # ack







