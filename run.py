import asyncio

import redis.asyncio as redis

from serve.worker.lama_worker import LaMaWorker

conn = redis.Redis()
# print(asyncio.run(conn.ping()))

lama_worker = LaMaWorker()

asyncio.run(lama_worker.process(conn,conn))
# lama_worker.test(conn)