import asyncio

import redis.asyncio as redis

from serve.worker.deblur_worker import DeBlurWorker
from serve.worker.lama_worker import LaMaWorker

conn = redis.Redis()
# print(asyncio.run(conn.ping()))

lama_worker = LaMaWorker()
deblur_worker = DeBlurWorker()

tasks = [lama_worker.process(conn, conn),
         deblur_worker.process(conn, conn)]


async def run_all(tasks):
    await asyncio.gather(*tasks)

asyncio.run(run_all(tasks))

# asyncio.run(lama_worker.process(conn,conn))
# asyncio.create_task(deblur_worker.process(conn,conn))
# lama_worker.test(conn)
