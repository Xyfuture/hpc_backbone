import redis as redis


conn = redis.Redis()
conn.ping()