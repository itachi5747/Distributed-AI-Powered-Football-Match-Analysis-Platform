import json
import os
from typing import Any, Awaitable, Callable

import aio_pika

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
EXCHANGE_NAME = "footballiq.direct"


async def get_connection() -> aio_pika.abc.AbstractRobustConnection:
    return await aio_pika.connect_robust(RABBITMQ_URL)


async def publish_message(routing_key: str, payload: Any, priority: int = 5) -> None:
    connection = await get_connection()
    async with connection:
        channel = await connection.channel()
        exchange = await channel.declare_exchange(
            EXCHANGE_NAME,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        body_payload = payload if isinstance(payload, dict) else payload.model_dump()
        body = json.dumps(body_payload)
        await exchange.publish(
            aio_pika.Message(
                body=body.encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                priority=priority,
                content_type="application/json",
            ),
            routing_key=routing_key,
        )


async def consume_queue(
    queue_name: str,
    on_message: Callable[[dict], Awaitable[None]],
    prefetch_count: int = 1,
) -> None:
    connection = await get_connection()
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=prefetch_count)
    queue = await channel.declare_queue(queue_name, durable=True, passive=True)

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process(requeue=False):
                try:
                    payload = json.loads(message.body.decode())
                    await on_message(payload)
                except aiormq.exceptions.ChannelInvalidStateError:
                    # Channel already closed (normal after graceful shutdown)
                    # Silently ignore to let the worker exit cleanly.
                    pass
                except Exception as exc:
                    print(f"[ERROR] Failed to process message: {exc}")
                    raise
