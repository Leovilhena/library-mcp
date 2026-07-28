import asyncio
from typing import Callable


async def run_and_wait(fn: Callable[..., str], *args: object) -> str:
    """Call a `learn`/`learn_text`-style function that schedules ingestion in
    the background via `asyncio.create_task` and returns immediately, then
    wait for that background task to actually finish before the caller
    asserts on resulting DB state.
    """
    before = asyncio.all_tasks()
    result = fn(*args)
    new_tasks = asyncio.all_tasks() - before
    for task in new_tasks:
        await task
    return result
