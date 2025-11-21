#!/usr/bin/env python3
"""
Concurrent batch client for /eval_once.

Reads a jsonl file; splits records into N groups (default 128) by problem id
hash; starts N workers, each bound to a fixed CPU tag, and posts records in its
group sequentially. Prints the number of records whose `passed` is False.

Example:
  python example.py /path/to/data.jsonl --host 127.0.0.1 --port 8080 \
         --workers 128 --cpu-start 0 --timeout 10
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict

import aiohttp


def detect_source(rec: Dict[str, Any]) -> str:
    if 'tigan_id' in rec or 'private_testcases' in rec:
        return 'leetcode'
    if 'problem_idx' in rec or 'test_case' in rec:
        return 'opencode'
    return 'leetcode'


def get_problem_id(rec: Dict[str, Any], idx: int) -> str:
    return str(rec.get('tigan_id') or rec.get('problem_idx') or rec.get('id') or idx)


async def post_with_retry(url: str, payload: Dict[str, Any], retries: int, timeout_total: float) -> Dict[str, Any]:
    delay = 0.2
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            to = aiohttp.ClientTimeout(total=timeout_total)
            headers = {'Content-Type': 'application/json', 'Connection': 'close'}
            async with aiohttp.request('POST', url, json=payload, timeout=to, headers=headers) as resp:
                # accept non-JSON content-type too
                return await resp.json(content_type=None)
        except Exception as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 2.0)
                continue
            raise


async def worker(url: str, cpu_tag: str, q: asyncio.Queue, timeout_s: float, mem_mb: int, counters: Dict[str, int],
                 failed: list, retries: int):
    while True:
        item = await q.get()
        if item is None:
            q.task_done()
            break
        idx, rec = item
        payload = {
            'record': rec,
            'source': detect_source(rec),
            'cpu_tag': cpu_tag,
            'timeout_s': timeout_s,
            'memory_limit_MB': mem_mb,
        }
        try:
            data = await post_with_retry(url, payload, retries=retries, timeout_total=None)
            if not isinstance(data, dict) or not data.get('passed', False):
                counters['false'] += 1
                err = None
                try:
                    err = data.get('profile', {}).get('error_msg') if isinstance(data, dict) else None
                except Exception:
                    err = None
                failed.append({'idx': idx, 'cpu_tag': cpu_tag, 'record': rec, 'error': err})
        except Exception as e:
            counters['false'] += 1
            failed.append({'idx': idx, 'cpu_tag': cpu_tag, 'record': rec, 'error': str(e)})
        finally:
            q.task_done()


async def main_async(args):
    url = f'http://{args.host}:{args.port}/eval_once'
    workers = int(args.workers)
    cpu_start = int(args.cpu_start)

    # Create per-worker queues and tasks
    queues = [asyncio.Queue() for _ in range(workers)]
    counters = {'false': 0}
    failed: List[Dict[str, Any]] = []
    tasks = []
    for i in range(workers):
        cpu_tag = str(cpu_start + i)
        tasks.append(
            asyncio.create_task(worker(url, cpu_tag, queues[i], args.timeout, args.mem_mb, counters, failed,
                                        retries=args.retries)))

    # Feed records to queues by problem id hash
    with open(args.jsonl, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                counters['false'] += 1
                continue
            pid = get_problem_id(rec, idx)
            wid = (hash(pid) % workers)
            await queues[wid].put((idx, rec))

    # Stop workers
    for q in queues:
        await q.put(None)
    await asyncio.gather(*tasks)

    # Write failed samples if requested
    if args.save_failed:
        try:
            import pathlib
            pathlib.Path(args.save_failed).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            with open(args.save_failed, 'w', encoding='utf-8') as f:
                for item in failed:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception:
            pass
    print(counters['false'])


def parse_args():
    ap = argparse.ArgumentParser(description='Run /eval_once over a jsonl with per-CPU workers.')
    ap.add_argument('jsonl', help='path to input jsonl file')
    ap.add_argument('--host', default=os.getenv('SANDBOX_HOST', '127.0.0.1'))
    ap.add_argument('--port', type=int, default=int(os.getenv('SANDBOX_PORT', '8080')))
    ap.add_argument('--workers', type=int, default=128, help='number of workers/CPUs (default 128)')
    ap.add_argument('--cpu-start', type=int, default=0, help='start CPU id (default 0)')
    ap.add_argument('--timeout', type=float, default=10.0, help='per-record timeout_s to send to server')
    ap.add_argument('--mem-mb', type=int, default=-1, help='memory limit MB to send to server (-1 for unlimited)')
    ap.add_argument('--retries', type=int, default=2, help='per-request retry times when network error occurs')
    ap.add_argument('--save-failed', default=None, help='path to save failed samples as JSONL')
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == '__main__':
    main()
