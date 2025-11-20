#!/usr/bin/env python3
"""
Minimal client example for /eval_once.

Usage:
  python example.py leetcode /volume/lli02/projects/example.json --host 127.0.0.1 --port 8080 --cpu-tag 2 --timeout 10
  python example.py opencode /volume/lli02/projects/opc_example.json

Env alternatives:
  SANDBOX_HOST, SANDBOX_PORT, CPU_TAG, TIMEOUT_S, MEM_MB
"""

import argparse
import json
import os
import sys
import urllib.request
import orjson
loads = orjson.loads

def post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode('utf-8', errors='replace')
    try:
        return json.loads(body)
    except Exception:
        return {'error': 'invalid json response', 'raw': body}


def main():
    ap = argparse.ArgumentParser(description='Call SandboxFusion /eval_once with a single record.')
    ap.add_argument('source', nargs='?', choices=['leetcode', 'opencode'], help='record type')
    ap.add_argument('json_path', nargs='?', help='path to record json (e.g., example.json)')
    ap.add_argument('--host', default=os.getenv('SANDBOX_HOST', '127.0.0.1'))
    ap.add_argument('--port', type=int, default=int(os.getenv('SANDBOX_PORT', '8080')))
    ap.add_argument('--cpu-tag', default=os.getenv('CPU_TAG'))
    ap.add_argument('--timeout', type=float, default=float(os.getenv('TIMEOUT_S', '20')))
    ap.add_argument('--mem-mb', type=int, default=int(os.getenv('MEM_MB', '-1')))
    args = ap.parse_args()

    # Defaults if not provided
    source = args.source or 'leetcode'
    json_path = args.json_path or ('/volume/lli02/projects/opc_example.json' if source == 'opencode' else '/volume/lli02/projects/example.json')
    count = 0
    with open(json_path, 'r', encoding="utf-8") as f:
        for line in f:
            record = loads(line)

    # with open(json_path, 'r') as f:
    #     record = json.load(f)

            payload = {
                'record': record,
                'source': source,
                'timeout_s': args.timeout,
                'memory_limit_MB': args.mem_mb,
            }
            if args.cpu_tag and str(args.cpu_tag).strip():
                payload['cpu_tag'] = args.cpu_tag

            url = f'http://{args.host}:{args.port}/eval_once'
            # print('POST', url)
            res = post(url, payload)
            print(json.dumps(res, indent=2, ensure_ascii=False))
            passed = res['passed']
            if not passed:
                count += 1
            if res['profile']['error_msg']:
                print(res['profile']['error_msg'])
    print(count)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

