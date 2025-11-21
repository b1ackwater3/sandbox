# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Union

import json
import re
from string import Template
from textwrap import dedent

from fastapi import APIRouter
from pydantic import BaseModel, Field

from sandbox.runners import CODE_RUNNERS
from sandbox.runners.types import CodeRunArgs, CommandRunStatus


# Minimal, isolated eval API for single-record evaluation + profiling.
eval_router = APIRouter()


class EvalOnceRequest(BaseModel):
    record: Dict[str, Any]
    source: Optional[str] = Field(None, examples=['leetcode', 'opencode', 'auto'])
    cpu_tag: Optional[Union[str, List[int]]] = None
    timeout_s: float = 10.0
    memory_limit_MB: int = -1


class Profile(BaseModel):
    avg_time: float
    all_time: List[float]
    error_msg: Optional[str] = None


class EvalOnceResponse(BaseModel):
    passed: bool
    profile: Profile


def _parse_cpu_tag(tag: Optional[Union[str, List[int]]]) -> List[int]:
    if tag is None:
        return []
    if isinstance(tag, list):
        return sorted(set(int(x) for x in tag))
    s = str(tag).strip()
    out: List[int] = []
    if not s:
        return out
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                out.extend(range(int(a), int(b) + 1))
            except Exception:
                continue
        else:
            try:
                out.append(int(part))
            except Exception:
                continue
    return sorted(set(out))


def _detect_source(rec: Dict[str, Any], prefer: Optional[str]) -> str:
    if prefer in ('leetcode', 'opencode'):
        return prefer
    # heuristics
    if 'tigan_id' in rec or 'private_testcases' in rec:
        return 'leetcode'
    if 'problem_idx' in rec or 'test_case' in rec:
        return 'opencode'
    return 'leetcode'


def _extract_code_from_generation(generation: str) -> str:
    # Prefer ```python fenced code
    m = re.findall(r"```python\s*([\s\S]*?)\s*```", generation, flags=re.IGNORECASE)
    if m:
        return m[0]
    m = re.findall(r"```\s*([\s\S]*?)\s*```", generation, flags=re.IGNORECASE)
    if m:
        return m[0]
    return generation or ''


def _build_driver_leetcode(rec: Dict[str, Any], cpus: List[int]) -> str:
    # Build a single python script that: set CPU affinity -> define solution -> run cases with timing+assert
    generation = rec.get('generation') or ''
    code = _extract_code_from_generation(generation)
    if not code:
        vps = rec.get('valid_python_solutions_final') or []
        if isinstance(vps, list) and vps:
            code = vps[0].get('code', '') or ''

    fn_name = rec.get('fn_name') or 'solve'

    # testcases format: {"input": {param: [v1, v2, ...]}, "output": [o1, o2, ...]}
    tcs = rec.get('private_testcases') or {}
    inputs = tcs.get('input') or {}
    outputs = tcs.get('output') or []

    # Render via Template to avoid f-string brace conflicts.
    tpl = Template(dedent(
        r"""
        import os, json, time, inspect, copy, math, re
        # Keep imports minimal to avoid polluting user code (e.g., do not override builtins.pow)
        from typing import *
        import re
        import itertools
        import bisect
        import collections, heapq, bisect, string, sys, functools, operator
        from string import *
        from sys import maxsize, stdin
        from bisect import bisect_left, bisect_right, insort_left, insort_right, insort
        from itertools import *
        from operator import xor
        try:
            from itertools import pairwise 
        except Exception:
            def pairwise(iterable):
                it = iter(iterable)
                prev = next(it, None)
                for x in it:
                    yield prev, x
                    prev = x
        from heapq import heappush, heappop, heapify
        from collections import *
        from functools import *
        try:
            import sortedcontainers  # optional
        except Exception:
            sortedcontainers = None  # type: ignore
        try:
            import lctk  # optional
        except Exception:
            lctk = None  # type: ignore

        class ListNode(object):
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next

        class TreeNode(object):
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        def __EVAL_build_list(vals):
            head = None
            cur = None
            for v in (vals or []):
                node = ListNode(v)
                if head is None:
                    head = node
                    cur = node
                else:
                    cur.next = node
                    cur = node
            return head

        def __EVAL_listnode_to_list(head):
            out = []
            cur = head
            while cur is not None:
                out.append(cur.val)
                cur = cur.next
            return out

        def __EVAL_build_tree(vals):
            if not vals:
                return None
            it = iter(vals)
            root_val = next(it)
            root = TreeNode(root_val) if root_val is not None else None
            if root is None:
                return None
            q = [root]
            while q:
                node = q.pop(0)
                # Only expand non-None nodes; Leetcode level-order input does not
                # provide children for None placeholders
                try:
                    lv = next(it)
                except StopIteration:
                    break
                if lv is not None:
                    node.left = TreeNode(lv)
                    q.append(node.left)
                try:
                    rv = next(it)
                except StopIteration:
                    break
                if rv is not None:
                    node.right = TreeNode(rv)
                    q.append(node.right)
            return root

        def __EVAL_tree_to_list(root):
            # Preserve positional indices like Leetcode serialization
            if root is None:
                return []
            out_map = {}
            q = [(root, 0)]
            while q:
                node, idx = q.pop(0)
                if node is None:
                    out_map[idx] = None
                    continue
                out_map[idx] = node.val
                q.append((node.left, 2 * idx + 1))
                q.append((node.right, 2 * idx + 2))
            size = (max(out_map.keys()) + 1) if out_map else 0
            arr = [None] * size
            for i, v in out_map.items():
                if i < size:
                    arr[i] = v
            while arr and arr[-1] is None:
                arr.pop()
            return arr

        __EVAL_PARAM_TYPES = $PARAM_TYPES
        __EVAL_RET_TYPE = $RET_TYPE

        def __EVAL_is_tree_type(t: str) -> bool:
            t = (t or '').lower()
            return ('treenode' in t) or ('binarytree' in t) or (t == 'tree') or ('tree' in t)

        def __EVAL_is_listnode_type(t: str) -> bool:
            t = (t or '').lower()
            return ('listnode' in t) or ('linkedlist' in t)

        def __EVAL_is_list_of_scalars(v) -> bool:
            if not isinstance(v, list):
                return False
            for x in v:
                if isinstance(x, (list, tuple, dict)):
                    return False
            return True

        def __EVAL_convert_arg(name, v, idx):
            # Priority: types -> name heuristic (no p/q) -> raw
            t = None
            try:
                if isinstance(__EVAL_PARAM_TYPES, list) and idx < len(__EVAL_PARAM_TYPES):
                    t = __EVAL_PARAM_TYPES[idx]
            except Exception:
                t = None
            if isinstance(v, list):
                if t:
                    if __EVAL_is_tree_type(t):
                        return __EVAL_build_tree(v)
                    if __EVAL_is_listnode_type(t):
                        return __EVAL_build_list(v)
                nm = (name or '').lower()
                # fallback heuristics only when types are missing
                if not t and __EVAL_is_list_of_scalars(v):
                    if ('root' in nm) or ('tree' in nm) or ('node' in nm):
                        return __EVAL_build_tree(v)
                    if (nm in ('head', 'head1', 'head2', 'heada', 'headb', 'l1', 'l2')) or nm.startswith('head') or nm.endswith('list'):
                        return __EVAL_build_list(v)
            return v

        def __EVAL_bind_args(param_names, inputs, index):
            keys = list(inputs.keys())
            used = set()
            args = []
            for pname in param_names:
                if pname in inputs:
                    args.append(__EVAL_convert_arg(pname, inputs[pname][index]))
                    used.add(pname)
                    continue
                # heuristic fallback: pick first unused key
                pick = None
                # prefer likely structure keys
                for k in keys:
                    if k in used:
                        continue
                    vk = inputs[k][index]
                    if isinstance(vk, list):
                        pick = k
                        break
                if pick is None:
                    for k in keys:
                        if k not in used:
                            pick = k
                            break
                if pick is None:
                    args.append(None)
                else:
                    used.add(pick)
                    args.append(__EVAL_convert_arg(pname, inputs[pick][index]))
            return args

        try:
            _cpus = set($CPUS)
            if _cpus:
                os.sched_setaffinity(0, _cpus)
        except Exception:
            pass

        $CODE

        def __EVAL_get_fn():
            try:
                s = Solution()
                return getattr(s, '$FN_NAME')
            except Exception:
                return globals().get('$FN_NAME')

        __EVAL_FN = __EVAL_get_fn()
        if __EVAL_FN is None:
            print(json.dumps({'passed': False, 'profile': {'avg_time': -1, 'all_time': [], 'error_msg': 'function not found'}}))
            raise SystemExit(0)

        __EVAL_INPUTS = json.loads(r'''$INPUTS_JSON''')
        __EVAL_OUTPUTS = json.loads(r'''$OUTPUTS_JSON''')

        _re_int = re.compile(r"^[+-]?\d+\Z")
        _re_float = re.compile(r"^[+-]?(?:\d*\.\d+|\d+\.\d*)(?:[eE][+-]?\d+)?\Z")

        def __EVAL_norm(x):
            # normalize numeric strings and containers for robust comparison
            try:
                import numpy as _np  # type: ignore
            except Exception:
                _np = None

            if _np is not None and hasattr(_np, 'isscalar') and _np.isscalar(x):
                try:
                    return x.item()
                except Exception:
                    pass

            if _np is not None and hasattr(_np, 'ndarray') and isinstance(x, _np.ndarray):
                try:
                    return __EVAL_norm(x.tolist())
                except Exception:
                    pass

            if isinstance(x, (int, float, bool)):
                return x
            if isinstance(x, str):
                s = x.strip()
                sl = s.lower()
                if sl in ('true', 'false'):
                    return sl == 'true'
                if sl in ('null', 'none'):
                    return None
                # try json decode for arrays/objects/primitives
                if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')) or _re_int.match(s) or _re_float.match(s) or sl in ('true','false','null','none'):
                    try:
                        return __EVAL_norm(json.loads(s))
                    except Exception:
                        pass
                if _re_int.match(s):
                    try:
                        return int(s)
                    except Exception:
                        return x
                if _re_float.match(s):
                    try:
                        return float(s)
                    except Exception:
                        return x
                return x
            if isinstance(x, (list, tuple)):
                t = [__EVAL_norm(i) for i in x]
                return type(x)(t)
            if isinstance(x, dict):
                return {k: __EVAL_norm(v) for k, v in x.items()}
            return x

        def __EVAL_equal(a, b):
            a = __EVAL_norm(a)
            b = __EVAL_norm(b)
            # numbers (int/float/bool)
            if isinstance(a, (int, float, bool)) and isinstance(b, (int, float, bool)):
                if isinstance(a, float) or isinstance(b, float):
                    try:
                        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
                    except Exception:
                        return a == b
                return a == b
            # sequences
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                if len(a) != len(b):
                    return False
                return all(__EVAL_equal(x, y) for x, y in zip(a, b))
            # dicts
            if isinstance(a, dict) and isinstance(b, dict):
                if set(a.keys()) != set(b.keys()):
                    return False
                return all(__EVAL_equal(a[k], b[k]) for k in a.keys())
            # fallback
            return a == b

        try:
            _sig = inspect.signature(__EVAL_FN)
            _param_names = [p.name for p in _sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name != 'self']
        except Exception:
            _param_names = None

        _keys = list(__EVAL_INPUTS.keys())
        _case_num = len(__EVAL_OUTPUTS) if __EVAL_OUTPUTS is not None else 0
        if _keys:
            try:
                _lens = [len(__EVAL_INPUTS[k]) for k in _keys]
                _case_num = min(_case_num, min(_lens)) if _case_num else min(_lens)
            except Exception:
                pass

        _times = []
        for _i in range(int(_case_num)):
            if _param_names and set(_param_names).issubset(set(_keys)):
                _args = [__EVAL_convert_arg(k, __EVAL_INPUTS[k][_i], idx) for idx, k in enumerate(_param_names)]
                _kwargs = {}
            else:
                if _param_names:
                    _args = __EVAL_bind_args(_param_names, __EVAL_INPUTS, _i)
                else:
                    _args = [__EVAL_convert_arg(k, __EVAL_INPUTS[k][_i], idx) for idx, k in enumerate(_keys)]
                _kwargs = {}
            _t0 = time.perf_counter()
            _got = __EVAL_FN(*_args, **_kwargs)
            _dt = time.perf_counter() - _t0
            _times.append(_dt)
            _exp = __EVAL_OUTPUTS[_i] if _i < len(__EVAL_OUTPUTS) else None

            # Coerce result guided by return type (if provided)
            try:
                if __EVAL_RET_TYPE and __EVAL_is_tree_type(__EVAL_RET_TYPE) and hasattr(_got, 'left'):
                    _got = __EVAL_tree_to_list(_got)
                elif __EVAL_RET_TYPE and __EVAL_is_listnode_type(__EVAL_RET_TYPE) and hasattr(_got, 'next'):
                    _got = __EVAL_listnode_to_list(_got)
            except Exception:
                pass
            # Also handle cases where return type is missing but object clearly is a node
            if hasattr(_got, 'left') and hasattr(_got, 'right'):
                _got = __EVAL_tree_to_list(_got)
            elif hasattr(_got, 'next') and hasattr(_got, 'val'):
                _got = __EVAL_listnode_to_list(_got)

            # Primary deep equality
            if not __EVAL_equal(_got, _exp):
                # Unordered list-of-lists fallback
                _matched = False
                try:
                    if isinstance(_got, list) and isinstance(_exp, list):
                        def _to_hashable(x):
                            x = __EVAL_norm(x)
                            return tuple(x) if isinstance(x, list) else (x,)
                        so = set(_to_hashable(x) for x in _got)
                        se = set(_to_hashable(x) for x in _exp)
                        if so == se:
                            _matched = True
                except Exception:
                    pass

                # In-place mutation fallback: when outputs is None-like, compare mutated first arg to expected
                if not _matched:
                    try:
                        if (_got is None) and (len(_args) > 0):
                            _cand = _args[0]
                            try:
                                arg0_t = __EVAL_PARAM_TYPES[0] if isinstance(__EVAL_PARAM_TYPES, list) and __EVAL_PARAM_TYPES else None
                            except Exception:
                                arg0_t = None
                            if arg0_t and __EVAL_is_tree_type(arg0_t) and hasattr(_cand, 'left'):
                                _cand = __EVAL_tree_to_list(_cand)
                            elif arg0_t and __EVAL_is_listnode_type(arg0_t) and hasattr(_cand, 'next'):
                                _cand = __EVAL_listnode_to_list(_cand)
                            else:
                                if hasattr(_cand, 'left') and hasattr(_cand, 'right'):
                                    _cand = __EVAL_tree_to_list(_cand)
                                elif hasattr(_cand, 'next') and hasattr(_cand, 'val'):
                                    _cand = __EVAL_listnode_to_list(_cand)
                            if __EVAL_equal(_cand, _exp):
                                _matched = True
                    except Exception:
                        pass

                if not _matched:
                    _msg = f'case {_i} mismatch: expect={_exp} got={_got}'
                    print(json.dumps({'passed': False, 'profile': {'avg_time': -1, 'all_time': _times, 'error_msg': _msg}}))
                    raise SystemExit(0)

        _avg = (sum(_times)/len(_times)) if _times else 0.0
        print(json.dumps({'passed': True, 'profile': {'avg_time': _avg, 'all_time': _times, 'error_msg': None}}))
        """
    ))
    return tpl.safe_substitute(
        CPUS=json.dumps(cpus),
        CODE=code,
        FN_NAME=fn_name,
        INPUTS_JSON=json.dumps(inputs, ensure_ascii=True),
        OUTPUTS_JSON=json.dumps(outputs, ensure_ascii=True),
        PARAM_TYPES=json.dumps(param_types_list, ensure_ascii=True),
        RET_TYPE=(json.dumps(ret_type_norm) if ret_type_norm is not None else 'null'),
    )


def _build_driver_opencode(rec: Dict[str, Any], cpus: List[int]) -> str:
    imports = rec.get('import', '') or ''
    generation = rec.get('generation') or ''
    code = _extract_code_from_generation(generation)

    # Collect assert lines
    test_case = rec.get('test_case') or ''
    asserts = [ln.strip() for ln in str(test_case).splitlines() if ln.strip().startswith('assert ')]

    tpl = Template(dedent(
        r"""
        import os, json, time
        from typing import *
        try:
            _cpus = set($CPUS)
            if _cpus:
                os.sched_setaffinity(0, _cpus)
        except Exception:
            pass

        $IMPORTS

        $CODE

        __EVAL_ASSERTS = $ASSERTS
        _times = []
        _g = globals()
        for _i, _line in enumerate(__EVAL_ASSERTS):
            _t0 = time.perf_counter()
            try:
                exec(_line, _g, _g)
            except AssertionError:
                _dt = time.perf_counter() - _t0
                _times.append(_dt)
                print(json.dumps({'passed': False, 'profile': {'avg_time': -1, 'all_time': _times, 'error_msg': f'assert failed: {_line}'}}))
                raise SystemExit(0)
            except Exception as _e:
                _dt = time.perf_counter() - _t0
                _times.append(_dt)
                print(json.dumps({'passed': False, 'profile': {'avg_time': -1, 'all_time': _times, 'error_msg': str(_e)}}))
                raise SystemExit(0)
            _dt = time.perf_counter() - _t0
            _times.append(_dt)

        _avg = (sum(_times)/len(_times)) if _times else 0.0
        print(json.dumps({'passed': True, 'profile': {'avg_time': _avg, 'all_time': _times, 'error_msg': None}}))
        """
    ))
    return tpl.safe_substitute(
        CPUS=json.dumps(cpus),
        IMPORTS=imports,
        CODE=code,
        ASSERTS=json.dumps(asserts, ensure_ascii=True),
    )


@eval_router.post('/eval_once', response_model=EvalOnceResponse)
async def eval_once(req: EvalOnceRequest) -> EvalOnceResponse:
    cpus = _parse_cpu_tag(req.cpu_tag)
    source = _detect_source(req.record, req.source)

    if source == 'leetcode':
        code = _build_driver_leetcode(req.record, cpus)
        outputs = (req.record.get('private_testcases') or {}).get('output') or []
        case_count = len(outputs)
    else:
        code = _build_driver_opencode(req.record, cpus)
        asserts = [ln for ln in (req.record.get('test_case') or '').splitlines() if ln.strip().startswith('assert ')]
        case_count = len(asserts)

    # total timeout ~= per-case timeout * cases + small overhead
    run_timeout = max(5.0, float(req.timeout_s) * max(1, case_count) + 2.0)

    result = await CODE_RUNNERS['python'](
        CodeRunArgs(code=code, run_timeout=run_timeout, memory_limit_MB=req.memory_limit_MB))

    # Parse result robustly; turn timeouts/empty stdout into a structured failure
    err: Optional[str] = None
    out: Optional[Dict[str, Any]] = None
    rr = getattr(result, 'run_result', None)
    if not rr:
        err = 'no run_result'
    elif rr.status == CommandRunStatus.TimeLimitExceeded:
        err = 'TimeLimitExceeded'
    else:
        s = rr.stdout or ''
        lines = [ln for ln in s.strip().splitlines() if ln.strip()]
        if lines:
            try:
                out = json.loads(lines[-1])
            except Exception as e:
                err = f'bad json stdout: {e}'
        else:
            err = rr.stderr or 'empty stdout'

    if err or (rr and rr.return_code not in (0, None)) or not isinstance(out, dict):
        return EvalOnceResponse(passed=False, profile=Profile(avg_time=-1, all_time=[], error_msg=err or 'runtime error'))

    return EvalOnceResponse(**out)  # type: ignore[arg-type]
