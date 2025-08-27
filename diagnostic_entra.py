#!/usr/bin/env python
"""diagnostic_entra.py

Diagnostics that exercise the SAME code path as the application (auth.entra.get_redis_client).
Use this when diagnostic.py (raw / independent) succeeds but the app still hangs, to isolate
issues inside get_redis_client or subsequent RedisVL operations.

Environment variables leveraged by entra.py:
  REDIS_HOST (required)              Target Redis host (e.g. myredis.redis.cache.windows.net)
  REDIS_PORT                         Target Redis port (default 10000 for AMR, else 6379)
  REDIS_DISABLE_ENTRA=1              Force plain (no Entra) path
  REDIS_ENTRA_DEBUG=1                Verbose timing + steps
  REDIS_SHOW_TOKEN_CLAIMS=1          Print JWT header/claims (truncated)
  REDIS_TOKEN_MANAGER_DEBUG=1        Elevate token manager logger
  REDIS_REQUIRE_MODULES=search,json  Warn if modules missing
  REDIS_SOCKET_TIMEOUT=5             Socket connect & command timeout
  REDIS_SSL=1                        Enable SSL (set 0/false to disable)

Script flags (optional):
  --index       Create & drop a temporary RediSearch index (needs redisvl)
  --json        Test JSON.SET/GET (needs RedisJSON)
  --modules     List modules explicitly (in addition to entra.py auto check)
  --ping-loop N Repeated pings (default 5)
  --namespace X Use a custom namespace for temp index names (default diag)

Exit code 0 = all selected tests passed; 1 = any failure.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

from auth.entra import get_redis_client  # uses your enhanced debug + timeouts

try:
    from redisvl.index import SearchIndex
except ImportError:  # pragma: no cover
    SearchIndex = None  # type: ignore

OK = "✅"
FAIL = "❌"
WARN = "⚠️"

def list_modules(client):
    try:
        mods = client.execute_command("MODULE", "LIST")
    except Exception as e:
        print(f"{WARN} MODULE LIST failed: {e}")
        return []
    names = []
    for m in mods:
        if isinstance(m, dict):
            n = m.get(b"name") or m.get("name")
            if n:
                names.append(n.decode() if isinstance(n, bytes) else n)
        elif isinstance(m, (list, tuple)):
            for i, v in enumerate(m):
                if v == b"name" and i + 1 < len(m):
                    n = m[i + 1]
                    names.append(n.decode() if isinstance(n, bytes) else n)
    return names

def test_index(client, namespace: str) -> bool:
    if SearchIndex is None:
        print(f"{WARN} redisvl not installed; skipping index test")
        return True
    schema = {
        "index": {"name": f"{namespace}_idx", "prefix": f"{namespace}_idx", "storage_type": "json"},
        "fields": [
            {"name": "k", "type": "tag"},
            {"name": "v", "type": "text"},
        ],
    }
    try:
        # Extra handshake to ensure auth before FT.CREATE
        try:
            client.ping()
        except Exception as e:
            print(f"{WARN} Pre-index PING failed (will still attempt create): {e}")
        idx = SearchIndex.from_dict(schema, client=client)
        t0 = time.perf_counter(); idx.create(overwrite=True, drop=True); dt = (time.perf_counter()-t0)*1000
        print(f"{OK} RediSearch index create in {dt:.1f} ms")
        return True
    except Exception as e:
        msg = str(e)
        if 'Authentication required' in msg or 'NOAUTH' in msg.upper():
            disabled = os.getenv('REDIS_DISABLE_ENTRA') in {'1','true','yes'}
            print(f"{FAIL} Index create failed (auth). Entra disabled={disabled}. If this Azure Managed Redis requires Entra, unset REDIS_DISABLE_ENTRA. Error: {msg}")
        else:
            print(f"{FAIL} Index create failed: {e}")
        return False

def test_json(client) -> bool:
    try:
        client.execute_command("JSON.SET", "diag_json:1", "$", '{"x":1}')
        val = client.execute_command("JSON.GET", "diag_json:1", "$")
        print(f"{OK} JSON roundtrip: {val}")
        return True
    except Exception as e:
        print(f"{FAIL} JSON test failed: {e}")
        return False

def ping_loop(client, count: int):
    for i in range(count):
        t0 = time.perf_counter()
        try:
            pong = client.ping()
            dt = (time.perf_counter()-t0)*1000
            print(f"  #{i+1}: {OK} {pong} {dt:.1f} ms")
        except Exception as e:
            print(f"  #{i+1}: {FAIL} {e}")
        time.sleep(0.8)

def parse_args():
    ap = argparse.ArgumentParser(description="Diagnostics using auth.entra.get_redis_client")
    ap.add_argument('--index', action='store_true', help='Test temporary RediSearch index creation')
    ap.add_argument('--json', action='store_true', help='Test RedisJSON roundtrip')
    ap.add_argument('--modules', action='store_true', help='List modules explicitly')
    ap.add_argument('--ping-loop', type=int, nargs='?', const=5, help='Run repeated pings (default 5)')
    ap.add_argument('--namespace', default='diag', help='Namespace prefix for temporary index')
    return ap.parse_args()

def main():
    args = parse_args()
    host = os.getenv('REDIS_HOST')
    if not host:
        print(f"{FAIL} REDIS_HOST not set")
        return 1
    port = int(os.getenv('REDIS_PORT', '10000'))
    print("Entrapath Diagnostics")
    print("Host:", host)
    print("Port:", port)
    print("Disable Entra:", os.getenv('REDIS_DISABLE_ENTRA') in {'1','true','yes'})
    print("Timeout:", os.getenv('REDIS_SOCKET_TIMEOUT', '5'))

    failed = False

    print("\n[CONNECT] Creating client via get_redis_client ...")
    try:
        client = get_redis_client(host, port)
        print(f"{OK} Client created")
    except Exception as e:
        print(f"{FAIL} Client creation failed: {e}")
        return 1

    print("[PING] Single ping test...")
    try:
        t0 = time.perf_counter(); pong = client.ping(); dt = (time.perf_counter()-t0)*1000
        print(f"{OK} PING {pong} {dt:.1f} ms")
    except Exception as e:
        print(f"{FAIL} PING failed: {e}")
        failed = True

    if args.modules:
        print("\n[MODULES]")
        mods = list_modules(client)
        if mods:
            print(f"{OK} Modules: {mods}")
        else:
            print(f"{WARN} No modules reported")

    if args.index:
        print("\n[INDEX]")
        if not test_index(client, args.namespace):
            failed = True

    if args.json:
        print("\n[JSON]")
        if not test_json(client):
            failed = True

    if args.ping_loop:
        print("\n[PING LOOP]")
        ping_loop(client, args.ping_loop)

    print("\nDone.")
    return 1 if failed else 0

if __name__ == '__main__':
    raise SystemExit(main())
