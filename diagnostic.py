#!/usr/bin/env python
"""diagnostic.py

Standalone diagnostics for Azure Managed Redis + redis-py-entraid + redisvl.
Run this script BEFORE launching the app to pinpoint where things hang.

Usage examples:
  uv run python diagnostic.py                                  # uses REDIS_HOST/REDIS_PORT env
  REDIS_HOST=host REDIS_PORT=10000 uv run python diagnostic.py --entra
  REDIS_HOST=host REDIS_PORT=10000 REDIS_ENTRA_DEBUG=1 uv run python diagnostic.py --entra --claims --modules --index

Key environment variables:
  REDIS_HOST                Redis host (required)
  REDIS_PORT                Redis port (default 6379, 10000 for AMR)
  REDIS_SSL=1               Enable SSL (set 0 to disable)
  REDIS_SOCKET_TIMEOUT      Socket connect & command timeout seconds (default 5)
  REDIS_TOKEN_MANAGER_DEBUG=1  Elevate redis.auth.token_manager logger
  AZURE_LOG_LEVEL=debug     Azure SDK verbose logging

Flags:
  --entra       Force Entra (token) auth path test
  --plain       Force plain (no credential_provider) test
  --claims      Print decoded JWT header & claims (truncated)
  --modules     List modules
  --ping-loop   Run repeated ping loop
  --index       Attempt temporary RediSearch index create/drop (requires redisvl & modules)
  --json        Test JSON.SET/GET (requires RedisJSON module)

Exit codes:
  0 success, 1 failure detected.
"""
from __future__ import annotations
import argparse
import base64
import json
import logging
import os
import sys
import time
from typing import Optional

import redis
from azure.identity import DefaultAzureCredential, AzureCliCredential

try:
    from redis_entra import create_from_default_azure_credential  # type: ignore
except ImportError:
    try:
        from redis_entra.cred_provider import create_from_default_azure_credential  # type: ignore
    except Exception:
        create_from_default_azure_credential = None  # type: ignore

# Optional redisvl tests
try:
    from redisvl.index import SearchIndex
except ImportError:
    SearchIndex = None  # type: ignore

DEFAULT_SCOPE = "https://redis.azure.com/.default"

COL_OK = "✅"
COL_FAIL = "❌"
COL_WARN = "⚠️"


def _decode_segment(seg: str):
    pad = '=' * (-len(seg) % 4)
    return json.loads(base64.urlsafe_b64decode(seg + pad))


def decode_token(token: str) -> tuple[Optional[dict], Optional[dict]]:
    try:
        header_b64, payload_b64, *_ = token.split('.')
        return _decode_segment(header_b64), _decode_segment(payload_b64)
    except Exception:
        return None, None


def build_client(host: str, port: int, *, credential_provider=None, timeout: float, ssl: bool) -> redis.Redis:
    kwargs = dict(
        host=host,
        port=port,
        ssl=ssl,
        decode_responses=True,
        socket_timeout=timeout,
        socket_connect_timeout=timeout,
        socket_keepalive=True,
        health_check_interval=30,
        retry_on_timeout=True,
    )
    if credential_provider:
        kwargs["credential_provider"] = credential_provider
    return redis.Redis(**kwargs)


def test_plain(host: str, port: int, timeout: float, ssl: bool) -> bool:
    print("\n[PLAIN] Testing plain connection (no Entra)...")
    try:
        cli = build_client(host, port, timeout=timeout, ssl=ssl)
        t0 = time.perf_counter(); pong = cli.ping(); dt = (time.perf_counter()-t0)*1000
        print(f"{COL_OK} PING (plain) returned {pong} in {dt:.1f} ms")
        return True
    except Exception as e:
        print(f"{COL_FAIL} Plain connection failed: {e}")
        return False


def test_entra(host: str, port: int, timeout: float, show_claims: bool, force_cli_cred: bool, ssl: bool) -> bool:
    print("\n[ENTRA] Testing Entra token-based connection...")
    if create_from_default_azure_credential is None:
        print(f"{COL_FAIL} redis-py-entraid not installed; cannot test Entra.")
        return False
    try:
        cred_cls = AzureCliCredential if force_cli_cred else DefaultAzureCredential
        cred = cred_cls(exclude_interactive_browser_credential=True)  # type: ignore[arg-type]
        t0 = time.perf_counter(); token = cred.get_token(DEFAULT_SCOPE); dt = (time.perf_counter()-t0)*1000
        ttl_min = (token.expires_on - time.time())/60
        print(f"{COL_OK} Acquired token in {dt:.1f} ms; expires in {ttl_min:.1f} min")
        if show_claims:
            h, p = decode_token(token.token)
            if h: print("  header:", h)
            if p:
                trunc = {k: (str(v)[:140]+('…' if len(str(v))>140 else '')) for k,v in p.items()}
                print("  claims:", trunc)
        provider = create_from_default_azure_credential((DEFAULT_SCOPE,))
        cli = build_client(host, port, timeout=timeout, credential_provider=provider, ssl=ssl)
        print("  issuing PING (auth handshake)...", end=" ")
        t1 = time.perf_counter(); pong = cli.ping(); dtp = (time.perf_counter()-t1)*1000
        print(f"{COL_OK} {pong} in {dtp:.1f} ms")
        return True
    except Exception as e:
        print(f"{COL_FAIL} Entra connection failed: {e}")
        return False


def list_modules(cli: redis.Redis) -> list[str]:
    try:
        mods = cli.execute_command("MODULE", "LIST")
    except Exception as e:
        print(f"{COL_WARN} MODULE LIST failed: {e}")
        return []
    names = []
    for m in mods:
        if isinstance(m, dict):
            n = m.get(b'name') or m.get('name')
            if n:
                names.append(n.decode() if isinstance(n, bytes) else n)
        elif isinstance(m, (list, tuple)):
            for i,v in enumerate(m):
                if v == b'name' and i+1 < len(m):
                    n = m[i+1]
                    names.append(n.decode() if isinstance(n, bytes) else n)
    return names


def test_modules(host: str, port: int, timeout: float, entra: bool, ssl: bool) -> None:
    print("\n[MODULES] Listing modules...")
    try:
        if entra and create_from_default_azure_credential:
            provider = create_from_default_azure_credential((DEFAULT_SCOPE,))
            cli = build_client(host, port, timeout=timeout, credential_provider=provider, ssl=ssl)
        else:
            cli = build_client(host, port, timeout=timeout, ssl=ssl)
        mods = list_modules(cli)
        if mods:
            print(f"{COL_OK} Modules: {mods}")
        else:
            print(f"{COL_WARN} No modules reported or retrieval failed")
    except Exception as e:
        print(f"{COL_FAIL} Module test failed: {e}")


def test_index(host: str, port: int, timeout: float, entra: bool, ssl: bool) -> None:
    print("\n[INDEX] Temporary RediSearch index test...")
    if SearchIndex is None:
        print(f"{COL_WARN} redisvl not installed; skipping index test")
        return
    schema = {
        "index": {"name": "diag_idx", "prefix": "diag_idx", "storage_type": "json"},
        "fields": [
            {"name": "k", "type": "tag"},
            {"name": "v", "type": "text"},
        ],
    }
    try:
        if entra and create_from_default_azure_credential:
            provider = create_from_default_azure_credential((DEFAULT_SCOPE,))
            cli = build_client(host, port, timeout=timeout, credential_provider=provider, ssl=ssl)
        else:
            cli = build_client(host, port, timeout=timeout, ssl=ssl)
        # Force auth handshake BEFORE FT.CREATE so we don't get 'Authentication required'
        cli.ping()
        idx = SearchIndex.from_dict(schema)
        idx.client = cli
        t0 = time.perf_counter(); idx.create(overwrite=True, drop=True); dti = (time.perf_counter()-t0)*1000
        print(f"{COL_OK} Index create OK in {dti:.1f} ms")
        cli.execute_command("JSON.SET", "diag_idx:1", "$", '{"k":"1","v":"hello"}')
        print(f"{COL_OK} JSON.SET succeeded")
    except Exception as e:
        print(f"{COL_FAIL} Index/JSON test failed: {e}")


def test_json(host: str, port: int, timeout: float, entra: bool, ssl: bool) -> None:
    print("\n[JSON] JSON module quick test...")
    try:
        if entra and create_from_default_azure_credential:
            provider = create_from_default_azure_credential((DEFAULT_SCOPE,))
            cli = build_client(host, port, timeout=timeout, credential_provider=provider, ssl=ssl)
        else:
            cli = build_client(host, port, timeout=timeout, ssl=ssl)
        cli.execute_command("JSON.SET", "diag_json:1", "$", '{"x":1}')
        val = cli.execute_command("JSON.GET", "diag_json:1", "$")
        print(f"{COL_OK} JSON roundtrip: {val}")
    except Exception as e:
        print(f"{COL_FAIL} JSON test failed: {e}")


def ping_loop(host: str, port: int, timeout: float, entra: bool, ssl: bool, iterations: int = 5, delay: float = 1.0):
    print("\n[PING LOOP] Repeated ping test...")
    try:
        if entra and create_from_default_azure_credential:
            provider = create_from_default_azure_credential((DEFAULT_SCOPE,))
            cli = build_client(host, port, timeout=timeout, credential_provider=provider, ssl=ssl)
        else:
            cli = build_client(host, port, timeout=timeout, ssl=ssl)
        for i in range(iterations):
            t0 = time.perf_counter()
            try:
                pong = cli.ping()
                dt = (time.perf_counter()-t0)*1000
                print(f"  #{i+1}: {COL_OK} {pong} {dt:.1f} ms")
            except Exception as e:
                print(f"  #{i+1}: {COL_FAIL} {e}")
            time.sleep(delay)
    except Exception as e:
        print(f"{COL_FAIL} Could not start ping loop: {e}")


def parse_args():
    ap = argparse.ArgumentParser(description="Redis / Entra diagnostic script")
    ap.add_argument('--entra', action='store_true', help='Test Entra token auth path')
    ap.add_argument('--plain', action='store_true', help='Test plain (no credential provider) path')
    ap.add_argument('--claims', action='store_true', help='Show token claims (with --entra)')
    ap.add_argument('--modules', action='store_true', help='List modules')
    ap.add_argument('--index', action='store_true', help='Create temporary index (requires redisvl)')
    ap.add_argument('--json', action='store_true', help='Test RedisJSON commands')
    ap.add_argument('--ping-loop', action='store_true', help='Run repeated pings')
    ap.add_argument('--force-cli-cred', action='store_true', help='Use AzureCliCredential only for token')
    return ap.parse_args()


def main():
    args = parse_args()
    host = os.getenv('REDIS_HOST')
    if not host:
        print(f"{COL_FAIL} REDIS_HOST env var not set")
        return 1
    port = int(os.getenv('REDIS_PORT', '6379'))
    timeout = float(os.getenv('REDIS_SOCKET_TIMEOUT', '5'))
    ssl = os.getenv('REDIS_SSL', '1').lower() not in {'0','false','no'}

    print("Redis Diagnostics")
    print("Host:", host)
    print("Port:", port)
    print("SSL:", ssl)
    print("Timeout:", timeout)
    print("Python:", sys.version.split()[0])

    any_fail = False

    if args.plain:
        if not test_plain(host, port, timeout, ssl):
            any_fail = True

    if args.entra:
        if not test_entra(host, port, timeout, args.claims, args.force_cli_cred, ssl):
            any_fail = True

    # Reuse plain path for module & other tests unless explicitly using entra
    test_entra_modules = args.entra
    if args.modules:
        test_modules(host, port, timeout, test_entra_modules, ssl)

    if args.index:
        test_index(host, port, timeout, test_entra_modules, ssl)

    if args.json:
        test_json(host, port, timeout, test_entra_modules, ssl)

    if args.ping_loop:
        ping_loop(host, port, timeout, test_entra_modules, ssl, iterations=5, delay=1.0)

    print("\nDone.")
    return 1 if any_fail else 0


if __name__ == '__main__':
    raise SystemExit(main())
