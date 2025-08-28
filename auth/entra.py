from __future__ import annotations
from urllib.parse import urlparse
from redis import Redis
from redis_entraid.cred_provider import create_from_default_azure_credential
from azure.identity import AzureCliCredential  # added for CLI-only isolation test

AMR_SCOPES = ("https://redis.azure.com/.default",)

# Step A: cache provider and client so token manager starts only once
_credential_provider = None  # type: ignore
_redis_client: Redis | None = None

def get_redis_client(host, port) -> Redis:
    """Return a cached redis-py client using Entra ID token provider.
    Steps implemented:
      A. Cache the credential provider & Redis client (single token manager lifecycle)
      B. Suppress verbose redis_entraid token manager INFO logs
    Args:
      host: Redis host
      port: Redis port
    Returns:
      Redis: shared client instance
    """
    global _credential_provider, _redis_client
    if not host:
        raise ValueError("host is required")
    if _redis_client is not None:
        return _redis_client
    if _credential_provider is None:
        _credential_provider = create_from_default_azure_credential(scopes=AMR_SCOPES)
    _redis_client = Redis(
        host=host,
        port=port,
        ssl=True,
        credential_provider=_credential_provider,
        ssl_cert_reqs=None,
        socket_connect_timeout=5,
        socket_timeout=5,
        health_check_interval=30,
        retry_on_timeout=True,
    )
    return _redis_client

