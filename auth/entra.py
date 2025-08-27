from __future__ import annotations
import os
from urllib.parse import urlparse
from redis import Redis
from redis_entraid.cred_provider import (
    create_from_default_azure_credential,
    create_from_managed_identity
)

AMR_SCOPES = ("https://redis.azure.com/.default",)

def _provider_for_local():
    """
    Local dev: uses DefaultAzureCredential internally (works with `az login`).
    Make sure your user is added to AMR Entra access policies.
    """
    return create_from_default_azure_credential(scopes=AMR_SCOPES)

def _provider_for_managed_identity():
    """
    Deployed in Azure:
      - System-assigned MI: leave AZURE_CLIENT_ID unset.
      - User-assigned MI: set AZURE_CLIENT_ID to the MI's client id.
    Ensure the MI is added to AMR Entra access policies.
    """
    client_id = os.getenv("AZURE_CLIENT_ID")  # only when using user-assigned MI
    return create_from_managed_identity(scopes=AMR_SCOPES, client_id=client_id)
    

def get_redis_client(host, port) -> Redis:
    """
    Returns a redis-py client.
      - AMR endpoints use Entra ID credential provider (token-based, no password).
      - Non-AMR endpoints fall back to plain Redis.from_url behavior.
    Env toggle:
      - USE_MANAGED_IDENTITY=true/1/yes -> use managed identity factory
      - otherwise -> use default Azure credential (CLI user for local)
    """
    if not host:
        raise ValueError("host is required")

    use_mi = os.getenv("USE_MANAGED_IDENTITY", "").lower() in {"1","true","yes"}
    provider = _provider_for_managed_identity() if use_mi else _provider_for_local()
    return Redis(host=host, port=port, ssl=True, credential_provider=provider)

