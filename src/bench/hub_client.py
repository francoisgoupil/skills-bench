"""Thin client over the Skore Hub REST API — the read path for monitoring.

Reference: https://api.skore.probabl.ai/docs
The endpoint you linked is `GET /projects/{workspace_public_id}` (list workspace projects).
Confirm the exact base path, auth header, and response shape against /docs — treat the
Swagger page as the source of truth and adjust the methods below if they differ.
"""
from __future__ import annotations

import os

import httpx


class SkoreHub:
    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base_url = (base_url or os.environ["SKORE_HUB_API"]).rstrip("/")
        token = token or os.environ.get("SKORE_HUB_TOKEN", "")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=30)

    def list_projects(self, workspace_public_id: str) -> list[dict]:
        """GET /projects/{workspace_public_id}"""
        r = self._client.get(f"/projects/{workspace_public_id}")
        r.raise_for_status()
        return r.json()

    def get(self, path: str, **params) -> dict | list:
        """Escape hatch for any other documented route."""
        r = self._client.get(path, params=params)
        r.raise_for_status()
        return r.json()


def projects_dataframe(workspace_public_id: str | None = None):
    """Convenience: projects as a DataFrame for the dashboard."""
    import pandas as pd
    wid = workspace_public_id or os.environ["SKORE_WORKSPACE_PUBLIC_ID"]
    return pd.json_normalize(SkoreHub().list_projects(wid))
