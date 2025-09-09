from __future__ import annotations

import base64
import secrets
from typing import Optional

from fastapi import HTTPException, Request, status

from .types import APIKeyAuth, AuthConfig, BasicAuth, BearerTokenAuth


class AuthValidator:
    def __init__(self, auth_config: Optional[AuthConfig] = None):
        self.auth_config = auth_config

    def __call__(self, request: Request) -> None:
        if self.auth_config is None:
            return

        if isinstance(self.auth_config, APIKeyAuth):
            self._verify_api_key(request)
        elif isinstance(self.auth_config, BasicAuth):
            self._verify_basic_auth(request)
        elif isinstance(self.auth_config, BearerTokenAuth):
            self._verify_bearer_token(request)

    def _verify_api_key(self, request: Request) -> None:
        auth_config = self.auth_config
        assert isinstance(auth_config, APIKeyAuth)

        header_value = request.headers.get(auth_config.header_name)
        if not header_value:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Missing {auth_config.header_name} header",
            )

        if not secrets.compare_digest(header_value, auth_config.api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

    def _verify_basic_auth(self, request: Request) -> None:
        auth_config = self.auth_config
        assert isinstance(auth_config, BasicAuth)

        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Basic "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid Authorization header",
            )

        try:
            encoded_credentials = authorization.split(" ", 1)[1]
            decoded = base64.b64decode(encoded_credentials).decode("utf-8")
            username, password = decoded.split(":", 1)
        except (ValueError, UnicodeDecodeError) as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Basic auth format",
            ) from e

        if not (
            secrets.compare_digest(username, auth_config.username)
            and secrets.compare_digest(password, auth_config.password)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )

    def _verify_bearer_token(self, request: Request) -> None:
        auth_config = self.auth_config
        assert isinstance(auth_config, BearerTokenAuth)

        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid Authorization header",
            )

        token = authorization.split(" ", 1)[1]
        if not secrets.compare_digest(token, auth_config.token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer token",
            )
