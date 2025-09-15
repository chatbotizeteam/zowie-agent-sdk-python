from __future__ import annotations

import json as libJson
import logging
from typing import Any, Dict, List, Optional

import requests

from .protocol import APICallEvent, APICallEventPayload, Event
from .utils import get_time_ms


class HTTPClient:
    events: List[Event]
    default_timeout_seconds: float

    def __init__(
        self, 
        events: List[Event], 
        default_timeout_seconds: float = 10.0, 
        include_headers_by_default: bool = True
    ):
        self.events = events
        self.default_timeout_seconds = default_timeout_seconds
        self.include_headers_by_default = include_headers_by_default
        self.logger = logging.getLogger("zowie_agent.HTTPClient")

    def _request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        json: Optional[Any] = None,
        timeout_seconds: Optional[float] = None,
        include_headers: Optional[bool] = None,
    ) -> requests.Response:
        should_include_headers = (
            include_headers if include_headers is not None 
            else self.include_headers_by_default
        )
        
        timeout = timeout_seconds or self.default_timeout_seconds
        self.logger.debug(f"Making {method} request to {url} with timeout {timeout}s")
        
        start = get_time_ms()
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json,
                timeout=timeout,
            )
            stop = get_time_ms()
            duration = stop - start

            self.logger.debug(
                f"{method} {url} completed: {response.status_code} in {duration}ms"
            )

            self.events.append(
                APICallEvent(
                    payload=APICallEventPayload(
                        url=url,
                        requestMethod=method,
                        requestHeaders=headers if should_include_headers else {},
                        requestBody=(libJson.dumps(json) if json is not None else None),
                        responseHeaders=dict(response.headers) if should_include_headers else {},
                        responseStatusCode=response.status_code,
                        responseBody=response.text,
                        durationInMillis=duration,
                    )
                )
            )
            return response
        except requests.exceptions.Timeout as e:
            stop = get_time_ms()
            duration = stop - start
            
            self.logger.warning(
                f"{method} {url} timed out after {duration}ms (timeout: {timeout}s): {str(e)}"
            )
            
            self.events.append(
                APICallEvent(
                    payload=APICallEventPayload(
                        url=url,
                        requestMethod=method,
                        requestHeaders=headers if should_include_headers else {},
                        requestBody=(libJson.dumps(json) if json is not None else None),
                        responseHeaders={},
                        responseStatusCode=504,
                        responseBody=str(e),
                        durationInMillis=duration,
                    )
                )
            )
            raise
        except requests.RequestException as e:
            stop = get_time_ms()
            duration = stop - start
            resp = getattr(e, "response", None)
            status_code = resp.status_code if resp is not None else 0
            
            self.logger.error(
                f"{method} {url} failed after {duration}ms "
                f"(status: {status_code}): {str(e)}"
            )
            
            self.events.append(
                APICallEvent(
                    payload=APICallEventPayload(
                        url=url,
                        requestMethod=method,
                        requestHeaders=headers if should_include_headers else {},
                        requestBody=(libJson.dumps(json) if json is not None else None),
                        responseHeaders=(
                            dict(resp.headers) 
                            if resp is not None and should_include_headers 
                            else {}
                        ),
                        responseStatusCode=status_code,
                        responseBody=(resp.text if resp is not None else str(e)),
                        durationInMillis=duration,
                    )
                )
            )
            raise

    def get(
        self, 
        url: str, 
        headers: Dict[str, str], 
        timeout_seconds: Optional[float] = None, 
        include_headers: Optional[bool] = None
    ) -> requests.Response:
        return self._request("GET", url, headers, None, timeout_seconds, include_headers)

    def post(
        self, 
        url: str, 
        json: Any, 
        headers: Dict[str, str], 
        timeout_seconds: Optional[float] = None, 
        include_headers: Optional[bool] = None
    ) -> requests.Response:
        return self._request("POST", url, headers, json, timeout_seconds, include_headers)

    def put(
        self, 
        url: str, 
        json: Any, 
        headers: Dict[str, str], 
        timeout_seconds: Optional[float] = None, 
        include_headers: Optional[bool] = None
    ) -> requests.Response:
        return self._request("PUT", url, headers, json, timeout_seconds, include_headers)

    def patch(
        self, 
        url: str, 
        json: Any, 
        headers: Dict[str, str], 
        timeout_seconds: Optional[float] = None, 
        include_headers: Optional[bool] = None
    ) -> requests.Response:
        return self._request("PATCH", url, headers, json, timeout_seconds, include_headers)

    def delete(
        self, 
        url: str, 
        headers: Dict[str, str], 
        timeout_seconds: Optional[float] = None, 
        include_headers: Optional[bool] = None
    ) -> requests.Response:
        return self._request("DELETE", url, headers, None, timeout_seconds, include_headers)
