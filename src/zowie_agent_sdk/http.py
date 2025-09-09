from __future__ import annotations

import json as libJson
from typing import Any, Dict, List, Optional

import requests

from .types import APICallEvent, APICallEventPayload, Event
from .utils import get_time_ms


class HTTPClient:
    events: List[Event]
    default_timeout_seconds: float

    def __init__(self, events: List[Event], default_timeout_seconds: float = 10.0):
        self.events = events
        self.default_timeout_seconds = default_timeout_seconds

    def _request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        json: Optional[Any] = None,
        timeout_seconds: Optional[float] = None,
    ) -> requests.Response:
        start = get_time_ms()
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json,
                timeout=(timeout_seconds or self.default_timeout_seconds),
            )
            stop = get_time_ms()

            self.events.append(
                APICallEvent(
                    payload=APICallEventPayload(
                        url=url,
                        requestMethod=method,
                        requestHeaders=headers,
                        requestBody=(libJson.dumps(json) if json is not None else None),
                        responseHeaders=dict(response.headers),
                        responseStatusCode=response.status_code,
                        responseBody=response.text,
                        durationInMillis=stop - start,
                    )
                )
            )
            return response
        except requests.exceptions.Timeout as e:
            stop = get_time_ms()
            self.events.append(
                APICallEvent(
                    payload=APICallEventPayload(
                        url=url,
                        requestMethod=method,
                        requestHeaders=headers,
                        requestBody=(libJson.dumps(json) if json is not None else None),
                        responseHeaders={},
                        responseStatusCode=504,
                        responseBody=str(e),
                        durationInMillis=stop - start,
                    )
                )
            )
            raise
        except requests.RequestException as e:
            stop = get_time_ms()
            resp = getattr(e, "response", None)
            self.events.append(
                APICallEvent(
                    payload=APICallEventPayload(
                        url=url,
                        requestMethod=method,
                        requestHeaders=headers,
                        requestBody=(libJson.dumps(json) if json is not None else None),
                        responseHeaders=dict(resp.headers) if resp is not None else {},
                        responseStatusCode=resp.status_code if resp is not None else 0,
                        responseBody=(resp.text if resp is not None else str(e)),
                        durationInMillis=stop - start,
                    )
                )
            )
            raise

    def get(
        self, url: str, headers: Dict[str, str], timeout_seconds: Optional[float] = None
    ) -> requests.Response:
        return self._request("GET", url, headers, None, timeout_seconds)

    def post(
        self, url: str, json: Any, headers: Dict[str, str], timeout_seconds: Optional[float] = None
    ) -> requests.Response:
        return self._request("POST", url, headers, json, timeout_seconds)

    def put(
        self, url: str, json: Any, headers: Dict[str, str], timeout_seconds: Optional[float] = None
    ) -> requests.Response:
        return self._request("PUT", url, headers, json, timeout_seconds)

    def delete(
        self, url: str, headers: Dict[str, str], timeout_seconds: Optional[float] = None
    ) -> requests.Response:
        return self._request("DELETE", url, headers, None, timeout_seconds)
