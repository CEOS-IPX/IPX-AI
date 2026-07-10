"""
============================================================
외부 LLM API 재시도 유틸
============================================================
Claude/Gemini 호출 시 일시적 오류(429 rate limit, 5xx, 529 overloaded,
타임아웃/연결 오류)에 한해 짧은 backoff로 재시도한다.
그 외 4xx(잘못된 요청, 인증 오류 등)는 재시도해도 결과가 같으므로
바로 실패 처리한다.

검색 파이프라인 안에서 동기적으로 기다리는 호출이므로 재시도 횟수와
대기 시간을 짧게 유지한다 (최대 2회 재시도, 총 지연 2초 이내).
============================================================
"""

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3  # 최초 1회 + 재시도 2회
_BACKOFF_SECONDS = [0.5, 1.5]
_RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}  # 529: Claude "overloaded_error"


async def post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    log_prefix: str = "",
    **kwargs,
) -> httpx.Response:
    """
    httpx POST 요청을 보내고, 일시적 오류 시 재시도한다.

    Args:
        client: 재사용할 httpx.AsyncClient
        url: 요청 URL
        log_prefix: 재시도 로그에 붙일 접두사 (예: "[HyDE]")
        **kwargs: client.post에 그대로 전달 (headers, json, params 등)

    Returns:
        성공한 httpx.Response (status_code 2xx, raise_for_status 통과)

    Raises:
        httpx.HTTPError: 재시도 소진 후에도 실패한 경우 (기존 호출부의
            except httpx.HTTPError 처리를 그대로 재사용할 수 있도록 그대로 전파)
    """
    last_exc: httpx.HTTPError

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = await client.post(url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            last_exc = e
            retryable = e.response.status_code in _RETRYABLE_STATUS
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_exc = e
            retryable = True

        if not retryable or attempt == _MAX_ATTEMPTS:
            raise last_exc

        delay = _BACKOFF_SECONDS[attempt - 1]
        logger.warning(
            f"{log_prefix} 일시적 오류로 재시도 ({attempt}/{_MAX_ATTEMPTS - 1}), "
            f"{delay}초 대기: {last_exc}"
        )
        await asyncio.sleep(delay)

    raise last_exc
