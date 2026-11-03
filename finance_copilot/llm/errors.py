"""Erros de configuração da LLM e tradução das falhas dos provedores em mensagens claras.

Os SDKs dos provedores lançam exceções diferentes para o mesmo problema (o Groq e o OpenRouter
usam o status HTTP, o Gemini usa RESOURCE_EXHAUSTED, o Ollama usa ConnectionError). Este módulo
reduz todas a um `LLMFailure`, usado em dois lugares: na decisão de tentar de novo
(builder.py::_is_transient) e na mensagem mostrada ao usuário (session.py).
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

# Esperas maiores que esta não são feitas automaticamente: o usuário recebe o tempo e decide.
MAX_AUTOMATIC_WAIT_SECONDS = 10.0


class LLMConfigurationError(RuntimeError):
    """Configuração impede criar o modelo. A mensagem é segura para mostrar ao usuário."""


class LLMNotConfiguredError(LLMConfigurationError):
    """Falta a credencial do provedor."""


class UnknownProviderError(LLMConfigurationError):
    pass


class ProviderPackageMissingError(LLMConfigurationError):
    pass


class CostPolicyError(LLMConfigurationError):
    """O modelo não é comprovadamente gratuito e LLM_FREE_ONLY=true."""


class IncompatibleModelError(LLMConfigurationError):
    """O modelo não oferece um recurso essencial do agente, como tool calling."""


FailureKind = Literal[
    "rate_limit",
    "quota",
    "payment",
    "auth",
    "not_found",
    "tool_unsupported",
    "connection",
    "timeout",
    "server",
    "bad_request",
    "unknown",
]

_MESSAGES: dict[str, str] = {
    "rate_limit": (
        "O provedor de IA recusou o pedido por limite de requisições (HTTP 429). "
        "Nenhuma operação foi realizada. Aguarde{wait} e tente de novo."
    ),
    "quota": (
        "A cota do plano gratuito deste provedor de IA acabou (limite diário ou da conta). "
        "Nenhuma operação foi realizada. Aguarde a renovação da cota ou escolha outro modelo "
        "na barra lateral: o Finance Copilot nunca troca de provedor sozinho."
    ),
    "payment": (
        "O provedor de IA exige créditos para atender este pedido (HTTP 402). Nenhuma operação "
        "foi realizada e nenhum modelo pago foi usado no lugar. Confira o saldo da conta: no "
        "OpenRouter, um saldo negativo bloqueia até os modelos gratuitos."
    ),
    "not_found": (
        "O modelo configurado não foi encontrado no provedor (pode ter sido descontinuado ou, "
        "no Ollama, ainda não foi baixado). Confira LLM_MODEL e docs/LLM_PROVIDERS.md."
    ),
    "tool_unsupported": (
        "O modelo selecionado não suporta tool calling, recurso essencial do agente. Nada foi "
        "executado. Escolha um modelo do catálogo com suporte a ferramentas."
    ),
    "bad_request": (
        "O provedor de IA recusou a requisição (HTTP 400). Nenhuma operação foi realizada. "
        "Tente reformular o pedido; se o erro continuar, veja o log do terminal."
    ),
    "connection": (
        "Não foi possível conectar ao servidor do modelo. Se estiver usando Ollama, confirme que "
        "ele está em execução (ollama serve) e que OLLAMA_BASE_URL está correto."
    ),
}


@dataclass(frozen=True)
class LLMFailure:
    kind: FailureKind
    retryable: bool
    retry_after: float | None = None

    @property
    def message(self) -> str | None:
        """Texto para o usuário; None mantém a mensagem genérica de quem chamou."""
        template = _MESSAGES.get(self.kind)
        if template is None:
            return None
        wait = f" cerca de {self.retry_after:.0f} segundos" if self.retry_after else " um minuto"
        return template.format(wait=wait)


def _chain(exc: BaseException):
    """A exceção e as causas encadeadas: o LangChain costuma embrulhar o erro do SDK."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _status(exc: BaseException) -> int | None:
    for item in _chain(exc):
        for attribute in ("status_code", "code"):
            value = getattr(item, attribute, None)
            if isinstance(value, int) and 100 <= value < 600:
                return value
        response = getattr(item, "response", None)
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


_WAIT_PATTERNS = [
    # Groq: "Please try again in 1m2.5s" / "try again in 7.66s"
    re.compile(r"try again in (?:(?P<m>\d+)m)?(?P<s>[\d.]+)s", re.IGNORECASE),
    # Gemini: "'retryDelay': '31s'" ou "Please retry in 31.2s"
    re.compile(r"retry(?:Delay'?:\s*'| in )(?P<s>[\d.]+)s", re.IGNORECASE),
]


def _retry_after(exc: BaseException, text: str) -> float | None:
    for item in _chain(exc):
        headers = getattr(getattr(item, "response", None), "headers", None)
        value = headers.get("retry-after") if headers is not None else None
        if value:
            try:
                return float(value)
            except ValueError:
                pass
    for pattern in _WAIT_PATTERNS:
        match = pattern.search(text)
        if match:
            minutes = float(match.groupdict().get("m") or 0)
            return minutes * 60 + float(match.group("s"))
    return None


# Indícios de cota esgotada, que não volta em segundos: tentar de novo só gasta requisições.
_QUOTA_HINTS = (
    "per day",
    "perday",
    "(rpd)",
    "(tpd)",
    "free-models-per-day",
    "insufficient_quota",
    "quota exceeded for quota metric",
    "credit balance is too low",
)


def classify_llm_error(exc: BaseException) -> LLMFailure:
    text = " ".join(str(item) for item in _chain(exc)).lower()
    status = _status(exc)

    if "does not support tools" in text or "no endpoints found that support tool use" in text:
        return LLMFailure("tool_unsupported", retryable=False)
    # Conexão recusada: nada escuta no endereço (típico do Ollama parado). Repetir não adianta.
    if "failed to connect to ollama" in text or any(
        isinstance(item, ConnectionRefusedError) for item in _chain(exc)
    ):
        return LLMFailure("connection", retryable=False)
    if status == 402:
        return LLMFailure("payment", retryable=False)
    if status == 429 or "resource_exhausted" in text or "rate limit" in text:
        if any(hint in text for hint in _QUOTA_HINTS):
            return LLMFailure("quota", retryable=False)
        wait = _retry_after(exc, text)
        retryable = wait is None or wait <= MAX_AUTOMATIC_WAIT_SECONDS
        return LLMFailure("rate_limit", retryable=retryable, retry_after=wait)
    if "insufficient_quota" in text or "credit balance is too low" in text:
        return LLMFailure("quota", retryable=False)
    if status in (401, 403):
        return LLMFailure("auth", retryable=False)
    if status == 404:
        return LLMFailure("not_found", retryable=False)
    if status in (408, 409):
        return LLMFailure("timeout", retryable=True)
    if status == 498 or (status is not None and status >= 500):
        # 498: capacidade esgotada no Groq (flex tier); 5xx não são cobrados pelos provedores.
        return LLMFailure("server", retryable=True)
    if status is not None and 400 <= status < 500:
        return LLMFailure("bad_request", retryable=False)
    if isinstance(exc, TimeoutError) or "timed out" in text:
        return LLMFailure("timeout", retryable=True)
    return LLMFailure("unknown", retryable=getattr(exc, "is_retryable", True))


def recover_rejected_tool_calls(exc: BaseException) -> list[dict[str, Any]] | None:
    """Recupera as tool calls que o provedor recusou por não seguirem o schema.

    O Groq valida os argumentos no servidor e, se falharem, devolve HTTP 400 `tool_use_failed`
    em vez da resposta, com o que o modelo gerou em `failed_generation`. Os outros provedores
    entregam a chamada, a validação local devolve o erro ao modelo e ele se corrige. Recuperar
    a chamada mantém esse mesmo fluxo; a validação local continua impedindo qualquer gravação.
    """
    for item in _chain(exc):
        body = getattr(item, "body", None)
        error = body.get("error", body) if isinstance(body, dict) else None
        if not isinstance(error, dict) or error.get("code") != "tool_use_failed":
            continue
        try:
            generated = json.loads(error.get("failed_generation") or "")
        except ValueError:
            return None
        items = generated if isinstance(generated, list) else [generated]
        calls = []
        for call in items:
            if not (
                isinstance(call, dict)
                and isinstance(call.get("name"), str)
                and isinstance(call.get("arguments"), dict)
            ):
                return None
            calls.append(
                {"name": call["name"], "args": call["arguments"], "id": f"call_{uuid4().hex}", "type": "tool_call"}
            )
        return calls or None
    return None
