"""Erros de domínio. A mensagem é segura para exibir ao usuário e para devolver à LLM."""


class FinanceError(Exception):
    """Base dos erros de regra de negócio."""


class InvalidOperationError(FinanceError):
    """A operação viola uma regra (valor, data, categoria inválidos etc.)."""


class NotFoundError(FinanceError):
    """O registro solicitado não existe."""
