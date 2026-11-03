"""Barra lateral: escolha do provedor e do modelo, custo e status da configuração.

O padrão é o modelo do .env. A escolha vale para a sessão do navegador e fica na URL, sem alterar
o .env. As opções vêm do catálogo (llm/models.toml); no modo gratuito, os pagos nem aparecem.
"""

import streamlit as st

from finance_copilot.bootstrap import App
from finance_copilot.config import Settings
from finance_copilot.llm.catalog import COST_LABELS, ModelSpec, find_model, selectable_models
from finance_copilot.llm.providers import PROVIDERS

FREE_DISCLAIMER = (
    "\"Gratuito\" reflete a documentação do provedor na data indicada no catálogo, não uma "
    "garantia: disponibilidade, limites e preços dependem do provedor e da sua conta. O app não "
    "controla o faturamento da conta externa."
)


def _options(settings: Settings) -> list[tuple[str, str]]:
    options = [(spec.provider, spec.model) for spec in selectable_models(settings)]
    configured = (settings.llm_provider, settings.llm_model)
    # O modelo do .env sempre aparece, mesmo bloqueado: assim o motivo do bloqueio fica visível,
    # em vez de outro modelo ser escolhido sem o usuário pedir.
    if configured not in options:
        options.insert(0, configured)
    return options


def _model_label(settings: Settings, provider: str, model: str) -> str:
    spec = find_model(settings, provider, model)
    if spec is None:
        return f"{model} (do .env)"
    star = "★ " if spec.recommended else ""
    return f"{star}{spec.display_name} · {COST_LABELS[spec.cost]}"


def choose_model(settings: Settings) -> tuple[str, str]:
    options = _options(settings)
    configured = (settings.llm_provider, settings.llm_model)
    from_url = (st.query_params.get("provider"), st.query_params.get("model"))
    current = from_url if from_url in options else configured

    providers = list(dict.fromkeys(provider for provider, _ in options))
    with st.sidebar:
        st.subheader("Modelo de IA", anchor=False)
        provider = st.selectbox(
            "Provedor",
            providers,
            index=providers.index(current[0]),
            format_func=lambda p: PROVIDERS[p].label if p in PROVIDERS else p,
        )
        models = [model for p, model in options if p == provider]
        default = current[1] if current[0] == provider else models[0]
        model = st.selectbox(
            "Modelo",
            models,
            index=models.index(default),
            format_func=lambda m: _model_label(settings, provider, m),
            key=f"model_for_{provider}",
        )
    if (provider, model) == configured:
        for name in ("provider", "model"):
            if name in st.query_params:
                del st.query_params[name]
    else:
        st.query_params["provider"] = provider
        st.query_params["model"] = model
    return provider, model


def _render_spec(spec: ModelSpec) -> None:
    st.markdown(f"**Ativo:** {spec.display_name}")
    st.caption(f"`{spec.provider}` · `{spec.model}` · {COST_LABELS[spec.cost]}")
    details = []
    if spec.parallel_tool_calls is False:
        details.append("uma ferramenta por vez")
    if not spec.supports_structured_output:
        details.append("sem structured output")
    if spec.tested != "live":
        details.append("ainda não validado com a API real")
    if details:
        st.caption("Observações: " + "; ".join(details) + ".")
    if spec.notes:
        st.caption(spec.notes)


def render_status(app: App) -> None:
    with st.sidebar:
        if app.model_spec is not None:
            _render_spec(app.model_spec)
        if app.llm_error:
            st.warning(app.llm_error)
        if app.settings.llm_free_only:
            st.caption("Modo gratuito ativo (LLM_FREE_ONLY=true): modelos pagos ficam ocultos e bloqueados.")
        st.caption(FREE_DISCLAIMER)
