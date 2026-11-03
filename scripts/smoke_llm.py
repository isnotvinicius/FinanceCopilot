"""Smoke test com a API real do provedor configurado. Só roda quando chamado explicitamente.

Faz cerca de 10 chamadas ao modelo (consome cota do plano gratuito) usando um banco temporário com
dados fictícios: seus dados em data/ não são tocados. Nunca roda no pytest.

    .venv/bin/python scripts/smoke_llm.py                       # modelo do .env
    .venv/bin/python scripts/smoke_llm.py --provider groq --model openai/gpt-oss-20b
    .venv/bin/python scripts/smoke_llm.py --steps chat,read     # só alguns passos

Passos:
  chat        mensagem simples, sem ferramentas
  read        ferramenta de leitura (get_financial_summary)
  structured  ferramenta com parâmetros estruturados (simulate_budget com lista de ajustes)
  write       escrita com aprovação humana: pausa, banco intacto, aprovação, gravação
  extract     structured output do "Preencher com IA"

Um passo que falha mostra o motivo; nada é simulado. Se tudo passar, você pode marcar o modelo
como tested = "live" em finance_copilot/llm/models.toml.
"""

import argparse
import sys
import tempfile
import time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finance_copilot.agent.builder import build_agent, open_checkpointer  # noqa: E402
from finance_copilot.agent.extraction import extract_transaction_draft  # noqa: E402
from finance_copilot.agent.session import AgentSession  # noqa: E402
from finance_copilot.config import get_settings  # noqa: E402
from finance_copilot.database import Database  # noqa: E402
from finance_copilot.llm import LLMConfigurationError, create_chat_model  # noqa: E402
from finance_copilot.llm.catalog import COST_LABELS  # noqa: E402
from finance_copilot.schemas import TransactionFilter  # noqa: E402
from finance_copilot.seed import seed  # noqa: E402
from finance_copilot.service import FinanceService  # noqa: E402

STEPS = ["chat", "read", "structured", "write", "extract"]


class StepFailed(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise StepFailed(message)


def last_entry(session: AgentSession, thread: str):
    return session.history(thread)[-1]


def send(session: AgentSession, thread: str, text: str):
    result = session.send(thread, text)
    check(result.error is None, f"erro do agente: {result.error}")
    return last_entry(session, thread)


def step_chat(session, service, bundle, thread):
    entry = send(session, thread, "Diga apenas 'ok' para confirmar que está funcionando.")
    check(bool(entry.text.strip()), "o modelo não respondeu texto")
    return f"resposta: {entry.text.strip()[:60]!r}"


def step_read(session, service, bundle, thread):
    entry = send(session, thread, "Quanto dinheiro eu tenho disponível neste mês?")
    used = {(e.name, e.status) for e in entry.tool_events}
    check(("get_financial_summary", "ok") in used, f"get_financial_summary não foi usada (tools: {used})")
    return f"tools: {sorted(name for name, _ in used)}"


def step_structured(session, service, bundle, thread):
    entry = send(
        session,
        thread,
        "Simule: quanto eu economizaria em 6 meses reduzindo Lazer em 20% e Alimentação em 10%?",
    )
    events = [e for e in entry.tool_events if e.name == "simulate_budget"]
    check(bool(events), f"simulate_budget não foi usada (tools: {[e.name for e in entry.tool_events]})")
    event = events[-1]
    check(event.status == "ok", f"simulate_budget falhou: {event.result}")
    adjustments = event.args.get("adjustments") or []
    check(len(adjustments) >= 2, f"esperava 2 ajustes estruturados, recebeu {adjustments}")
    return f"args: {event.args}"


def step_write(session, service, bundle, thread):
    before = service.list_transactions(TransactionFilter(limit=500)).total_count
    result = session.send(thread, "Gastei 42,50 com padaria hoje.")
    check(result.error is None, f"erro do agente: {result.error}")
    pending = session.pending_approval(thread)
    check(pending is not None, "o modelo não propôs a escrita (nenhuma aprovação pendente)")
    action = pending.actions[0]
    check(action.name == "create_transaction", f"ferramenta inesperada: {action.name}")
    check(service.list_transactions(TransactionFilter(limit=500)).total_count == before, "gravou antes da aprovação!")
    result = session.decide(thread, pending.interrupt_id, approve=True)
    check(result.error is None, f"erro após aprovar: {result.error}")
    after = service.list_transactions(TransactionFilter(limit=500)).total_count
    check(after == before + 1, f"esperava 1 lançamento novo, há {after - before}")
    return f"args aprovados: {action.args}"


def step_extract(session, service, bundle, thread):
    if not bundle.spec.supports_structured_output:
        raise StepFailed("o catálogo marca este modelo sem structured output")
    draft = extract_transaction_draft(
        bundle.model, service, "Uber de 32,90 hoje de manhã", method=bundle.spec.structured_output_method
    )
    check(draft.amount == Decimal("32.90"), f"valor extraído incorreto: {draft}")
    return f"rascunho: {draft.model_dump(mode='json')}"


RUNNERS = {
    "chat": step_chat,
    "read": step_read,
    "structured": step_structured,
    "write": step_write,
    "extract": step_extract,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", help="sobrescreve LLM_PROVIDER")
    parser.add_argument("--model", help="sobrescreve LLM_MODEL")
    parser.add_argument("--steps", default=",".join(STEPS), help=f"lista separada por vírgulas: {','.join(STEPS)}")
    parser.add_argument(
        "--pause", type=float, default=20.0, help="segundos entre passos, para respeitar limites por minuto"
    )
    parser.add_argument("--yes", action="store_true", help="não pede confirmação antes de chamar a API")
    args = parser.parse_args()

    settings = get_settings()
    if args.provider or args.model:
        settings = settings.with_llm(args.provider or settings.llm_provider, args.model or settings.llm_model)
    try:
        bundle = create_chat_model(settings)
    except LLMConfigurationError as exc:
        print(f"Configuração inválida: {exc}")
        return 2

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        print(f"Passos desconhecidos: {sorted(unknown)}")
        return 2

    spec = bundle.spec
    print(f"Modelo: {spec.provider} / {spec.model} ({COST_LABELS[spec.cost]})")
    print(f"Passos: {', '.join(steps)} · ~{2 * len(steps)} chamadas ao modelo")
    if not args.yes and input("Chamar a API real agora? [s/N] ").strip().lower() not in ("s", "sim", "y"):
        print("Cancelado.")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        service = FinanceService(Database(Path(tmp) / "finance.db"))
        seed(service)
        saver = open_checkpointer(Path(tmp) / "state.db")
        session = AgentSession(build_agent(bundle.model, saver), service)
        results = []
        for index, name in enumerate(steps):
            if index:
                time.sleep(args.pause)
            started = time.monotonic()
            try:
                detail = RUNNERS[name](session, service, bundle, thread=f"smoke-{name}")
                status = "PASSOU"
            except StepFailed as exc:
                detail, status = str(exc), "FALHOU"
            except Exception as exc:  # o smoke test relata qualquer falha, sem esconder
                detail, status = f"{type(exc).__name__}: {exc}", "FALHOU"
            elapsed = time.monotonic() - started
            results.append((name, status))
            print(f"[{status}] {name} ({elapsed:.1f}s) {detail}")
        saver.conn.close()

    failed = [name for name, status in results if status != "PASSOU"]
    print("\nResumo:", "todos os passos passaram." if not failed else f"falharam: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
