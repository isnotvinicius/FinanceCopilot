# Finance Copilot

Aplicação local de finanças pessoais que combina um dashboard interativo com um agente de IA. O agente interpreta linguagem natural, escolhe ferramentas, consulta e altera dados reais em SQLite e sempre pede confirmação antes de gravar.

```text
Você:    Gastei 85 reais com pizza ontem
Agente:  [propõe create_transaction → R$ 85,00 · Pizza · Alimentação · 30/10/2026]
Você:    [clica em Aprovar]
Agente:  Registrado! (o dashboard atualiza na mesma tela)
```

A LLM interpreta o pedido, escolhe as ferramentas e explica os resultados. **Toda conta é feita em Python ou SQL.** Os dados são fictícios e não há integração bancária.

## Stack

| Item | Uso |
|---|---|
| Python 3.12.3 | Linguagem |
| LangChain 1.4 (`create_agent`) | Agente, tools, middleware, structured output |
| LangGraph 1.2 + `langgraph-checkpoint-sqlite` | Estado, checkpoints, interrupções |
| SQLite | Dados financeiros e, em outro arquivo, estado do agente |
| Streamlit + Plotly | Dashboard e chat |
| Pydantic | Validação e schemas |
| Pytest | Testes (sem rede nem credenciais) |
| LangSmith | Tracing opcional |

## Instalação

```bash
cd FinanceCopilot
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Se o primeiro comando reclamar de `ensurepip` (Ubuntu/WSL sem `python3-venv`) e você não quiser usar `sudo apt install python3.12-venv`, crie o ambiente assim:

```bash
python3 -m venv --clear --without-pip .venv
pip --python .venv/bin/python install pip
.venv/bin/python -m pip install -r requirements.txt
```

## Configuração

```bash
cp .env.example .env
```

O padrão é o **Groq no plano gratuito**. Crie uma chave em <https://console.groq.com/keys> e preencha:

```ini
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
LLM_FREE_ONLY=true
GROQ_API_KEY=gsk_...
```

Provedores suportados: `groq`, `google_genai` (Gemini), `openrouter`, `ollama` (local, sem chave), `anthropic` e `openai`. Para trocar, mude só o `.env` ou escolha na barra lateral da aplicação. Como criar chaves gratuitas, modelos testados, limites e solução de erros: **[docs/LLM_PROVIDERS.md](docs/LLM_PROVIDERS.md)**.

**Custo:** com `LLM_FREE_ONLY=true` (padrão), só os modelos marcados como gratuitos ou locais em `finance_copilot/llm/models.toml` podem ser usados; modelos pagos ou de custo não verificado são bloqueados, e o app nunca troca de modelo ou de provedor sozinho. Ainda assim, ele não controla o faturamento da sua conta no provedor: mantenha a conta na camada gratuita (sem cartão cadastrado, quando possível). Para usar Anthropic ou OpenAI de propósito, defina `LLM_FREE_ONLY=false`.

Sem um modelo válido, o dashboard e os lançamentos manuais funcionam normalmente; apenas o chat e o "Preencher com IA" ficam inativos, com o motivo na tela.

**Tracing (opcional):** defina `LANGSMITH_TRACING=true` e `LANGSMITH_API_KEY` no `.env`. As mensagens e os resultados das tools são enviados ao LangSmith quando o tracing está ligado.

## Execução

```bash
.venv/bin/python -m finance_copilot.seed   # dados fictícios (pode repetir: não duplica)
.venv/bin/python -m streamlit run app.py
```

Abra <http://localhost:8501>. Para recomeçar do zero, apague `data/finance.db` e `data/agent_state.db` com a aplicação parada.

## Testes

```bash
.venv/bin/python -m pytest          # não usa rede nem credenciais
.venv/bin/python -m pytest -m live  # usa a LLM real e consome cota/créditos; exige chave no .env
.venv/bin/python scripts/smoke_llm.py   # smoke test real e curto do modelo do .env (pede confirmação)
```

| Arquivo | O que cobre |
|---|---|
| `tests/test_money.py` | Conversão e formatação de dinheiro |
| `tests/test_service.py` | Saldo, categorias, comparações, simulações, validação, idempotência, seed |
| `tests/test_agent.py` | Grafo real com LLM roteirizada: tools, aprovação, duplicação, erros, memória, structured output |
| `tests/test_llm.py` | Clientes reais dos 6 provedores contra um servidor HTTP local (aprovação, 429, cota, 402); fábrica, catálogo e modo gratuito |
| `tests/test_llm_errors.py` | Erros de cada SDK traduzidos em retry e mensagem ao usuário |
| `tests/test_ui.py` | Página real do Streamlit via `AppTest`, incluindo o fluxo mínimo completo |
| `tests/test_docs.py` | Arquivos e símbolos citados na documentação existem |
| `tests/test_llm_live.py` | Escolha de tools por um modelo real (só com `-m live`) |

## Exemplos de uso

**Registrar e alterar** (sempre com confirmação):

- Gastei 85 reais com pizza ontem
- Recebi 800 de um freela hoje
- Meu salário agora é 7 mil
- Muda o valor da pizza para 80
- Apaga o lançamento da pizza

**Consultar:**

- Quanto dinheiro eu tenho disponível neste mês?
- Qual categoria representa minha maior despesa?
- Gastei mais com alimentação neste mês ou no mês passado?
- Quais categorias mais contribuíram para o aumento dos meus gastos?
- Quanto gastei com alimentação em setembro? → E em outubro?

**Simular** (sem gravar, identificado na interface):

- Quanto economizaria reduzindo o lazer em 20%?
- Se eu economizar R$ 300 por mês, quanto terei acumulado em seis meses?

## Estrutura

```text
app.py                     Entrada do Streamlit
finance_copilot/
├── config.py              Variáveis de ambiente, fuso, LangSmith
├── llm/                   Único ponto que conhece os provedores de LLM
│   ├── factory.py         init_chat_model a partir da configuração
│   ├── providers.py       Particularidades de cada provedor
│   ├── catalog.py         Catálogo de modelos e política de custo (LLM_FREE_ONLY)
│   ├── models.toml        Modelos, custo e recursos verificados
│   └── errors.py          Rate limit, cota e outras falhas em mensagens claras
├── bootstrap.py           Monta os objetos da aplicação
├── money.py               Centavos, parsing e formatação em BRL
├── schemas.py             Entradas e resultados dos serviços (Pydantic)
├── errors.py              Erros de domínio
├── database.py            Conexão e esquema SQLite
├── repository.py          Consultas SQL parametrizadas
├── service.py             Regras financeiras e cálculos
├── seed.py                Dados fictícios reproduzíveis
├── agent/
│   ├── builder.py         create_agent, middlewares, checkpointer
│   ├── tools.py           As 9 tools (@tool)
│   ├── tool_schemas.py    Argumentos das tools
│   ├── approvals.py       Política de aprovação e texto de confirmação
│   ├── prompts.py         System prompt e prompt de extração
│   ├── context.py         Contexto de execução
│   ├── session.py         Fronteira entre interface e grafo
│   └── extraction.py      Structured output do preenchimento rápido
└── ui/                    Página, gráficos, formulários, chat e seleção de modelo
scripts/smoke_llm.py       Smoke test opcional com a API real
tests/                     Testes automatizados
docs/                      Documentação abaixo
```

## Documentação

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): arquitetura, fluxos e decisões
- [docs/LLM_PROVIDERS.md](docs/LLM_PROVIDERS.md): provedores de LLM, planos gratuitos, limites e smoke tests
- [docs/LANGCHAIN_GUIDE.md](docs/LANGCHAIN_GUIDE.md): conceitos de LangChain e LangGraph, apontando para o código
- [docs/INTERVIEW.md](docs/INTERVIEW.md): guia de estudo para a entrevista e roteiro de demonstração
