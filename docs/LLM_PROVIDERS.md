# Provedores de LLM

O Finance Copilot funciona com seis provedores: **Groq**, **Google Gemini**, **OpenRouter**, **Ollama** (local), **Anthropic** e **OpenAI**. O foco é desenvolver e demonstrar o projeto **sem gastar com API**, usando planos gratuitos ou modelos locais.

> **Situação da validação (03/11/2026):** todos os provedores foram testados com o **cliente LangChain real contra um servidor HTTP simulado** (`tests/test_llm.py`), incluindo tool calling, aprovação humana, rate limit e cota. Com a **API real**, só o **Groq · `openai/gpt-oss-120b`** foi validado: o smoke test passou nos 5 passos (texto, leitura, parâmetros estruturados, escrita aprovada, structured output). Os demais ainda não, por falta de chave ou de Ollama no ambiente. Antes de uma demonstração com outro provedor, rode o [smoke test](#smoke-test-com-a-api-real).

## Resumo: qual usar

| Situação | Provedor e modelo | Por quê |
|---|---|---|
| Padrão para desenvolver e demonstrar | `groq` · `openai/gpt-oss-120b` | Plano gratuito sem cartão, muito rápido, tool calling e structured output documentados |
| Alternativa na nuvem | `google_genai` · `gemini-3.5-flash-lite` | Camada gratuita com cota diária maior por requisição; suporta chamadas paralelas |
| Testar muitos modelos gratuitos | `openrouter` · `openrouter/free` | Um roteador escolhe um modelo gratuito com suporte a ferramentas a cada chamada |
| Sem internet, sem limites de cota | `ollama` · `qwen3:8b` | Roda na sua máquina; precisa de ~8 GB de RAM livres |

## Custo: o que o app garante e o que não garante

Com `LLM_FREE_ONLY=true` (padrão):

- Só passam os modelos marcados com `cost = "free"` ou `cost = "local"` em `finance_copilot/llm/models.toml`. **Nada é deduzido do nome:** um modelo fora do catálogo é bloqueado, mesmo que seja de um provedor com plano gratuito.
- O carregamento do catálogo recusa marcações impossíveis: Anthropic/OpenAI como gratuitos, IDs do OpenRouter sem `:free` (exceto `openrouter/free`) e modelos `-cloud` do Ollama como locais.
- Modelos pagos não aparecem na barra lateral.
- **Não há fallback:** se o modelo configurado falhar ou for bloqueado, o chat mostra o motivo. O app nunca troca de modelo ou de provedor sozinho, e nunca envia ao OpenRouter uma lista de modelos alternativos.
- **Retries limitados:** os retries internos dos clientes são desligados (o do Gemini faria 6). No máximo 2 novas tentativas por chamada, com espera de 2 s e 4 s, e só para falhas temporárias. Cota diária esgotada, falta de créditos (402), chave inválida e rate limit com espera maior que 10 s não são repetidos.
- **Limite por pergunta:** no máximo 10 chamadas ao modelo por turno (`ModelCallLimitMiddleware`).
- O dashboard e os gráficos nunca chamam a LLM; o "Preencher com IA" só chama quando o botão é clicado.

O que o app **não** garante: o faturamento da sua conta. Preços, limites e a lista de modelos gratuitos mudam sem aviso, e a marcação "gratuito" reflete a documentação do provedor na data `checked_on` do catálogo. Cabe a você manter a conta na camada gratuita:

- **Groq:** não faça upgrade para o plano Developer (que exige cartão).
- **Gemini:** não ative o Cloud Billing no projeto da chave; sem billing, o projeto fica na camada gratuita.
- **OpenRouter:** não deixe saldo negativo (ele bloqueia até os modelos gratuitos). Comprar US$ 10 em créditos aumenta o limite diário dos modelos `:free`, mas não é necessário.
- **Ollama:** use só modelos locais. Modelos com sufixo `-cloud` rodam nos servidores do Ollama.

Para usar Anthropic ou OpenAI de propósito, defina `LLM_FREE_ONLY=false`. Esses modelos consomem créditos.

## Tabela comparativa

Legenda: ✅ suportado e testado (com servidor simulado; com a API real quando a última coluna diz Sim) · 📄 documentado pelo provedor, não testado com a API real · ⚠️ com ressalvas · — não se aplica.

| Provedor · modelo | Custo | Tool calling | Chamadas paralelas | Structured output (método) | Memória do agente | Fluxo de confirmação | Validado com API real |
|---|---|---|---|---|---|---|---|
| Groq · `openai/gpt-oss-120b` | Gratuito | ✅ | Não (uma por vez) | ✅ `function_calling` | ✅ | ✅ | **Sim** (smoke test 5/5, 03/11/2026) |
| Groq · `openai/gpt-oss-20b` | Gratuito | 📄 | Não (uma por vez) | 📄 `function_calling` | ✅ | ✅ | Não |
| Gemini · `gemini-3.5-flash-lite` | Gratuito | ✅ 📄 | 📄 Sim | ✅ `json_schema` | ✅ | ✅ | Não |
| Gemini · `gemini-3.6-flash` | Gratuito | 📄 | 📄 Sim | 📄 `json_schema` | ✅ | ✅ | Não |
| Gemini · `gemini-2.5-flash` | Gratuito, desliga em 01/12/2026 | 📄 | 📄 Sim | 📄 `json_schema` | ✅ | ✅ | Não |
| Gemini · `gemini-2.5-flash-lite` | Gratuito, só contas antigas | 📄 | 📄 Sim | 📄 `json_schema` | ✅ | ✅ | Não |
| OpenRouter · `openrouter/free` | Gratuito | ✅ ⚠️ varia por chamada | ⚠️ depende do modelo sorteado | ✅ ⚠️ `function_calling` | ✅ | ✅ | Não |
| OpenRouter · IDs `:free` do catálogo | Gratuito | ⚠️ anunciado em `/api/v1/models` | ⚠️ desconhecido | ⚠️ `function_calling` | ✅ | ✅ | Não |
| Ollama · `qwen3:8b` | Local | ✅ ⚠️ depende do modelo | Sim | ✅ ⚠️ `json_schema` | ✅ | ✅ | Não |
| Ollama · `llama3.1:8b`, `gpt-oss:20b` | Local | ⚠️ | ⚠️ | ⚠️ `json_schema` | ✅ | ✅ | Não |
| Anthropic · Claude | Pago | ✅ | Sim | 📄 padrão | ✅ | ✅ | Não neste ambiente |
| OpenAI | Pago | ✅ | Sim | 📄 padrão | ✅ | ✅ | Não neste ambiente |

Como ler as colunas de memória e confirmação: a memória (checkpoint SQLite) e o fluxo de confirmação (`HumanInTheLoopMiddleware`) são do LangGraph e **não dependem do provedor**. O structured output marcado com ✅ foi testado por `tests/test_llm.py::test_structured_output_uses_catalog_method_with_real_client`. O teste `tests/test_llm.py::test_real_client_full_flow_with_approval` roda o ciclo completo com o cliente real de cada provedor: proposta da escrita, pausa, banco intacto, aprovação, gravação, resultado da tool reenviado ao modelo. O que só uma API real confirma é se o **modelo** escolhe a ferramenta certa e preenche os argumentos corretamente.

### Diferenças entre provedores que o código trata

| Diferença | Onde aparece | Como foi tratada |
|---|---|---|
| O provedor valida a tool call e recusa a resposta inteira (HTTP 400 `tool_use_failed`) | Groq. Visto na API real: `effective_from: "2026-11-01"` em vez de `"2026-11"` | `finance_copilot/agent/builder.py::_recover_rejected_tool_calls` recupera a chamada de `failed_generation` e a devolve ao fluxo normal: a validação local recusa sem gravar, o erro vai ao modelo e ele corrige. Nos outros provedores esse caminho já era o padrão |
| Sem chamadas paralelas de ferramentas | GPT-OSS no Groq | Nada a mudar: o agente executa uma ferramenta por chamada. Custa mais chamadas por pergunta, dentro do limite de 10 |
| Structured output estrito exige todos os campos obrigatórios | Groq (`json_schema` com `strict`) | O catálogo define `function_calling` para o Groq; o "Preencher com IA" usa o método do catálogo |
| Modelo sorteado a cada chamada | `openrouter/free` | `require_parameters: true` faz o OpenRouter só rotear para endpoints que aceitam ferramentas |
| IDs de tool call vazios ou repetidos | Alguns backends abertos | `finance_copilot/agent/builder.py::_unique_tool_call_ids` substitui antes da aprovação. Sem isso, uma 2ª despesa igual seria tratada como repetição e não seria gravada (há teste) |
| Retries internos do cliente | Gemini faz 6 por padrão | A fábrica define `max_retries=0`; quem decide repetir é o middleware |
| Timeout | O `ChatOllama` não tem parâmetro próprio | Vai em `client_kwargs` (`finance_copilot/llm/providers.py::PROVIDERS`) |
| Modelo sem ferramentas | Vários modelos do Ollama | Bloqueado se o catálogo ou o perfil do LangChain indicarem; em execução, o erro `does not support tools` vira uma mensagem clara, sem fingir que funcionou |

## Groq (prioridade)

1. Crie uma conta em <https://console.groq.com> (não pede cartão).
2. Gere uma chave em **API Keys** (<https://console.groq.com/keys>).
3. No `.env`:

```dotenv
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
LLM_FREE_ONLY=true
LLM_TIMEOUT_SECONDS=60
GROQ_API_KEY=gsk_...
```

Integração: `langchain-groq` (`ChatGroq`).

**Limites do plano gratuito** para `openai/gpt-oss-120b` e `openai/gpt-oss-20b`, segundo a página de rate limits na data do catálogo: 30 requisições/min, 1.000/dia, **8.000 tokens/min**, 200.000 tokens/dia. Confirme os valores da sua conta em <https://console.groq.com/settings/limits>.

**O que isso significa na prática:** cada chamada ao modelo envia ~3.200 tokens (system prompt ~900 + schemas das 9 ferramentas ~2.200, estimativa por caracteres), e uma pergunta que usa ferramenta faz pelo menos 2 chamadas. Conte com **cerca de uma pergunta com ferramenta por minuto**. Ao estourar, o chat mostra "aguarde cerca de N segundos"; esperas de até 10 s são feitas automaticamente.

**Modelos desativados:** `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `qwen/qwen3-32b` e `moonshotai/kimi-k2-instruct` foram desativados para contas free/developer em 2026 (página de deprecations do Groq). `qwen/qwen3.8-27b` é Enterprise.

## Google Gemini

1. Acesse <https://aistudio.google.com/apikey> e clique em **Create API key** (cria um projeto do Google Cloud).
2. **Não** ative o Cloud Billing nesse projeto: sem billing, ele fica na camada gratuita.
3. No `.env`:

```dotenv
LLM_PROVIDER=google_genai
LLM_MODEL=gemini-3.5-flash-lite
GOOGLE_API_KEY=AIza...
```

Integração: `langchain-google-genai` (`ChatGoogleGenerativeAI`).

**Modelos:**

- `gemini-3.5-flash-lite` (recomendado) e `gemini-3.6-flash`: gratuitos na camada free, sem data de desligamento anunciada.
- `gemini-2.5-flash`: modelo de referência pedido para o projeto, mas **será desligado em 01/12/2026**; a partir daí, as chamadas vão para o `gemini-3.5-flash-lite`.
- `gemini-2.5-flash-lite`: continua disponível, mas **só para contas que já o usavam**. Contas novas devem usar o 3.5 Flash-Lite.

**Limites:** valem por projeto (não por chave), e a cota diária renova à meia-noite do horário do Pacífico. O Google não garante os valores; consulte os da sua conta no AI Studio. Ao esgotar, a API devolve `429 RESOURCE_EXHAUSTED`.

**Privacidade:** na camada gratuita, o Google pode usar o conteúdo enviado para melhorar seus produtos. Use apenas os dados fictícios do projeto.

## OpenRouter

1. Crie uma conta em <https://openrouter.ai> e gere uma chave em <https://openrouter.ai/settings/keys>.
2. No `.env` (a chave é própria; **nunca** coloque em `OPENAI_API_KEY`):

```dotenv
LLM_PROVIDER=openrouter
LLM_MODEL=openrouter/free
OPENROUTER_API_KEY=sk-or-...
```

**Integração:** `ChatOpenAI` (de `langchain-openai`) com `base_url=https://openrouter.ai/api/v1`. O pacote oficial `langchain-openrouter` 0.2.9 existe, mas rebaixaria o `pydantic` validado no projeto (2.14.2 → 2.12.5), então não foi adotado.

**Modelos:**

- `openrouter/free`: o roteador escolhe um modelo gratuito por requisição, entre os que suportam os recursos pedidos (ferramentas, structured output). Na prática, o modelo **pode mudar a cada chamada**, inclusive no meio de uma conversa, e a qualidade varia. É bom para experimentar e ruim para uma demonstração previsível.
- IDs terminados em `:free`: o catálogo traz `google/gemma-4-31b-it:free` e `nvidia/nemotron-3-ultra-550b-a55b:free`, que anunciavam suporte a `tools` em `GET /api/v1/models` em 03/11/2026. A lista de modelos gratuitos muda com frequência.

**Limites:** 20 requisições/min e 50/dia nos modelos gratuitos (1.000/dia depois de comprar pelo menos US$ 10 em créditos). Consulte o uso com `GET https://openrouter.ai/api/v1/key` (campo `free_model_daily_requests`).

**Como adicionar outro modelo `:free`:**

1. Confira em <https://openrouter.ai/models?supported_parameters=tools&max_price=0> que ele é gratuito e aceita `tools`.
2. Adicione-o a `finance_copilot/llm/models.toml` com `cost = "free"` e a data em `checked_on`.
3. Rode o smoke test com ele.

## Ollama (local)

O Ollama roda o modelo na sua máquina: não há chave, cota nem custo por token, mas o desempenho depende do seu hardware e os modelos pequenos erram mais ao escolher ferramentas.

1. Instale: no Linux/WSL, `curl -fsSL https://ollama.com/install.sh | sh` (Windows e macOS: <https://ollama.com/download>).
2. Inicie o servidor, se ele não subir sozinho: `ollama serve`.
3. **Baixe um modelo com suporte a ferramentas.** O app nunca baixa modelos automaticamente:

```bash
ollama pull qwen3:8b      # ~5 GB, recomendado
ollama list               # confirma o download
```

4. No `.env`:

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
```

Modelos com ferramentas aparecem com a etiqueta *tools* em <https://ollama.com/search?c=tools>. O catálogo traz `qwen3:8b` (com `reasoning = true`, que separa o raciocínio da resposta exibida), `llama3.1:8b` e `gpt-oss:20b` (~14 GB, precisa de 16 GB de memória). Para usar outro modelo local, adicione-o ao catálogo com `cost = "local"`.

**WSL:** se o Ollama estiver instalado no Windows e o app no WSL, `localhost` pode não alcançar o Windows. Use o IP do host em `OLLAMA_BASE_URL`, ou instale o Ollama dentro do WSL.

## Plano gratuito × execução local

| | Plano gratuito (Groq, Gemini, OpenRouter) | Local (Ollama) |
|---|---|---|
| Custo | Zero enquanto a conta ficar na camada free | Zero; usa sua CPU/GPU e energia |
| Limites | Por minuto e por dia, definidos pelo provedor | Só o hardware |
| Qualidade | Modelos grandes (ex.: 120B) | Modelos menores (8B a 20B) erram mais na escolha de ferramentas |
| Velocidade | Groq é muito rápido; os outros variam | Depende da GPU; em CPU pode levar dezenas de segundos |
| Privacidade | Dados vão ao provedor (Gemini free pode usá-los) | Nada sai da máquina |
| Disponibilidade | Modelos e limites mudam sem aviso | Estável depois do download |

## Seleção na interface

A barra lateral mostra o provedor, o modelo, o custo (gratuito, pago ou local), as observações do catálogo e o motivo quando o chat está inativo (ex.: falta `GROQ_API_KEY`). A escolha vale para a sessão do navegador e fica na URL; o `.env` não é alterado. Cada modelo tem as próprias conversas: trocar de modelo abre outra conversa, e voltar ao anterior traz a conversa dele de volta, com uma eventual aprovação pendente. Assim, o histórico e os checkpoints de modelos diferentes não se misturam.

Se você adicionar a chave no `.env` com o app aberto, reinicie o Streamlit.

## Erros comuns

| Mensagem ou sintoma | Causa | O que fazer |
|---|---|---|
| "Chave de API não encontrada... Defina GROQ_API_KEY" | Variável vazia no `.env` | Preencha a variável indicada e reinicie |
| "Modelo bloqueado pelo modo gratuito" | Modelo pago, ou fora do catálogo como gratuito | Escolha um modelo do catálogo ou siga as instruções da mensagem para verificá-lo e adicioná-lo |
| "Provedor de LLM desconhecido" | Erro de digitação em `LLM_PROVIDER` | Use `groq`, `google_genai`, `openrouter`, `ollama`, `anthropic` ou `openai` |
| "limite de requisições (HTTP 429)... aguarde cerca de N segundos" | Limite por minuto (Groq: 8K tokens/min) | Espere o tempo indicado; faça perguntas mais curtas |
| "A cota do plano gratuito deste provedor acabou" | Limite diário ou da conta | Espere a renovação (Groq e OpenRouter: diária; Gemini: meia-noite do Pacífico) ou escolha outro modelo na barra lateral |
| "exige créditos (HTTP 402)" | OpenRouter com saldo negativo ou modelo não gratuito | Confira o saldo; use só `openrouter/free` ou `:free` |
| "O provedor de IA recusou a requisição (HTTP 400)" | Pedido rejeitado pelo provedor (não é problema de chave) | Reformule; veja o log do terminal. Tool calls inválidas do Groq são corrigidas automaticamente pelo modelo |
| "Verifique a configuração (.env)" | Chave inválida ou revogada (401/403) | Gere outra chave no painel do provedor |
| "O modelo configurado não foi encontrado" | Modelo descontinuado ou, no Ollama, não baixado | Atualize `LLM_MODEL` ou rode `ollama pull <modelo>` |
| "Não foi possível conectar ao servidor do modelo" | Ollama parado ou URL errada | `ollama serve` e confira `OLLAMA_BASE_URL` |
| "não suporta tool calling" | Modelo sem ferramentas | Use um modelo do catálogo |

## Smoke test com a API real

O script `scripts/smoke_llm.py` faz cerca de 10 chamadas ao modelo, com pausas de 20 s entre os passos para respeitar limites por minuto. Ele usa um banco temporário com dados fictícios, então seus dados em `data/` não são tocados. Só roda quando chamado: não faz parte do `pytest`.

```bash
.venv/bin/python scripts/smoke_llm.py                                    # modelo do .env
.venv/bin/python scripts/smoke_llm.py --provider ollama --model qwen3:8b # outro modelo
.venv/bin/python scripts/smoke_llm.py --steps chat,read --pause 5        # só alguns passos
```

| Passo | O que verifica |
|---|---|
| `chat` | Mensagem simples, resposta em texto |
| `read` | Ferramenta de leitura (`get_financial_summary`) |
| `structured` | Ferramenta com parâmetros estruturados (`simulate_budget` com uma lista de 2 ajustes) |
| `write` | Escrita com aprovação: pausa, banco intacto antes de aprovar, gravação depois |
| `extract` | Structured output do "Preencher com IA" |

Antes de chamar a API, o script mostra o modelo e pede confirmação (`--yes` pula). Se todos os passos passarem, marque o modelo com `tested = "live"` no catálogo e atualize a tabela comparativa acima.

Os testes de avaliação mais longos ficam em `tests/test_llm_live.py` (`.venv/bin/python -m pytest -m live`). Eles fazem mais chamadas e, no Groq gratuito, tendem a esbarrar no limite por minuto.

## Como a troca de modelo funciona

```text
.env / barra lateral ─► Settings ─► llm/catalog.py::resolve_model ─► llm/factory.py ─► BaseChatModel
                                     (provedor, custo, tools)          (init_chat_model)        │
                                                                                                ▼
                                   create_agent(model, tools=ALL_TOOLS, middleware=[...], checkpointer)
```

As 9 ferramentas são `@tool` do LangChain com schemas Pydantic. O `bind_tools` de cada integração converte esses schemas para o formato do provedor (functions da OpenAI/Groq/OpenRouter, `functionDeclarations` do Gemini, `tools` do Ollama). O agente, os middlewares (aprovação, limite de chamadas, retry), o checkpoint e o banco recebem apenas um `BaseChatModel` e não sabem qual provedor está por trás. Por isso trocar de modelo não muda nenhuma ferramenta nem regra financeira.

## Fontes consultadas (03/11/2026)

- Groq: [rate limits](https://console.groq.com/docs/rate-limits), [tool use](https://console.groq.com/docs/tool-use), [structured outputs](https://console.groq.com/docs/structured-outputs), [models](https://console.groq.com/docs/models), [deprecations](https://console.groq.com/docs/deprecations), [errors](https://console.groq.com/docs/errors)
- Gemini: [pricing](https://ai.google.dev/gemini-api/docs/pricing), [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits), [deprecations](https://ai.google.dev/gemini-api/docs/deprecations), [models](https://ai.google.dev/gemini-api/docs/models)
- OpenRouter: [limits](https://openrouter.ai/docs/api-reference/limits), [tool calling](https://openrouter.ai/docs/guides/features/tool-calling), [free models router](https://openrouter.ai/docs/cookbook/get-started/free-models-router-playground), [lista de modelos](https://openrouter.ai/api/v1/models)
- Ollama: [download](https://ollama.com/download), [modelos com tools](https://ollama.com/search?c=tools)
