# Guia de estudo para a entrevista

Respostas curtas para falar em voz alta, com o arquivo que sustenta cada uma. Se não souber abrir o arquivo citado e explicar o trecho, estude-o antes.

## Qual problema o projeto resolve

Registrar gastos e analisar finanças é trabalhoso, e por isso as pessoas abandonam planilhas. Aqui, basta escrever "gastei 85 com pizza ontem" ou "por que gastei mais este mês?". O sistema interpreta, consulta dados reais e pede confirmação antes de alterar qualquer coisa.

Como demonstração técnica, o foco é um problema mais difícil do que o domínio: **como colocar uma LLM, que é probabilística, em cima de um sistema que precisa estar sempre correto**, como é o caso de dinheiro.

## Por que usar um agente em vez de só programação convencional

A parte convencional continua existindo: cálculos, validação, gráficos e formulários são código comum. O agente resolve o que código fixo faz mal:

- **Entrada aberta:** "gastei 85 na pizza ontem", "pizza 85 reais", "ontem comi uma pizza de 85". Regex para isso é frágil.
- **Caminho variável:** "qual categoria mais contribuiu para o aumento?" e "quanto economizo cortando lazer?" exigem ferramentas e sequências diferentes.
- **Contexto:** "E em outubro?" só faz sentido com a pergunta anterior.
- **Explicação:** transformar um JSON de comparação em uma resposta útil.

Onde o caminho é fixo, não uso agente. O dashboard não passa pela LLM, e o "Preencher com IA" é uma única chamada de structured output. Isso está documentado em [ARCHITECTURE.md](ARCHITECTURE.md#quando-usar-agente-e-quando-usar-workflow-determinístico).

## Por que LangChain em vez de chamar a API diretamente

Com a API direta, eu mesmo escreveria o laço de tool calling, a conversão de schemas para cada provedor, a persistência do histórico, a pausa e retomada, e retries.

| Sem framework | Neste projeto |
|---|---|
| Laço manual de tool calling | `create_agent` |
| Formato diferente por provedor | `init_chat_model`: mudança apenas no `.env` (6 provedores, veja [LLM_PROVIDERS.md](LLM_PROVIDERS.md)) |
| Tabela própria de histórico | `SqliteSaver` |
| Máquina de estados de aprovação | `HumanInTheLoopMiddleware` + `interrupt` |
| Instrumentação manual | LangSmith por variáveis de ambiente |

**Custos reais:** mais dependências, APIs que mudam rápido e abstrações que podem esconder o que acontece. Foi preciso ler o código instalado para confirmar o comportamento das versões atuais. Para uma única chamada simples, eu usaria o SDK direto.

## Como funciona a comunicação entre modelo e tools

1. O agente envia ao modelo as mensagens e os JSON Schemas das tools.
2. O modelo responde com uma `AIMessage` com `tool_calls`: nome, argumentos e id. **Ele não executa nada.**
3. O Pydantic valida os argumentos, e a função Python é executada.
4. O resultado volta como `ToolMessage`, com o mesmo id.
5. O modelo decide se chama outra tool ou responde.

**Onde ver:** `finance_copilot/agent/tools.py`, e o ciclo real em `tests/test_llm.py::test_real_client_full_flow_with_approval`.

## Como o agente mantém contexto

Modelos não têm memória entre chamadas. O `SqliteSaver` salva as mensagens a cada passo, separadas por `thread_id`. A cada nova chamada, o histórico é carregado e reenviado.

O `thread_id` fica na URL, então um F5 restaura a conversa, inclusive uma aprovação pendente. O estado do agente fica em `agent_state.db`, separado dos dados financeiros.

**Onde ver:** `finance_copilot/agent/builder.py::open_checkpointer`, `tests/test_agent.py::test_followup_receives_previous_turns_from_checkpoint`.

## Como são evitadas alucinações financeiras

Reduzo o espaço onde uma alucinação pode acontecer:

1. **Contas em código.** Totais, diferenças e projeções vêm do serviço, em centavos inteiros (`finance_copilot/service.py`).
2. **Valores prontos.** As tools devolvem o texto já formatado (`{"cents": 8500, "brl": "R$ 85,00"}`), e o modelo só precisa repetir.
3. **Ausência explícita.** `data_status` e campos `None` diferenciam "sem dados" de "R$ 0,00"; comparações sem dados têm `comparable: false`.
4. **Simulação marcada por código.** O aviso na tela vem de `is_simulation` no resultado da tool, não do texto do modelo.
5. **Prompt:** instrui a consultar sempre e a nunca reaproveitar números antigos.
6. **Transparência:** o painel "Ferramentas usadas" permite conferir a resposta contra os dados.

**Limite honesto:** o modelo ainda pode errar ao redigir, como trocar um número na resposta. Isso reduz o risco e facilita a auditoria, mas não o elimina. Os números do dashboard, por sua vez, não dependem da LLM. Uma melhoria seria verificar automaticamente se cada valor citado na resposta aparece nos resultados das tools.

## Como a aplicação protege operações sensíveis

- **Aprovação obrigatória:** as 4 tools de escrita passam pelo `HumanInTheLoopMiddleware`, que age no grafo, fora do controle do modelo. Um teste garante que toda tool esteja classificada.
- **Confirmação com dados reais:** o cartão é gerado dos argumentos e do banco, não do texto da LLM.
- **Sem SQL arbitrário:** só existem funções específicas com consultas parametrizadas.
- **Validação em camadas:** Pydantic na tool, serviço e `CHECK` no banco.
- **Duplicação:** guarda do `interrupt_id` e chave de idempotência no banco, gravada na mesma transação.

**Frase-chave:** *o prompt orienta, o código garante.* Se o modelo for enganado por uma descrição maliciosa, ele ainda não consegue gravar sem aprovação.

## Decisões arquiteturais

| Decisão | Motivo | Alternativa rejeitada |
|---|---|---|
| Um agente com 9 tools | Resolve o problema e é simples de depurar | Multiagente: roteamento e latência sem ganho |
| `create_agent` em vez de grafo próprio | O laço padrão atende | `StateGraph` manual: mais código para o mesmo resultado |
| Dois arquivos SQLite | Conversa e dinheiro têm ciclos de vida diferentes | Um banco só: acopla dados da aplicação ao formato do LangGraph |
| Centavos inteiros | Exatidão | `float`: `0.1 + 0.2 != 0.3` |
| Salário fora de `transactions`, com vigência | Evita duplicar receita e preserva o passado | Transação mensal automática: precisa de job e pode duplicar |
| `tool_call_id` como chave de idempotência | Estável entre retomadas, salvo no checkpoint | Hash do conteúdo: bloquearia duas compras iguais legítimas |
| Registro de operações em tabela | Vale para as 4 escritas e guarda o resultado | Coluna `UNIQUE` em `transactions`: não cobre update, delete e salário |
| Exclusão lógica | Auditoria | Exclusão física: perde histórico |
| Resumos por mês inteiro | O salário é mensal, e não há rateio a explicar | Datas livres: exigiria regra de proporcionalidade |
| Tools recebem o serviço por `context_schema` | Sem globais, e os testes usam outro banco | Closures ou variáveis globais |
| Histórico lido do checkpoint | Uma só fonte de verdade; sobrevive ao F5 | Cópia em `st.session_state`: dessincroniza |

## Limitações atuais

**Verificação:**

- O fluxo foi testado com LLM roteirizada e com os clientes reais de todos os provedores contra um servidor local. Com a API real, o Groq (`openai/gpt-oss-120b`) passou no `scripts/smoke_llm.py` em 03/11/2026; os demais provedores e o `tests/test_llm_live.py` completo ainda não foram executados contra a API real.
- A integração com o serviço LangSmith também não foi exercitada de ponta a ponta, só a lógica que a ativa.

**Funcionais:**

- O cartão aprova ou rejeita todas as ações de uma vez, sem decisão individual.
- Não há edição direta no cartão: para corrigir, o usuário rejeita e informa o motivo, e o modelo propõe de novo.
- O cartão avisa quando uma operação será recusada, mas o botão Aprovar continua ativo; a recusa vem depois, sem gravar nada.
- Sem streaming: a resposta aparece só ao final, com um indicador de espera.
- Categorias fixas. Datas futuras não são aceitas, portanto não há contas agendadas.
- Comparações e resumos usam meses inteiros.
- Sem lista de conversas antigas: uma conversa é acessível apenas pela URL.
- Exclusão lógica sem tela de restauração.

**Técnicas:**

- O histórico cresce sem resumo. Uma conversa muito longa pode estourar o limite do modelo; "Nova conversa" contorna isso. O arquivo de checkpoints também não é limpo.
- Duas abas na mesma conversa são protegidas contra gravação duplicada, mas podem exibir um estado momentaneamente desatualizado.
- A idempotência usa o `tool_call_id`. Como alguns provedores repetem ou omitem esse id, `finance_copilot/agent/builder.py::_unique_tool_call_ids` substitui ids ausentes ou repetidos antes da aprovação.
- Nos planos gratuitos, os limites por minuto são apertados (Groq: 8K tokens/min, cerca de uma pergunta com ferramenta por minuto). O app avisa e não insiste, mas a demonstração precisa de pausas.
- Mensagens e dados vão para o provedor de LLM e, se ligado, para o LangSmith. Por isso, apenas dados fictícios.
- Aplicação local, sem autenticação, de usuário único.
- Tema claro fixo.

## Melhorias futuras

Em ordem de valor:

1. **Avaliação contínua:** conjunto de casos no LangSmith, medindo a escolha de tools e a precisão entre modelos.
2. **Verificador de números na resposta:** comparar os valores citados com os resultados das tools.
3. **Streaming** de tokens e do progresso das tools.
4. **Gestão do histórico** com `SummarizationMiddleware` ou `ContextEditingMiddleware`, com cuidado para os resumos não preservarem valores antigos.
5. **Edição direta e decisão por ação** no cartão.
6. **Importação de extrato CSV**, classificando em lote com structured output.
7. **Multiusuário:** `user_id` nas tabelas e autenticação; Postgres com o checkpointer correspondente.
8. **Memória de longo prazo** com `store` do LangGraph, para preferências como "Uber é sempre Transporte".

## Perguntas técnicas prováveis

**O que é um agente, para você?**
Um modelo em um laço, que escolhe ações, observa resultados e decide quando parar. A diferença para um workflow é quem decide o próximo passo: o modelo, não o código.

**Qual a diferença entre LangChain e LangGraph?**
LangGraph é o motor de baixo nível: grafos com estado, checkpoints e interrupções. LangChain traz as abstrações sobre ele, como `create_agent`, tools e middlewares. O `create_agent` devolve um grafo LangGraph.

**O que acontece, passo a passo, quando clico em Aprovar?**
`decide` confere se o `interrupt_id` ainda está pendente e invoca o grafo com `Command(resume=...)`. O LangGraph carrega o checkpoint e reexecuta o nó do middleware, e `interrupt()` devolve a decisão. Segue para o nó `tools`, que valida os argumentos e executa a tool. O serviço grava em uma transação com o registro da chave. O resultado volta ao modelo, que responde. A interface faz `st.rerun()` e o dashboard relê o banco.

**E se o processo cair durante a pausa?**
Nada se perde: a pausa está no SQLite. Ao reiniciar, o cartão reaparece, e a aprovação funciona. `tests/test_agent.py::test_conversation_and_pending_approval_survive_restart`.

**E se cair depois de gravar, antes de salvar o checkpoint?**
A retomada executa a tool de novo com o mesmo `tool_call_id`. O serviço encontra a chave, não grava e devolve o resultado original com `replayed: true`. É o que mostra `tests/test_agent.py::test_replaying_checkpoint_after_commit_does_not_duplicate`.

**Por que a lógica de idempotência não fica na tool?**
Porque precisa estar na mesma transação da gravação. Se estivesse fora, uma queda entre os dois passos causaria inconsistência. `finance_copilot/service.py::_apply_once`.

**E se o modelo enviar argumentos errados?**
O Pydantic rejeita, e o erro volta como `ToolMessage` para o modelo corrigir. O usuário nem vê um cartão de aprovação, pois o predicado `when` só pede aprovação para argumentos válidos.

**O predicado `when` não abre uma brecha na aprovação?**
Não, porque só desativa a aprovação para o que o próprio schema da tool vai recusar. Ele depende apenas dos argumentos. Por isso, verificações que olham o banco ficaram de fora: o estado poderia mudar entre a checagem e a execução.

**Como você lida com prompt injection?**
Assumo que pode ocorrer, por exemplo em uma descrição salva. Por isso, o prompt marca dados como dados, mas a defesa real é o limite de capacidades: não há SQL, não há acesso à rede pelas tools, e escritas exigem aprovação com os valores reais visíveis.

**Por que não RAG?**
RAG serve para recuperar texto não estruturado. Aqui os dados são estruturados, e a resposta exige agregação exata, o que SQL faz. Embeddings não somam centavos.

**Por que não dar à LLM uma tool de text-to-SQL?**
Ela seria flexível, mas abriria risco de consultas erradas com aparência correta e de acesso indevido, e os cálculos voltariam a depender do modelo. Tools específicas limitam o que pode dar errado.

**Como testa um agente sem custo nem flakiness?**
Um chat model roteirizado substitui só a LLM; grafo, tools e banco são reais. Clientes reais são testados contra um servidor HTTP local. A qualidade das decisões fica nos testes `live`, à parte, porque são caros e não determinísticos.

**Como você mediria a qualidade em produção?**
Taxa de escolha correta de tools em um conjunto de casos, taxa de rejeição nas confirmações (sinal direto de interpretação errada), erros de tool, número de chamadas por pedido, latência e custo. O LangSmith fornece os traces.

**Por que `temperature` é opcional?**
Baixa temperatura ajuda na consistência, mas alguns modelos recusam o parâmetro. A configuração fica no `.env`, e sem valor usa-se o padrão do modelo.

**O que você mudaria com mais tempo?**
Primeiro, avaliação contínua com modelos reais, porque hoje a qualidade da escolha de tools é a parte menos medida. Depois, o verificador de números e o streaming.

**Por que Streamlit?**
Velocidade para um MVP em Python. A desvantagem é o modelo de reexecução do script a cada interação, que torna a idempotência dos formulários indispensável. Em produção, eu separaria uma API e um frontend próprio.

## Roteiro de demonstração (5 minutos)

**Preparação:** chave no `.env`, banco recém-populado (apague `data/*.db` e rode o seed), `.venv/bin/python -m pytest` já executado em um terminal. Ensaie com o modelo real: as frases exatas do modelo variam.

**0:00 – Contexto (30 s).** "É um copiloto financeiro. O ponto técnico é confiabilidade: a LLM interpreta e explica, mas não faz contas e não grava sem aprovação." Mostre a tela: dashboard à esquerda, agente à direita.

**0:30 – Dashboard sem IA (30 s).** Troque o mês. "Isto é SQL e Python, sem LLM: não há motivo para gastar uma chamada aqui."

**1:00 – Escrita com aprovação (1 min 15 s).** Digite: **Gastei 85 reais com pizza ontem**.

- Mostre o cartão, aponte a data resolvida e a categoria deduzida. "O agente escolheu a tool. A pausa é um `interrupt` do LangGraph: nada foi gravado."
- **Aperte F5.** "A execução pausada está no SQLite, não na memória."
- Clique em **Aprovar** e aponte as despesas e o saldo mudando.
- Abra "Ferramentas usadas". "Isto é o que realmente foi executado."

**2:15 – Análise com várias ferramentas (45 s).** Digite: **Quais categorias mais contribuíram para o aumento dos meus gastos?** Depois: **Quanto gastei com alimentação em setembro?** e **E em outubro?** "O segundo pedido não diz 'alimentação': o contexto vem do checkpoint." Ajuste os nomes dos meses às datas do seed.

**3:00 – Simulação (30 s).** Digite: **Quanto economizaria reduzindo o lazer em 20%?** Aponte o aviso. "Esse aviso vem de um campo do resultado da tool, não do texto do modelo."

**3:30 – Rejeição e dados ausentes (30 s).** Digite: **Apaga o aluguel deste mês**, depois **Rejeitar** com um motivo. Em seguida: **Quanto gastei em janeiro de 2020?** "Ele diz que não há dados, em vez de responder R$ 0,00."

**4:00 – Código (45 s).** Abra `finance_copilot/agent/builder.py::build_agent` e `finance_copilot/agent/approvals.py::build_interrupt_on`. Depois `finance_copilot/service.py::_apply_once`: "Registro da operação e gravação na mesma transação: repetir a execução não duplica."

**4:45 – Testes e fechamento (15 s).** Mostre o resultado do pytest. "Os testes usam grafo, checkpoint e banco reais, trocando só a LLM. Sem as proteções de aprovação e de idempotência, esses testes falham: isso foi verificado removendo cada uma." Mencione uma limitação e o próximo passo, a avaliação contínua.

**Plano B (sem internet ou API fora do ar):** mostre o dashboard, os formulários e o `pytest`, e explique o fluxo em `tests/test_agent.py::test_approval_persists_and_agent_reports_real_result`.

### Fatos para ter na ponta da língua

- 9 tools: 5 de leitura e 4 de escrita.
- 5 middlewares, cada um com justificativa.
- 2 bancos: `finance.db` e `agent_state.db`.
- Dinheiro em centavos inteiros.
- Chave de idempotência: `agent:<thread_id>:<tool_call_id>`.
- Grafo: `model` e `tools` em ciclo, mais os nós dos middlewares.
