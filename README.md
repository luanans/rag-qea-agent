# RAG Q&A Agent

Q&A Agent sobre artigos científicos de Machine Learning, construído com uma arquitetura RAG (Retrieval-Augmented Generation) e um agente com function calling via Gemini.

Os três papers cobertos são:
- **Attention Is All You Need** (Vaswani et al., 2017)
- **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (Lewis et al., 2020)

---

## Visão geral da arquitetura

```
Usuário
  │
  │  POST /ask {"question": "..."}
  ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI  (app/)                                    │
│  routes.py → QAAgent.answer(question)               │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  QAAgent  (agent/qa_agent.py)                       │
│                                                     │
│  Loop de function calling (até max_iterations=5):   │
│  1. Envia pergunta + schemas das tools ao Gemini    │
│  2. Gemini decide qual tool chamar                  │
│  3. Agente executa a tool via ToolRegistry          │
│  4. Resultado volta ao Gemini                │
│  5. Repete até Gemini produzir resposta final       │
└────────────┬──────────────────┬─────────────────────┘
             │                  │
             ▼                  ▼
┌────────────────┐   ┌─────────────────────┐
│ search_        │   │ extract_section     │
│ documents      │   │ (tools/)            │
│ (tools/)       │   │                     │
│                │   │ Retorna o texto      │
│ Busca semân-   │   │ completo de uma     │
│ tica via       │   │ seção (abstract,    │
│ VectorStore    │   │ conclusion, etc.)   │
└───────┬────────┘   └──────────┬──────────┘
        │                       │
        ▼                       ▼
┌───────────────────┐   ┌───────────────────┐
│ VectorStore       │   │ SectionStore      │
│ (rag/)            │   │ (rag/)            │
│                   │   │                   │
│ ChromaDB +        │   │ Dict em memória   │
│ Gemini embeddings │   │ carregado de      │
│ cosine similarity │   │ sections.pkl      │
└───────────────────┘   └───────────────────┘
        ▲                       ▲
        └───────────┬───────────┘
                    │
        ┌───────────────────────┐
        │  ingest.py            │
        │                       │
        │  1. Download PDFs     │
        │  2. Parse (Docling)   │
        │  3. Salva SectionStore│
        │  4. Chunking + embed  │
        │  5. Salva ChromaDB    │
        └───────────────────────┘
```

**Fluxo de uma pergunta:**
1. A API recebe a pergunta via `POST /ask`
2. O `QAAgent` monta o histórico de conversa e passa os schemas das tools ao Gemini
3. O Gemini decide chamar `search_documents` (busca semântica) e/ou `extract_section` (seção completa)
4. O agente executa cada tool call via `ToolRegistry` e devolve os resultados ao Gemini
5. O Gemini sintetiza a resposta final citando papers e seções

---

## Tools vs. Agente — separação e motivações

### Tools (`tools/`)

As tools são **funções determinísticas e testáveis** que realizam operações de recuperação de informação. Cada tool:

- Herda de `BaseTool[InputT, OutputT]` — interface genérica com validação Pydantic de entrada/saída
- Expõe `to_gemini_schema()` — gera o JSON Schema que o Gemini usa para decidir quando e como chamar a tool
- Retorna `ToolResult` — container com `.success`, `.output` (JSON) e `.error`, desacoplado do protocolo HTTP ou do SDK do Gemini
- Não tem conhecimento do agente nem do modelo de linguagem

| Tool | O que faz | Quando o Gemini a usa |
|---|---|---|
| `search_documents` | Busca semântica nos chunks via ChromaDB | Perguntas sobre conceitos, mecanismos, resultados |
| `extract_section` | Retorna o texto completo de uma seção | Perguntas sobre abstract, introdução ou conclusão |

### Agente (`agent/`)

O `QAAgent` é a **camada de orquestração**: ele não sabe de recuperação de informação — delega tudo às tools. Sua responsabilidade é:

1. Gerenciar o histórico de conversa (`contents`)
2. Passar os schemas das tools ao Gemini a cada iteração
3. Interpretar as `function_call` responses do Gemini
4. Executar as chamadas via `ToolRegistry`
5. Encerrar o loop quando o Gemini produz uma resposta textual final

**Por que separar?** As tools podem ser testadas unitariamente sem nenhum mock do Gemini — basta instanciar a tool com um `SectionStore` ou `VectorStore` de teste. O agente pode ser testado sem embeddings ou ChromaDB reais — basta mockar o `google.genai.Client`. Essa separação reduz o acoplamento e torna os testes rápidos e confiáveis.

---

## Setup — do zero ao primeiro `/ask`

### Pré-requisitos

- Python 3.14+
- `uv` (gerenciador de pacotes)
- Chave de API do Google AI Studio: [aistudio.google.com](https://aistudio.google.com)

### 1. Instalar dependências

```bash
uv sync
```

### 2. Configurar variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
GEMINI_API_KEY=sua_chave_aqui
```

As demais configurações têm valores padrão razoáveis (ver `app/config.py`). Para customizar:

```env
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=gemini-embedding-001
CHROMA_PERSIST_DIR=./persist/chroma
CHUNK_SIZE=500
CHUNK_OVERLAP=50
AGENT_MAX_ITERATIONS=5
```

### 3. Ingerir os papers (primeira vez apenas)

```bash
python ingest.py
```

Esse comando:
1. Faz download dos três PDFs do arxiv para `./data/papers/`
2. Parseia os PDFs com Docling e extrai as seções
3. Persiste o `SectionStore` em `./persist/sections.pkl`
4. Gera embeddings com `gemini-embedding-001` e salva no ChromaDB em `./persist/chroma/`

O processo leva alguns minutos na primeira execução. Nas execuções seguintes, o download e o ChromaDB são pulados automaticamente se já existirem.

### 4. Subir a API

```bash
uvicorn app.main:app
```

A API estará disponível em `http://localhost:8000`. Documentação interativa: `http://localhost:8000/docs`.

### 5. Fazer uma pergunta

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main contribution of the Transformer architecture?"}'
```

### 6. Rodar os testes

```bash
python -m pytest
```

Os testes são unitários e não precisam de API key, ChromaDB ou papers baixados.

---

## Decisões técnicas

### Modelo de embedding: `gemini-embedding-001`

O `gemini-embedding-001` foi escolhido por apresentar os melhores resultados de retrieval nos benchmarks avaliados (MTEB Retrieval: 67,71; MTEB geral: 75,85) entre os modelos testados, que incluíam `all-MiniLM-L6-v2`, `bge-large-en-v1.5` e `e5-large-v2`. O modelo é acessado via API ao custo de $0,15/M tokens, eliminando a necessidade de GPU local para embeddings.

O sistema usa embeddings **assimétricos**: `RETRIEVAL_DOCUMENT` para indexar os chunks e `RETRIEVAL_QUERY` para vetorizar as consultas, o que reflete o uso recomendado do modelo para tarefas de retrieval.

### Vector store: ChromaDB

ChromaDB foi escolhido por ser leve, não exigir infra adicional (roda em processo, persiste em disco), e ter uma API simples compatível com o padrão do projeto. Para produção com milhões de chunks, a substituição por Qdrant ou Pinecone seria direta — `VectorStore` isola completamente o ChromaDB do resto do sistema.

A similaridade usada é **cosseno** (`hnsw:space: cosine`), adequada para embeddings normalizados.

### Estratégia de chunking: palavras com overlap

O chunker divide o texto por número de palavras (`chunk_size=500`) com sobreposição (`overlap=50`). A divisão é feita **por seção** — chunks não cruzam fronteiras de seção, preservando o contexto semântico. Cada chunk carrega metadados de `paper_id`, `title` e `section`, permitindo filtros no momento da busca.

O tamanho de 500 palavras foi escolhido por caber confortavelmente no contexto de embedding do Gemini e por representar um parágrafo substancial sem ser excessivamente longo.

### Parser de PDF: Docling

Docling foi escolhido por identificar automaticamente a estrutura do documento (cabeçalhos de seção, parágrafos, fórmulas, listas) com mais fidelidade que extratores puramente textuais como PyMuPDF. O parser normaliza os nomes de seção via `SECTION_ALIASES` — mapeando variações como "1 Introduction", "6 Conclusion" e "Conclusions" para identificadores canônicos — o que permite que o `ExtractSectionTool` funcione de forma previsível independente de como cada paper rotulou suas seções.

### Agente: Gemini com function calling nativo

O `QAAgent` usa o SDK `google-genai` com function calling nativo do Gemini 2.5 Flash. Não foi usada nenhuma framework de agente (LangChain, LlamaIndex, CrewAI) para manter o controle explícito sobre o loop de raciocínio e facilitar o debugging. O loop é simples: até `max_iterations` rodadas, o agente executa tools até o Gemini produzir uma resposta textual final.

### API: FastAPI com Pydantic

FastAPI com validação Pydantic em todos os endpoints. A injeção de dependência via `Depends` garante que `VectorStore`, `SectionStore` e `QAAgent` sejam singletons (`lru_cache`), evitando re-carregamento a cada request.

---

## Limitações conhecidas

**Corpus fixo.** O sistema cobre apenas três papers. Adicionar novos papers requer re-executar `ingest.py`. Não há endpoint de ingestão dinâmica.

**Sem memória de conversa.** Cada chamada ao `/ask` é tratada de forma independente. O agente não mantém contexto entre perguntas — não é possível fazer perguntas de acompanhamento como "e sobre o segundo paper que você mencionou?".

**Rate limiting sem retry.** O sistema expõe o erro 429 do Gemini diretamente ao cliente sem backoff automático ou fila. Sob uso intenso, o cliente precisa gerenciar os retries.

**Chunking por palavras.** A divisão por número de palavras é simples, mas pode cortar sentenças no meio. Uma alternativa melhor seria usar sentenças ou parágrafos como unidade mínima, mas aumentaria a complexidade do chunker.

**Sem reranking.** Os resultados do ChromaDB são retornados diretamente por score de cosseno, sem um estágio de reranking (ex: cross-encoder). Isso pode retornar chunks relevantes superficialmente mas semanticamente distantes da pergunta.

**Ingestão serial.** `ingest.py` processa os papers sequencialmente. Para um corpus maior, a paralelização do parsing e do embedding seria necessária.

**Deployment não coberto.** Não há Dockerfile, `docker-compose.yml`, ou configuração de CI/CD. A aplicação roda apenas localmente.
