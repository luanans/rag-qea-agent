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

### Avaliação (RAGAS)
 Métricas obtidas para uma amostra de 10 perguntas:

| Métrica | Score |
|---|---|
| Faithfulness | 0.939 |
| Answer Relevancy | 0.840 |
| Context Precision | 0.917 |
| Context Recall | 0.900 |


---

## Tools vs. Agente — separação e motivações

### Tools (`tools/`)

As tools são **funções puras de recuperação de informação**, instanciáveis e testáveis de forma isolada. Cada tool:

- Herda de `BaseTool[InputT, OutputT]` — interface genérica com validação Pydantic de entrada/saída
- Expõe `to_gemini_schema()` — gera o JSON Schema que o Gemini usa para decidir quando e como chamar a tool
- Retorna `ToolResult` — container com `.success`, `.output` (JSON) e `.error`, sem acoplar nenhuma dependência do SDK do Gemini
- Não tem conhecimento do agente que a utiliza

| Tool | O que faz | Quando o Gemini a usa |
|---|---|---|
| `search_documents` | Busca semântica nos chunks | Perguntas sobre conceitos, mecanismos, resultados |
| `extract_section` | Retorna o texto completo de uma seção | Perguntas sobre abstract, introdução ou conclusão |

### Agente (`agent/`)

O `QAAgent` é a **camada de orquestração**: ele não recupera a informação, delega esta responsabilidade às tools. Sua responsabilidade é:

1. Gerenciar o histórico de conversa (`contents`)
2. Passar os schemas das tools ao Gemini a cada iteração
3. Deserializar o `ToolResult` e repassá-lo ao histórico de conversa
4. Executar as chamadas via `ToolRegistry`
5. Encerrar o loop quando o Gemini produz uma resposta textual final

**Por que tools e agente separados?**

Modelos de linguagem são eficazes em raciocínio e geração de linguagem, mas operam sobre um contexto fixo: não consultam fontes externas, não executam código e não têm acesso a informações além do que foi visto no treinamento. As tools atuam nessa lacuna: são o mecanismo pelo qual o agente estende suas capacidades. Sem essa separação, lógica de recuperação e lógica de orquestração se misturariam em um único componente difícil de evoluir e substituir.

A separação também torna os testes precisos e rápidos: as tools podem ser validadas unitariamente com um `SectionStore` ou `VectorStore` de teste, sem nenhum mock do Gemini; o agente pode ser testado sem embeddings ou ChromaDB reais, bastando mockar o `google.genai.Client`. Cada camada pode falhar de forma independente e ser corrigida de forma isolada.

---

## Setup — do zero ao primeiro `/ask`

### 1. Instalar dependências

```bash
uv sync
```

### Pré-requisitos

- Python 3.14+
- Chave de API do Google AI Studio: [aistudio.google.com](https://aistudio.google.com)
### 1. Instalar dependências

Instale as dependências via pip:
```bash
pip install -r requirements.txt
```

Como alternativa, se você utiliza o uv:

```bash
uv sync
```

### 2. Configurar variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
GEMINI_API_KEY=sua_chave_aqui
```

As demais configurações têm valores padrão (ver `app/config.py`). Para customizar:

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

O projeto tem dois tipos de teste: unitários (rápidos, sem dependências externas) e E2E (requerem API key e ingest executado).

**Apenas unitários** — não precisam de API key, ChromaDB ou papers baixados:
```bash
python -m pytest -m "not e2e"
```

**Apenas E2E** — requerem `.env` com `GEMINI_API_KEY` e `ingest.py` já executado:
```bash
python -m pytest -m e2e -v
```

**Todos os testes:**
```bash
python -m pytest
```

---

## Decisões técnicas

### Modelo de embedding: `gemini-embedding-001`

A escolha do modelo gemini-embedding-001 justifica-se não apenas pelos excelentes resultados em benchmarks, com métricas de 67,71 em MTEB Retrieval e 75,85 no MTEB geral, mas também por sua performance superior nos testes locais comparado aos modelos all-MiniLM-L6-v2, bge-large-en-v1.5 e e5-large-v2. Além do desempenho, o acesso via API foi um fator decisivo, pois elimina a necessidade de GPU local e apresenta um custo competitivo de $0,15 por milhão de tokens.

O sistema usa embeddings **assimétricos**: `RETRIEVAL_DOCUMENT` para indexar os chunks e `RETRIEVAL_QUERY` para vetorizar as consultas, o que reflete o uso recomendado do modelo para tarefas de retrieval.

### Vector store: ChromaDB

O ChromaDB foi selecionado por sua natureza lightweight, operando em processo e com persistência em disco, o que dispensa a necessidade de infraestrutura adicional. A métrica de similaridade adotada é a de cosseno (hnsw:space: cosine), métrica que foca na direção dos vetores (contexto).

### Estratégia de chunking: Chuncking Hierárquico

Primeiro o Docling divide os documentos em seções, depois o chunker divide o texto por número de palavras (`chunk_size=500`) com sobreposição (`overlap=50`). A divisão feita **por seção** preserva o contexto semântico. Cada chunk carrega metadados de `paper_id`, `title` e `section`, permitindo filtros no momento da busca.

O tamanho de 500 palavras foi escolhido por caber na janela do contexto de `gemini-embedding-001` e por representar um parágrafo substancial sem ser excessivamente longo.

### Parser de PDF: Docling

Docling foi escolhido por identificar automaticamente a estrutura do documento (cabeçalhos de seção, parágrafos, fórmulas, listas) com mais fidelidade que extratores puramente textuais como PyMuPDF. O parser normaliza os nomes de seção via `SECTION_ALIASES`, mapeando variações como "1 Introduction", "6 Conclusion" e "Conclusions" para identificadores canônicos, o que permite que o `ExtractSectionTool` funcione de forma previsível independente de como cada paper rotulou suas seções.

### Agente: Gemini com function calling nativo

O QAAgent utiliza o SDK oficial google-genai com a funcionalidade de function calling nativa do Gemini 2.5 Flash. Não foram utilizados frameworks de orquestração por se tratar de um projeto de escopo reduzido, onde as capacidades nativas do SDK já atendem todas as necessidades de forma simples e direta. O fluxo de execução segue um loop previsível: durante até max_iterations (default=5), o agente aciona as ferramentas necessárias conforme a orientação do modelo, encerrando o processo assim que o Gemini gera a resposta textual final.

Com a descontinuação da família gemini 2.0, iniciada em março de 2026, a migração para as versões 2.5 Flash ou 2.5 Flash-Lite tornou-se a recomendação oficial para novos projetos. Dessa forma, o agente utiliza o `gemini-2.5-flash`.

### API: FastAPI com Pydantic

FastAPI com validação Pydantic em todos os endpoints. A injeção de dependência via `Depends` garante que `VectorStore`, `SectionStore` e `QAAgent` sejam singletons (`lru_cache`), evitando re-carregamento a cada request.

---

## Limitações conhecidas
A solução utilizou estratégias robustas para garantir os critérios exigidos, apresentando desempenho satisfatório. As limitações abaixo visam a evolução do projeto:

### Chunking Fixo vs. Semântico
O chunk_size fixo demonstrou um bom desempenho dentro do escopo do projeto. No entanto, mesmo com o uso de overlapping, em cenários mais complexos ainda existe a possibilidade perda de contexto. A evolução para um chunking semântico (baseado na variação de significado entre sentenças) é o próximo passo lógico. Essa solução foi preterida inicialmente pela simplicidade e velocidade de aplicar o chunking fixo dentro das seções já estruturadas pelo Docling.

### Perda de Contexto Multimodal
Embora o Docling seja excelente para extração estrutural, artigos de ML dependem fortemente de tabelas de resultados e gráficos. Se o parser converter tabelas em texto bruto de forma desordenada ou ignorar elementos visuais, o agente terá dificuldade em responder perguntas quantitativas específicas caso a informação não esteja replicada no corpo do texto.

### Abordagem de RAG Híbrido
O RAG atual é do tipo denso, baseado na similaridade de cosseno de embeddings. Nos testes realizados, o desempenho essa abordagem demonstrou desempenho satisfatório. Entretanto, para sistemas que enfrentam falhas na busca de termos técnicos muito específicos ou siglas raras, recomenda-se uma abordagem híbrida (hybrid retrieval), combinando a busca vetorial (semântica) com a busca por palavras-chave (BM25/keyword search).

### Ausência de Reranking
Atualmente, os chunks retornados pelo ChromaDB são ordenados estritamente por similaridade de cosseno. A implementação de um Cross-Encoder para Reranking no topo dos resultados iniciais aumentaria significativamente a precisão, garantindo que o contexto mais relevante seja o primeiro a ser entregue ao Gemini.

### Ingestão Serial e Escalabilidade
O processamento dos papers ocorre de forma serial. Para corpora maiores, a arquitetura de ingestão precisará ser paralelizada (via multiprocessing ou task queues) para otimizar o tempo de parsing e geração de embeddings.

### Observabilidade
O sistema atual não tem de tracing. A integração de ferramentas como LangSmith, Langfuse ou OpenTelemetry é essencial para monitorar custos e latência em produção.
