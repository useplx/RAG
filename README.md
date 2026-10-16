# RAG

RAG provides retrieval components for PLeX applications:

- a multilingual embedding model;
- document chunk storage;
- vector indexing and search;
- optional web search;
- context construction for generation;
- a graph hook for suppressing unsupported vision tokens.

RAG is a supporting package. It does not produce a standalone assistant or
`.mln` application.

## Project status

The retrieval components are exposed as independent package parts. A consuming
application chooses the Graph, Orchestration, and Native pieces it needs. RAG does
not inject a generated path or assume where generation input lives.

`[[hooks]]` can select either RAG's graph transformation or its Orchestration
augmentation facet. The consuming project chooses the represented path, so RAG
can be attached without modifying the generator package's C++ source.

Generator integration is selected through the `generator` capability. RAG
does not depend on a concrete generator interface package.

## Components

| Component | Purpose |
|---|---|
| Embedder | Converts queries and passages into comparable vectors. |
| Embedder tokenizer | Tokenizes text for the multilingual embedder. |
| TurboVec provider | Creates, loads, updates, searches, and serializes vector indexes. |
| Chunk store | Maps retrieved identifiers back to source text. |
| Context builder | Fits retrieved passages into the configured context budget. |
| Search provider | Performs optional web search through libcurl. |
| Vision-token hook | Prevents unsupported vision tokens from being sampled as generated text. |

## Dependencies

RAG currently requires:

- MyeLiN;
- utf8proc for the embedder tokenizer;
- Rust 1.89.0 and Cargo for TurboVec;
- libcurl for the optional search provider.

## Package configuration

RAG publishes these variables:

| Variable | Default | Meaning |
|---|---:|---|
| `top_k` | `5` | Maximum number of retrieved passages. |
| `bit_width` | `4` | Vector index storage width. |
| `context_budget` | `2200` | Maximum context text assembled for the generator, in characters. |
| `minimum_similarity` | `0.2` | Minimum cosine similarity accepted before a passage can reach the generator. |
| `require_lexical_match` | `true` | Requires at least one informative query term in each accepted passage. |

A consuming project configures them with:

```toml
[packages.rag]
top_k = 3
bit_width = 4
context_budget = 2200
minimum_similarity = 0.2
require_lexical_match = true
```

The context budget must be chosen together with the generator's own input limit.

## Retrieval assets

A composed application expects these assets:

```text
assets/
├── rag_index.tvim
├── rag_chunks.bin
└── embedder_tokenizer.json
```

The index and chunk store must be produced together from the same documents and
embedder version. Rebuild both when the encoder or chunking strategy changes.

Build the optional `rag-index` tool in a consumer that links MyeLiN Runtime and
the package's TurboVec library, then pass an artifact containing
`embedder.forward`, the output asset directory, and the documents:

```bash
rag-index application.mln assets documentation.md more-docs.md
```

The tool uses the same packaged Graph and tokenizer used at query time. It
replaces `rag_index.tvim` and `rag_chunks.bin` together.

See [RAG documentation](https://useplx.github.io/packages/rag/).

## Query behavior

At query time, the retrieval implementation:

1. embeds the user query;
2. searches the configured vector index;
3. resolves matching chunk identifiers;
4. builds a context payload within `context_budget`;
5. returns the augmented buffer to the composed application.

Queries and indexed passages must use the prefixes expected by the multilingual
E5 model:

- `query: ` for queries;
- `passage: ` for indexed passages.

## Validate retrieval

Before connecting retrieval to a generator, validate:

- the index and chunk store were generated together;
- the same embedder version is used for indexing and querying;
- expected documents appear in top-k search results;
- the assembled context fits the selected generator;
- missing or empty assets are treated as deployment failures.

Run the native retrieval and context tests with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Documentation

Full documentation lives at **[https://useplx.github.io/packages/rag/](https://useplx.github.io/packages/rag/)**.

- [Package contract](https://useplx.github.io/packages/rag/)
- [Compose packages](https://useplx.github.io/guide/compose-packages/)


## License

RAG is source-available under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Noncommercial use is permitted
without charge. Commercial use requires a separate agreement with the copyright
holder.

Third-party libraries, checkpoints, documents, and datasets retain their own
licenses.
