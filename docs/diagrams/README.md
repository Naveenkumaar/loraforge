# Interactive diagrams

Standalone, explorable HTML views of loraforge (built with Archify). Open the
`.html` files in a browser: pan, zoom, search, follow a path, and click a node's
`SRC` badge to jump to the exact code file and lines it was drawn from.

| View | What it answers |
|---|---|
| [loraforge-architecture.html](loraforge-architecture.html) | What the system is made of (RAG pipeline + fine-tuning adapters) and how the parts connect |
| [loraforge-flow.html](loraforge-flow.html) | Answering one question, end to end (scope → retrieve → rerank → generate) |

Each `.html` is rebuilt from its `*.candidate.json` with Archify; edit the JSON, not the HTML.
