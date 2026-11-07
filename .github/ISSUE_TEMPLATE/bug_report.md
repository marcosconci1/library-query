---
name: Bug report
about: Report something that does not work as documented
labels: bug
---

**What happened**

A clear description of the problem.

**What you expected**

What should have happened instead.

**Steps to reproduce**

1. Run `library-query ...`
2. The output is ...

**Environment**

* OS:
* Python version (`python3 --version`):
* library-query version (`uv run library-query describe --json` reports it):
* Embeddings endpoint type (OpenRouter, local vLLM, Ollama, other):

**Diagnostic output**

If the failure touches indexing or retrieval, paste the relevant part of
`library-query status`. Remove anything that looks like a credential
first.

```text
(paste output)
```
