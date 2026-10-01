# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub security advisories](https://github.com/Yashjindal11/synapsi/security/advisories/new)
rather than public issues. Include steps to reproduce and the affected version.
You should receive an acknowledgement within a week.

## Supported versions

Only the latest minor release receives security fixes during the 0.x series.

## Security model

SynapSI sends text to model providers and executes tools chosen by models.
Treat model output as untrusted input.

- **Secrets**: API keys are read from environment variables only. Config files
  name the variable (`api_key_env`) and unknown config keys (such as `api_key`)
  are rejected. Keys are never logged or included in results.
- **Tools**: the calculator parses an arithmetic AST (no `eval`) with size
  limits; `sql_query` opens SQLite read-only and allows only `SELECT`;
  `fetch_url` refuses private, loopback, and link-local addresses and does not
  follow redirects. There is deliberately no built-in code-execution tool.
- **Prompt injection**: retrieved documents and web pages can contain
  instructions aimed at agents. SynapSI labels such text as evidence and never
  executes it, but a model may still be influenced by it. Review evidence and
  restrict tools for high-stakes use.
- **Web API**: intended for local use. It binds to `127.0.0.1` by default, has
  no user accounts, and lets requests choose only strategy and mode (never
  models, URLs, or keys). Set `SYNAPSI_API_TOKEN` before exposing it on a
  network, and put it behind TLS.
- **Reports**: HTML reports and the dashboard escape all model-generated text
  and only link `http(s)` URLs.
- **Memory**: long-term memory is off by default and stores only the question,
  answer, verdict, and summary — never context, facts, or evidence.
