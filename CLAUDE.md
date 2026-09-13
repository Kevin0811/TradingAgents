# TradingAgents (fork)

## Repository boundaries — read before any git/gh operation

This repo is a fork. It has two remotes:

- `origin` → `https://github.com/Kevin0811/TradingAgents` — **this fork**. All work happens here: branches, commits, pushes, PRs, issues.
- `upstream` → `https://github.com/TauricResearch/TradingAgents` — **the original project**.

**Never perform any write operation against `TauricResearch/TradingAgents` (upstream) or against `https://github.com/TauricResearch/TradingAgents` directly, under any circumstance.** This includes, without limitation:

- creating branches, commits, or pushes
- opening, commenting on, or closing pull requests
- opening, commenting on, or closing issues
- any `gh` command (`gh pr create`, `gh issue create`, etc.) that targets `TauricResearch/TradingAgents` instead of `Kevin0811/TradingAgents`

When using `gh`, always pass `--repo Kevin0811/TradingAgents` explicitly (or ensure the current directory/branch tracks `origin`, not `upstream`) rather than relying on `gh`'s default repo detection, since this repo has both remotes configured.

Read-only operations against upstream (e.g. `git fetch upstream`, comparing against `upstream/main` to catch up with the original project) are fine — the restriction is specifically about writes.

## Scope of this fork's own work

This fork's active development is focused on the **API layer** (`tradingagents/api/`) and its CLI/Docker packaging — see recent commit history (`feat(api)`, `fix(api)`, `fix(service)`) — not the core multi-agent trading engine inherited from upstream. CI reflects this: linting, security scanning, and the test job are scoped to `tradingagents/api` and `cli`, not the full codebase (see `.github/workflows/ci.yml`).
