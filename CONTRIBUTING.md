# Contributing

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/). A `commit-msg` hook rejects
messages that don't match (run `uv run pre-commit install` once per clone).

```
<type>(<optional scope>): <summary>

<optional body: what changed and why>
```

**Types**

| Type       | Use for                                               |
| ---------- | ----------------------------------------------------- |
| `feat`     | New behaviour (a command, strategy, data source)      |
| `fix`      | Bug fix                                               |
| `perf`     | Faster or lighter, same behaviour                     |
| `refactor` | Code change with no behaviour change                  |
| `test`     | Adding or fixing tests only                           |
| `docs`     | Documentation only                                    |
| `build`    | Dependencies, `pyproject.toml`, `uv.lock`             |
| `ci`       | GitHub Actions, pre-commit config                     |
| `chore`    | Anything else that doesn't touch `src/` behaviour     |
| `revert`   | Reverting an earlier commit                           |

**Scopes** are optional and name the area touched, e.g. `data`, `backtest`, `strategy`, `live`,
`cli`, `config`, `deps`. Add new scopes as the code grows; keep them one word.

**Rules**

- Summary in the imperative mood, lower case, no trailing period, 72 characters max:
  `feat(data): download klines from binance`, not `Added kline download.`
- One logical change per commit. If the summary needs "and", split it.
- Use the body to explain *why* when it isn't obvious. Wrap at 72 characters.
- Mark breaking changes with `!` after the type/scope and a `BREAKING CHANGE:` footer, e.g.
  `feat(config)!: rename MOONSHOT_API_KEY`. This matters most for anything that changes how
  live trading behaves or how stored data is read.
- Never commit secrets, `.env` files or downloaded market data.

## Branches and pull requests

- Branch from `main` as `<type>/<short-description>`, e.g. `feat/kline-download`.
- `main` must always pass CI. Changes land through a pull request.
- Pull requests are always squash-merged, so each PR becomes one commit on `main` and its title
  becomes that commit's message. Keep each PR to one logical change.

### PR titles

The PR title follows the commit message format above, and CI rejects titles that don't. Because it
is the only line that survives in `git log`, write it for someone scanning history months later:

- Describe the outcome, not the activity: `feat(data): resume kline downloads after interruption`,
  not `feat: data changes` or `fix: address review comments`.
- Be specific enough to find with `git log --grep`: name the command, strategy or module.
- Rename the PR when its scope changes during review, before merging.
- The description (from the template) records why and how it was tested; it stays on GitHub and
  is linked from the squashed commit by its `(#123)` suffix.
