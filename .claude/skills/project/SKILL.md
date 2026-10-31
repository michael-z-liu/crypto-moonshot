---
name: project
description: Scope a piece of work with the user, then build it autonomously milestone by milestone with worker subagents, and open a PR at the end. Run /project <what you want> to start, or /project resume to continue a build after a restart or usage limit.
argument-hint: "<what to build> | resume"
disable-model-invocation: true
---

# /project

You are the master agent. You own the plan, delegate milestones to the `implementer` subagent,
review everything it produces, and are the only one who talks to the user. Work happens in three
phases: **scope** (with the user), **build** (unattended), **hand-off** (a PR for the user).

Arguments: `$ARGUMENTS`. If they are `resume`, go straight to [Resume](#resume).

## Phase 1: Scope

1. Read `CLAUDE.md`, `CONTRIBUTING.md` and `docs/architecture.md`, plus whatever code the request
   touches. Understand the request well enough to split it into milestones.
2. Ask the user about anything that would change *what* gets built: behaviour, interfaces, scope
   edges, how success is measured. Decide *how* to build it yourself. Prefer a few pointed
   questions over a long questionnaire; go back and forth as often as needed.
3. **Size it to one PR.** PRs are squash-merged and should be one logical change, roughly one step
   of the build order in `docs/architecture.md`. If the request is bigger, propose splitting it
   into several projects, scope the first one, and list the rest under *Follow-ups* in its plan.
4. Draft the milestones. Each one is a coherent, testable slice that leaves the check command
   passing, with:
   - **Delivers**: what exists afterwards
   - **Done when**: concrete acceptance criteria, including the tests that prove it
   - **Worker**: who builds it (see [Choosing a worker](#choosing-a-worker))
5. **Always ask for one final confirmation**, even if nothing seems unclear. Show the goal, what
   is out of scope, the milestones with their workers, and the branch name, then ask whether to
   start building. Do not write code before the user says yes.

## Phase 2: Build

Once the user confirms, work without asking the user anything unless a
[stop condition](#stop-conditions) is met.

### Start

1. Branch: if on `main`, create `<type>/<short-description>` (see `CONTRIBUTING.md`). Otherwise
   keep the current branch; the desktop app may already have created one for this session.
2. Write the plan to `docs/plans/<short-description>.md` from [plan-template.md](plan-template.md)
   with status `building`, commit it as `docs(plans): add plan for <short-description>`, and push
   the branch (`git push -u origin HEAD`).
3. Start the heartbeat (see [Heartbeat](#heartbeat)).

### For each milestone, in order

1. Mark it `in progress` in the plan and set `Updated`.
2. Either build it yourself or start the `implementer` subagent in the background with the model
   from the plan. Give it a self-contained brief: the milestone's *Delivers* and *Done when*, the
   relevant files and design notes, decisions from earlier milestones that affect it, and the plan
   file's path. It has none of this conversation.
3. While it works, stay available: answer the user, and fold any change they ask for into the plan
   (see [Change requests](#change-requests)).
4. When it reports back, review (see [Review](#review)). Send fixes back to the same worker with
   `SendMessage` so it keeps its context; after two rounds that still fail, either fix it yourself,
   re-run the milestone with a stronger model, or stop if a stop condition applies.
5. Accept: mark the milestone `done`, list its commits, record any decisions, set `Updated`,
   commit the plan (`docs(plans): complete <milestone>`), and push.

### Choosing a worker

Pick the cheapest option that will get it right first time. A failed attempt costs more than the
price difference.

| Worker   | Use for                                                                                       |
| -------- | --------------------------------------------------------------------------------------------- |
| `master` | A few small edits, plan and doc updates, small review fixes. Starting a worker costs a cold read of the code, so don't use one for small jobs. |
| `haiku`  | Mechanical work with a clear pattern to follow: config, boilerplate, renames, tests that copy existing ones. |
| `sonnet` | The default for feature work.                                                                  |
| `opus`   | Subtle correctness: order placement, money and quantity rounding, crash recovery, concurrency, parity between backtest and live. Also a retry of a milestone `sonnet` could not finish. |

### Review

Never accept a milestone on the worker's word.

1. Run the full check command from `CLAUDE.md` yourself. It must pass.
2. Check `git status` is clean and `git log` shows Conventional Commits for this milestone.
3. Read `git diff --stat` for the milestone, then the parts that carry the logic. Check them
   against *Done when*: are the tests real and do they cover the criteria, is it typed and UTC,
   does it follow `docs/architecture.md` (or update it, if the design had to change)?
4. Look specifically for: secrets or keys in code, logs or fixtures; anything that could place an
   order outside the testnet; behaviour the milestone didn't ask for.

### Change requests

The user may message you mid-build. Record the change under *Decisions* in the plan and adjust
the remaining milestones. If it affects the milestone in progress, message the worker with
`SendMessage` or let it finish and adjust in review, whichever wastes less work. If the change
alters what the user approved in a way they may not have intended, confirm it with them; they are
present, so this is not a stop.

### Stop conditions

Stop and ask the user only when:

- the work would read or use real API keys or `.env`, touch Binance mainnet, or run with
  `--mainnet` or `--mode live`
- the approved scope turns out to be wrong: the user's goal can't be met as approved, or meeting
  it needs something they excluded
- a milestone still fails review after two fix rounds and a retry with a stronger model
- the work needs a force push, history rewrite of pushed commits, or changes outside this
  repository

To stop: set status `blocked`, write the question under *Blocked on*, commit and push the plan,
send a push notification if that tool is available, and end your turn with the question.
Everything else, including design changes, is your call: decide, record it under *Decisions*,
and keep going.

### Heartbeat

The heartbeat restarts the build after a usage limit or any other stop. It is a recurring
`CronCreate` job in this session:

- cron: every 20 minutes at an off-minute, for example `7,27,47 * * * *`
- prompt: `Heartbeat: follow the Resume section of .claude/skills/project/SKILL.md.` (the skill
  only runs when the user types `/project`, so the heartbeat points at the file instead)

Cron jobs only fire while the session is idle, so a heartbeat never interrupts you. They live only
in this session and expire after 7 days; [Resume](#resume) re-creates the job if it is missing.
Delete it with `CronDelete` when the plan reaches `review` or `done`.

## Phase 3: Hand-off

1. Run the full check command once more on the final branch.
2. Open the PR with `gh pr create`. The title follows `CONTRIBUTING.md` (it becomes the squash
   commit on `main`). The body follows `.github/pull_request_template.md` and adds: the milestones
   and how each was verified, decisions and deviations from the plan, and anything the user should
   check by hand.
3. Watch CI once. If the desktop app offers PR monitoring, use that; otherwise run
   `gh pr checks --watch`. Fix failures in new commits.
4. Set status `review`, record the PR link, commit and push the plan, delete the heartbeat, and
   tell the user the PR is ready. Never merge it: the user reviews and merges.

## Resume

Use this when invoked as `/project resume`, by the heartbeat or by the user.

1. Find the plan: the `docs/plans/*.md` file on this branch whose status is `building` or
   `blocked`. If there is none, say so and stop.
2. If a worker you started is still running in this session, do nothing and end the turn.
3. If status is `blocked` and the user hasn't answered *Blocked on*, do nothing and end the turn.
4. If this is a new session (you have no memory of the build), check `Updated`: if it is less than
   30 minutes ago, another session may still be building, so ask the user before continuing.
5. Make sure the heartbeat exists (`CronList`); create it if not.
6. Reconcile the plan with git. For the milestone `in progress`, read its commits and any
   uncommitted changes. Uncommitted work from a worker that was cut off is a starting point, not
   something to throw away: brief a new worker to finish the milestone from the current working
   tree, or finish it yourself if little is left.
7. Continue the build loop from there.
