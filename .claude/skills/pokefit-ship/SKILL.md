---
name: pokefit-ship
description: Commit and push pokefit work so it redeploys https://pokefit.party, safely for a PUBLIC repo — author identity, no model files, conventional commit, tracker update, CI watch and a live-site check. Use when a round of work is verified and ready to commit/push.
---

# Ship a pokefit change

Every push to `main` redeploys the live site. Run `pokefit-verify` first.

## 1. Public-repo hygiene (before every commit)

```bash
git config user.email                  # must be 381345+tribixbite@users.noreply.github.com
git status --short                     # add files by name; never `git add -A`
git diff --cached --name-only          # no *.3mf *.stl *.skp *.skb *.dae, no personal paths or data
```
If the identity is wrong on a new clone:
`git config user.name tribixbite && git config user.email 381345+tribixbite@users.noreply.github.com`.

## 2. Tracker and docs

- `NOTES.md`: delete finished to-dos, add new open items and the measurements a decision still needs
  (history belongs in `git log`, not the tracker).
- User-visible features: update `README.md`, `web/public/llms.txt` and the About text in `web/index.html`.

## 3. Commit

Conventional commit (`feat:`, `fix:`, `docs:` …), body = what changed and why, signed on its own line with
an em dash and the model name (e.g. `— Opus 5.5`). No "Claude", no co-author trailer. Write the message to a
file or a quoted heredoc. Never amend, rebase, reset or force-push without the user's explicit permission.

## 4. Push and watch

Push only when the user asked for it or the round is complete and verified.
```bash
git push
gh run list -L 2                       # find the run id
gh run watch <id> --exit-status        # must end "success"
```
If CI fails, read the failing step's log (`gh run view <id> --log-failed`), fix forward with a new commit.

## 5. Live check (UI or engine changes)

Open https://pokefit.party in the browser agent: the page's `main.js?v=<hash>` must match the new build
(Cloudflare caches assets 4 h; the stamp is what makes a deploy take effect), the engine badge loads, and
the changed feature works with no console errors.
