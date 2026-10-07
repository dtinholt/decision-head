# Publishing a fresh public repository

This project was developed inside a private monorepo. Earlier commits of
that monorepo contain private addresses and paths that were later
removed from the tree (security review 2026-10-06, finding 16). The
public repository must therefore start from an export of the reviewed
tree **with no history**. Do not push a `git subtree split`, a
`filter-repo` rewrite, or any other branch derived from the monorepo
history.

The steps below are for the maintainer. Run them from a machine that has
the private monorepo checked out.

## 1. Freeze the reviewed tree

In the monorepo, make sure the package directory has no uncommitted
changes, and note the commit you are publishing:

```
cd <monorepo>
git status --short tools/decision-head     # must print nothing
git rev-parse HEAD                         # record this in your release notes
```

Run the checks one last time from the package directory:

```
cd tools/decision-head
make lint typecheck test
decisionhead selftest
```

## 2. Export the tree without history

`git archive` writes the tracked files at one commit and nothing else:
no `.git` directory, no history, and none of the ignored files (`.venv`,
caches, `.env`, local audit or cache files).

```
OUT="$(mktemp -d)/decision-head"
mkdir -p "$OUT"
git -C <monorepo> archive --format=tar HEAD:tools/decision-head | tar -x -C "$OUT"
ls -A "$OUT"
```

Check that `ls` shows no `.git`, `.venv`, `.env`, `*.egg-info`,
`__pycache__` or `*.jsonl` outside `examples/`.

## 3. Verify the export is clean

Run this from inside the export. It looks for home-directory paths,
private and carrier-grade NAT IPv4 ranges, Tailscale and local-network
DNS suffixes, and common secret shapes:

```
cd "$OUT"
grep -rnIE \
  -e '/Users/[A-Za-z]' -e '/home/[a-z]' -e '~/[A-Za-z]' \
  -e '\b192\.168\.[0-9]' \
  -e '\b10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b' \
  -e '\b172\.(1[6-9]|2[0-9]|3[01])\.[0-9]' \
  -e '\b100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]' \
  -e '\.ts\.net\b' -e '[a-z0-9-]\.(local|lan)([:/"'"'"' ]|$)' \
  -e 'sk-[A-Za-z0-9]{20}' -e 'ghp_[A-Za-z0-9]{20}' -e 'BEGIN [A-Z ]*PRIVATE KEY' \
  . && echo "REVIEW THE HITS ABOVE" || echo "clean"
```

Expected output: `clean`.

The generic patterns cannot catch the names of your own machines, your
private repositories, or people other than the author, and those names
must not be written into the public tree even as search patterns. Keep
them in a file **outside** the export, one extended regular expression
per line, and run:

```
grep -rnIiE -f /path/outside/the/tree/private-patterns.txt . \
  && echo "REVIEW THE HITS ABOVE" || echo "clean"
```

Expected output: `clean`. A typical file lists machine hostnames, the
monorepo's name, employer-internal project names, and the names of
anyone mentioned in private notes.

Known matches that are acceptable:

* `LICENSE`, `CITATION.cff`, `pyproject.toml` and the README's citation
  block name the author. That is intended.
* `203.0.113.10` (RFC 5737 documentation range) is the example backend
  address throughout. It is not matched by the patterns above.
* The two review documents in `docs/` describe finding 16 in neutral
  words; the original values were redacted on 2026-10-06.

If either grep reports anything else, fix it in the monorepo, commit,
and start again from step 1. Do not edit the export by hand: the export
must match a reviewed commit.

## 4. Create the public repository

```
cd "$OUT"
git init -b main
git add -A
git commit -m "decisionhead 0.1.0: initial public release"
gh repo create dtinholt/decision-head --public --source . --remote origin --push \
  --description "Typed-question decision service with calibrated probabilities from any logprob-capable model"
```

Then, in the repository settings on GitHub:

1. Enable **Private vulnerability reporting** (Security tab). `SECURITY.md`
   tells reporters to use it.
2. Protect `main`: require the `CI` workflow to pass before merging.
3. Confirm Dependabot is active (`.github/dependabot.yml`).
4. Wait for the first `CI` run on `main` to pass before announcing the
   repository. The README badges refer to it.
5. To publish a release, push a tag `v0.1.0`; the `Release` workflow
   builds the wheel and sdist and attaches them to a GitHub release.

## 5. After publishing

* Keep developing in whichever repository you choose as the source of
  truth. If that stays the monorepo, repeat steps 1 to 3 for every
  public update and copy the reviewed tree over the public checkout as
  a normal commit there. Never push monorepo history to the public
  remote.
* Run the step 3 greps in CI of the private side before each sync, so a
  private reference is caught before it reaches the export.
