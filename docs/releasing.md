# Releasing

Versions and the CHANGELOG are written by [release-please](https://github.com/googleapis/release-please)
from Conventional Commit messages. Nobody edits version numbers by hand.

## How a release happens

1. Features and fixes reach `master` the usual way: feature branch → `staging` → `master`.
2. On every push to `master`, the **Release** workflow (`.github/workflows/release-please.yml`)
   opens or updates a pull request titled `chore(master): release X.Y.Z`. It bumps the
   version in `pyproject.toml` and `frontend/package.json`, syncs `uv.lock` and
   `frontend/package-lock.json`, and adds the CHANGELOG entry.
3. When you want to release, merge that pull request. release-please then tags `vX.Y.Z`,
   publishes a GitHub release with the same notes, and opens a pull request that merges
   `master` back into `staging`. Merge that one too.

Until you merge the release PR, it keeps collecting whatever lands on `master`.

## Which version comes next

Before 1.0 (`bump-minor-pre-major` and `bump-patch-for-minor-pre-major` in
`release-please-config.json`):

| Commit | Example | Next version |
|---|---|---|
| `fix: …` | `fix(api): explain missing SCW_MODEL` | 0.5.0 → 0.5.1 |
| `feat: …` | `feat(frontend): provider status in settings` | 0.5.0 → 0.5.1 |
| `feat!: …` or a `BREAKING CHANGE:` footer | `feat!: Scaleway as default provider` | 0.5.0 → 0.6.0 |
| `docs`, `chore`, `build`, `ci`, `test`, `refactor`, `style` | | no release on its own |

`feat`, `fix`, `perf`, `security` and `revert` appear in the CHANGELOG; the other types are
hidden. From 1.0 on, `feat` bumps the minor and a breaking change the major version.

The **Commit messages** check fails a pull request with a commit that lacks one of these
prefixes, since release-please would silently leave that commit out.

Upgrade steps that a commit message cannot carry (a changed default, a new `.env` variable,
a re-ingestion) go into the CHANGELOG by hand, below the generated entry, in the release PR.

## One-time setup: the release token

Pull requests opened with the workflow's own `GITHUB_TOKEN` do not trigger other
workflows, so CI would never run on the release PR and its required checks would wait
forever. The workflow therefore uses a token stored as the `RELEASE_PLEASE_TOKEN` secret:

1. Open [the new fine-grained token page](https://github.com/settings/personal-access-tokens/new)
   (or [your fine-grained tokens](https://github.com/settings/personal-access-tokens) →
   Generate new token). (By hand: your profile
   picture, top right → **Settings** → **Developer settings**, at the bottom of the left
   sidebar → Personal access tokens → **Fine-grained tokens** → Generate new token. These
   are your *account* settings; the repository's Settings tab has no Developer settings.)
2. Repository access: **Only select repositories** → this repository. One token can cover
   several repositories: select each one that uses release-please and store the same token
   in each of them.
3. Permissions → Repository permissions: **Contents: Read and write**, **Pull requests: Read
   and write** (Metadata: Read-only is added automatically).
4. Choose an expiry and put a reminder in your calendar; the release workflow fails once
   it has expired.
5. In this repository (its **Settings** tab) → Secrets and variables → Actions → New repository secret,
   name `RELEASE_PLEASE_TOKEN`, value the token.

## Changing the next version by hand

To release a specific version (e.g. `1.0.0` when leaving the 0.x series), set
`"release-as": "1.0.0"` for the `"."` package in `release-please-config.json`, merge it to
`master`, and remove the line again after the release.
