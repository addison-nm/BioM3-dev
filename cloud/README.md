# BioM3 cloud jobs (`cloud/`)

SkyPilot task files that run BioM3 jobs on commercial GPU cloud — **Mithril** or
**AWS** — inside the prebuilt public GHCR image `ghcr.io/natural-machine/biom3:cuda-dev`. Same job scripts
(`scripts/cloud/<job>.sh`, baked into the image), different provider.

- Per-job env-var reference: [`../docs/setup/cloud_jobs.md`](../docs/setup/cloud_jobs.md)
- The image itself + multi-node notes: [`../docker/README.md`](../docker/README.md)

## Files

| Job | Mithril | AWS |
| --- | --- | --- |
| test (pytest) | `test.mithril.yaml` | `test.aws.yaml` |
| pretrain (Stage-3 from scratch) | `pretrain.mithril.yaml` | — |
| finetune (Stage-3) | `finetune.mithril.yaml` | `finetune.aws.yaml` |
| generate (Stage-3 inference) | `generate.mithril.yaml` | `generate.aws.yaml` |

All use the **host-docker-pull** pattern (no `image_id`): `setup:` pulls the image;
`run:` executes `docker run … scripts/cloud/<job>.sh` on each node. Multi-node is
enabled by `num_nodes > 1` (see the docker README).

---

## Publishing the image (GHCR) — runbook

The image is published **public** at **`ghcr.io/natural-machine/biom3`**, tagged
`cuda-<sha>` (immutable, per commit) and `cuda-dev` (moving; what `cloud/*.yaml` track).

Why GHCR + public:
- **Cost**: Mithril instances are ephemeral, so *every launch pulls the whole ~11.6 GB
  image*. From ECR that is internet egress (~$0.09/GB ≈ **$1/launch**, billed under the
  AWS EC2/"EC2-Other" line). GitHub Packages is **free for public packages** ("usage is
  free for public packages … data transferred in from any source is free").
- **Simplicity**: a public image needs **no pull authentication**, so the launch path
  carries no registry token at all (no 2420-char, 12-hour ECR token to re-mint).

> **Before you publish**: the image bakes `src/`, `scripts/`, `tests/` (incl. the test
> HDF5s) and `configs/`. Publishing it **makes all of that world-readable** — confirm
> that is intended, especially if `natural-machine/BioM3-dev` is a private repo.

### One-time: create a token and publish

```bash
# 1. Create a classic PAT (push side only).
#    GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
#    → Generate new token (classic) → scopes: write:packages, read:packages
#    → If the org enforces SAML SSO: click "Configure SSO" → Authorize for natural-machine
export GHCR_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx

# 2. Log in to GHCR (needed to PUSH; pulling a public image needs no login).
echo "$GHCR_TOKEN" | docker login ghcr.io -u addison-nm --password-stdin

# 3. Build, then push with git-derived version tags (cuda-<sha> + cuda-dev).
#    push.sh refuses a dirty tree so cuda-<sha> truthfully matches the commit.
docker/build.sh --variant cuda --awscli --platform linux/amd64
docker/push.sh --variant cuda

# 4. Make the package PUBLIC — the first push creates it PRIVATE by default.
#    Web: https://github.com/orgs/natural-machine/packages → biom3
#         → Package settings → Danger Zone → Change visibility → Public
#    (Optionally under "Manage Actions access", link the BioM3-dev repo.)

# 5. Verify an ANONYMOUS pull — this is exactly what a Mithril instance does.
docker logout ghcr.io
docker pull ghcr.io/natural-machine/biom3:cuda-dev
```

### On every image change (code, deps, job scripts)

The image bakes `src/ scripts/ tests/ configs/`, so any change to them needs a rebuild
and repush before it reaches a cloud run:

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u addison-nm --password-stdin
docker/build.sh --variant cuda --awscli --platform linux/amd64
docker/push.sh --variant cuda       # commit first — push.sh refuses a dirty tree
```

### What this does NOT remove

**AWS credentials are still required** — GHCR only replaces the *image registry*. The
container entrypoint still `aws s3 sync`s weights/data from S3, so launches still pass
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`, and the S3 egress
for weight syncs still bills to AWS. Narrow it with `BIOM3_WEIGHTS_INCLUDES`.

## Launching (Mithril)

Launch via the **bundled sky CLI** (`mithril sky launch`), not the `mithril launch`
wrapper — only the former honors `--secret` (redacted creds), the `secrets: null`
pattern, and `config.mithril.limit_price`.

```bash
eval "$(aws configure export-credentials --format env)"
mithril sky launch -c biom3-ft-$(date +%y%m%d-%H%M%S) cloud/finetune.mithril.yaml -y --down \
  --secret AWS_ACCESS_KEY_ID --secret AWS_SECRET_ACCESS_KEY --secret AWS_SESSION_TOKEN \
  --env RUN_ID=my_run --env NGPU=4
```
(The image is public on GHCR — pulled anonymously — so there's no registry token; AWS
creds are passed only for the container's S3 weight-sync. `scripts/cloud/mithril_launch.sh`
wraps this and auto-loads `configs/jobs/local.env`.)

Or the helper, which harvests AWS creds, auto-loads `configs/jobs/local.env`, and picks a unique name:

```bash
scripts/cloud/mithril_launch.sh cloud/finetune.mithril.yaml biom3-ft --env RUN_ID=my_run
```

### Overriding job values (never edit the stable yaml)

The `envs:` in each task file are **defaults**, grouped into a *BioM3 job spec* block
and an *infra / data-staging* block. Override per run (precedence: `--env` > `--env-file`
> `envs:`):

- `--env RUN_ID=foo --env NGPU=4` — individual overrides
- `--env-file configs/jobs/finetune.env` — sidecar dotenv (see `configs/jobs/*.example.env`)
- `--env EXTRA_ARGS="--batch_size 8 --lr 1e-4"` — extra trainer/sampler flags (forwarded to `"$@"`)
- Set any field to `null` in a yaml to make it **required** (the launch fails, before a
  GPU is provisioned, unless the user supplies it).

---

## Mithril: launching from a separate machine

You can drive Mithril launches from any host — e.g. an EC2 instance hosting a web app.
The launching machine is a **thin client**: no GPU, no Docker, no repo, no weights.
Everything heavy happens on the Mithril-provisioned instance.

### The launching machine needs

1. **`uv`** (or pip) — to install the CLI.
2. **`mithril-client`** — `uv tool install -U mithril-client` (provides `mithril` + the
   bundled `sky`).
3. **Mithril auth** *(separate from AWS — easy to forget)*: `~/.config/mithril/config.yaml`
   with `api_key` + `project_id`. Interactive `mithril setup`, or headless via the
   `MITHRIL_API_KEY` / `MITHRIL_PROJECT` env vars (best for a service).
4. **AWS CLI + credentials** — needed **only for the S3 weight-sync** (the public GHCR
   image pulls anonymously — no registry auth). `aws configure export-credentials`
   harvests them to pass as `--secret`; the container uses them to `aws s3 sync` weights.
   So the identity needs just `s3:GetObject`/`ListBucket` on the weights bucket. On EC2,
   an **IAM instance role** with that permission is cleanest — no static keys.
5. **The task YAML file** (`cloud/<job>.mithril.yaml`) — and, if used, the
   `scripts/cloud/mithril_launch.sh` helper and any `--env-file` sidecar. **Not the whole
   repo**: the host-pull yamls have no `workdir:`; the code is baked into the image.
6. **Outbound HTTPS egress** to `api.mithril.ai`, `ghcr.io` (image pull), and the AWS
   S3 / STS endpoints (matters in a locked-down VPC).
7. A little local disk — `mithril sky launch` runs a **local SkyPilot API server** and
   keeps state in `~/.sky/` (it generates its own SSH keys, cluster records, logs).

### The launching machine does NOT need

- ❌ **Docker** — the pull / run happen on the *remote* GPU instance.
- ❌ a **registry token** — the GHCR image is public, pulled anonymously.
- ❌ a **GPU**, the **weights**, or the **BioM3-dev repo**.

### Cross-account note

The GHCR image is public, so there's no cross-account *registry* concern. The only
cross-account consideration is **S3 read on the weights bucket**: if the launching
identity is in a different AWS account than the bucket owner, it needs cross-account S3
access (bucket policy) or must assume a role in the bucket's account.

### Operational notes for a service that launches repeatedly

- **Use a unique cluster name every launch.** Mithril retains bid names indefinitely, so
  reusing a name fails with a *misleading* `ResourcesUnavailableError` (not real GPU
  scarcity). The helper appends a timestamp; a web app must do the same.
- **`--down` only fires after a *job* finishes.** A provision-stage failure leaves the
  instance billing — reconcile on error with `mithril sky status` / `mithril sky down <c>`.
- **The local API server accumulates state / occasional zombie executors.** Ship
  [`scripts/cloud/mithril_reset.sh`](../scripts/cloud/mithril_reset.sh) alongside; run it
  when a launch wedges (it stops the API server, reaps zombies, clears stale locks).
- **Temporary creds expire** (SSO/role, ~1–12 h). Re-harvest AWS creds
  per launch (the helper does this every time).
