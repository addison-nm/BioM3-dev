# BioM3 on Mithril — quick runbook

Barebones, linear steps to provision a Mithril GPU instance and run BioM3 in the
container. For the full reference (all options, bind-mount mode, multi-arch, etc.) see
[README.md](../docker/README.md). Gotchas we hit are noted inline as **⚠**.

---

## 0.a Provision (Mithril console)

- Create instance: **single GPU** (B200), spot bid or reservation.
- Select your **SSH key**.
- **No persistent storage needed** — weights stream from S3; results go back to S3 or stay
  on the instance for the session.
- Note the instance **IP** once it reaches **Allocated**.

## 0.b Provision + run via task.yaml (Mithril CLI — recommended)

**One command** provisions a B200, builds the image, and runs the smoke, driven by
[`mithril/run_tests.task.yaml`](./run_tests.task.yaml). This automates steps 1 and 4–7 below.

**One-time: install + authenticate the `ml` CLI.**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # install uv (the CLI's package manager)
uv tool install -U --refresh mithril-client         # installs the `ml` command
ml setup                                            # interactive auth — or use the env vars below
```

Non-interactive auth instead of `ml setup`:

```bash
export MITHRIL_API_KEY=fkey_xxx     # from https://app.mithril.ai/account/api-keys
export MITHRIL_PROJECT=proj_xxx     # curl -H "Authorization: Bearer $MITHRIL_API_KEY" https://api.mithril.ai/v2/projects
```

**Launch** (run from the repo root so `workdir: .` syncs this repo):

```bash
ml launch mithril/run_tests.task.yaml -c biom3-test    # provision + setup(build) + run(smoke)
ml logs biom3-test                                  # stream output (re-attach anytime)
ml ssh  biom3-test                                  # shell into the box (debug / interactive)
ml down biom3-test                                  # ⚠ tear down — the cluster stays UP (billing) until you do
```

Edit `mithril/run_tests.task.yaml` first to set `limit_price`, GPU count, and (optionally,
via the commented `envs:` block) S3 weights. The task runs `setup` (build) then `run`
(the smoke); jump to [§7](#7-run-a-biom3-smoke) to see/change what it runs. Steps 1 and
4–7 below are the manual equivalent — use them for the console path (0.a) or for
interactive debugging after `ml ssh`.

## 1. SSH in (only for the console path 0.a — `ml ssh`/the task.yaml replaces this for 0.b)

```bash
chmod 600 <path-to-key.pem>                 # ⚠ required after copying the key, else publickey error
ssh -i <path-to-key.pem> ubuntu@<instance-ip>   # ⚠ user is 'ubuntu' — omitting it = publickey error
```

## 2. Host setup

```bash
nvidia-smi          # expect "NVIDIA B200"; "CUDA Version: 13" is the driver max — fine for our 12.9 image

# run docker without sudo (⚠ so your AWS creds / env vars reach the container):
sudo usermod -aG docker $USER && newgrp docker
docker ps           # should work without sudo now

# AWS CLI v2 (⚠ apt has no 'awscli' package here):
sudo apt-get update && sudo apt-get install -y unzip curl
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip -q awscliv2.zip && sudo ./aws/install && aws --version
```

## 3. AWS credentials (IAM roles → temporary creds)

We use IAM roles, which issue **temporary** credentials (a trio incl. a session token — not
static `AKIA…` keys). On a machine where your AWS access already works, dump them:

```bash
aws configure export-credentials --format env       # prints 3 `export AWS_…` lines
```

Paste those lines on the instance, then:

```bash
export AWS_DEFAULT_REGION=us-east-2      # ⚠ exact spelling: us-east-2 (not use-east-2)
aws sts get-caller-identity              # should print your role ARN
```

⚠ Don't `sudo aws …` — sudo drops your env vars. ⚠ Temp creds expire (1–12 h); refresh if a
long run outlives them.

## 4. Get the code

```bash
git clone https://github.com/addison-nm/BioM3-dev.git
cd BioM3-dev && git checkout feat-docker
```

## 5. Build the image (with awscli for the S3 sync)

```bash
docker/build.sh --awscli --platform linux/amd64
```

First build ≈ 10–40 min (the ~3 GB torch wheel). **To skip rebuilding on future instances,
push once to a registry and pull next time** — see [§9](#9-optional-skip-the-rebuild-next-time).

## 6. Confirm the GPU works in the container

Also settles the Blackwell/`sm_100` question. No data needed:

```bash
docker run --rm --gpus all biom3:gpu python -c \
 "import torch; x=torch.randn(4096,4096,device='cuda'); print('cuda ok:', torch.cuda.get_device_name(0), (x@x).sum().item())"
```

Expect `cuda ok: NVIDIA B200 …`. ⚠ If you get `CUDA error: no kernel image is available`, the
torch build lacks Blackwell kernels — switch the Dockerfile's torch index from `cu129` to
`cu128` and rebuild.

## 7. Run a BioM3 smoke

### 7a. Training — no external data (uses the subset HDF5 baked into the image)

```bash
unset BIOM3_WEIGHTS_URI BIOM3_DATA_URI BIOM3_OUTPUTS_PUSH_URI   # nothing to sync; it's all in the image

docker/run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 1 cuda mithril_test01 \
    --epochs 1 \
    --primary_data_path tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5
```

From-scratch needs no pretrained weights; the 1000-row dataset ships in the image. Results
land in `./outputs/`. A completed step proves: image → GPU → BioM3 training on the B200.

### 7b. Generation — weights from S3, built-in test input

```bash
export BIOM3_WEIGHTS_URI=s3://nm-portal-global-data-955510722784/biom3/weights
# ⚠ ALWAYS narrow the pull — without this it syncs the whole ~40 GB weights tree:
export BIOM3_WEIGHTS_INCLUDES="LLMs/* PenCL/*.bin Facilitator/*.bin ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin"
unset BIOM3_DATA_URI

docker/run.sh biom3_PenCL_inference --input_data_path None \
  --config_path configs/inference/stage1_PenCL.json \
  --model_path weights/PenCL/BioM3_PenCL_epoch20.bin \
  --output_path outputs/pencl.pt --device cuda
```

(Then Stage 2 / Stage 3 follow the same pattern — see [README.md](../docker/README.md#generation-stage-1--2--3).)

## 8. Real training later (when you have a dataset)

The full Stage-2 embedding HDF5 isn't in S3 (the bucket has only `biom3/weights`, no
`biom3/data`). To train for real, upload a dataset to S3, then either set `BIOM3_DATA_URI`
to its prefix or pre-download it and pass `--primary_data_path`.

## 9. Optional: skip the rebuild next time (ECR)

Build once, reuse the image on every future instance:

```bash
ECR=<acct>.dkr.ecr.us-east-2.amazonaws.com
aws ecr create-repository --repository-name biom3 --region us-east-2          # one-time
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin $ECR
docker tag biom3:gpu $ECR/biom3:gpu && docker push $ECR/biom3:gpu

# on a future instance — no build:
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin $ECR
docker pull $ECR/biom3:gpu
BIOM3_IMAGE=$ECR/biom3:gpu docker/run.sh ...
```

## 10. Tear down

Stop/terminate the instance to stop billing — `ml instance delete biom3-test -y` (CLI) or
via the Mithril console. Nothing on the instance persists, so save anything you need first
(results are in `./outputs/`; push them to S3 if you want them).
```bash
aws s3 cp --recursive ./outputs s3://nm-portal-global-data-955510722784/biom3/outputs/mithril_test01
```
