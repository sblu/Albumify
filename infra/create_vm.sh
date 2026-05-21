#!/usr/bin/env bash
# Spin up a GCP T4 spot VM in us-central1 for the Albumify LoRA fine-tune.
#
# Required env vars:
#   PROJECT       — your GCP project id
# Optional env vars:
#   VM_NAME       — default: albumify-train
#   ZONE          — default: us-central1-a
#   MACHINE_TYPE  — default: n1-standard-4
#   GPU_TYPE      — default: nvidia-tesla-t4
#   GPU_COUNT     — default: 1
#   DISK_GB       — default: 100
#   IMAGE_FAMILY  — default: pytorch-latest-gpu (Deep Learning VM)
#   IMAGE_PROJECT — default: deeplearning-platform-release
#
# Behavior:
#   --provisioning-model=SPOT         (cheaper, preemptible)
#   --instance-termination-action=DELETE   (auto-delete on preemption so we
#                                           don't pay a stopped-instance bill)
#
# Cost note: T4 spot in us-central1 is ~$0.11/hr GPU + ~$0.06/hr CPU+disk.
# A full fine-tune at ~471 imgs/epoch × 30 epochs at batch 8 takes ~25 min
# on T4, so total bill is well under $1.

set -euo pipefail

: "${PROJECT:?PROJECT env var must be set to a GCP project id}"
VM_NAME="${VM_NAME:-albumify-train}"
ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"
DISK_GB="${DISK_GB:-100}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-latest-gpu}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

echo ">>> Creating $VM_NAME in $ZONE on project $PROJECT (spot, auto-delete on preempt)"
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="${DISK_GB}GB" \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --metadata="install-nvidia-driver=True" \
  --scopes=cloud-platform

echo
echo ">>> Wait ~30s for the VM to become reachable, then SSH in:"
echo "    gcloud compute ssh $VM_NAME --zone $ZONE --project $PROJECT"
echo
echo ">>> Delete the VM when done (do NOT leave it running):"
echo "    gcloud compute instances delete $VM_NAME --zone $ZONE --project $PROJECT"
