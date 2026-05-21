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
#   IMAGE_FAMILY  — default: pytorch-2-9-cu129-ubuntu-2204-nvidia-580 (DLVM)
#   IMAGE_PROJECT — default: deeplearning-platform-release
#
# Behavior:
#   SPOT=1 (default)  → --provisioning-model=SPOT + auto-delete on preempt.
#                        Cheap (~$0.17/hr) but the VM can vanish mid-run.
#   SPOT=0           → on-demand instance, no preemption risk (~$0.55/hr).
#                        Use for runs > ~20 min to avoid losing progress.
#
# Cost note: T4 spot in us-central1 is ~$0.11/hr GPU + ~$0.06/hr CPU+disk.
# A LoRA fine-tune (30 epochs, batch 8) takes ~25 min on T4 → well under $1
# either way. A from-scratch ngf=96 60-epoch run takes ~75 min → ~$0.70
# on-demand vs ~$0.21 spot (but the spot run can be lost partway).

set -euo pipefail

: "${PROJECT:?PROJECT env var must be set to a GCP project id}"
VM_NAME="${VM_NAME:-albumify-train}"
ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"
DISK_GB="${DISK_GB:-100}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
SPOT="${SPOT:-1}"

if [ "$SPOT" = "1" ]; then
  PROVISIONING_FLAGS=(--provisioning-model=SPOT --instance-termination-action=DELETE)
  PROVISIONING_LABEL="spot, auto-delete on preempt"
else
  PROVISIONING_FLAGS=(--provisioning-model=STANDARD)
  PROVISIONING_LABEL="on-demand"
fi

echo ">>> Creating $VM_NAME in $ZONE on project $PROJECT ($PROVISIONING_LABEL)"
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="${DISK_GB}GB" \
  --maintenance-policy=TERMINATE \
  "${PROVISIONING_FLAGS[@]}" \
  --metadata="install-nvidia-driver=True" \
  --scopes=cloud-platform

echo
echo ">>> Wait ~30s for the VM to become reachable, then SSH in:"
echo "    gcloud compute ssh $VM_NAME --zone $ZONE --project $PROJECT"
echo
echo ">>> Delete the VM when done (do NOT leave it running):"
echo "    gcloud compute instances delete $VM_NAME --zone $ZONE --project $PROJECT"
