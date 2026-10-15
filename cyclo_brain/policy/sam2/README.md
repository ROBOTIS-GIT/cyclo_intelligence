# SAM2 annotation workflow

Current UI: **Autonomy → Annotate → SAM Fine-tuning / Auto Labeling**.
See [WORKFLOW_KO.md](WORKFLOW_KO.md) for reviewed examples, training, automatic drafts,
review, model storage, and the limitations of the object-specific decoder approach.

The sections below describe the original click engine and API; the current dataset
stores regions inside each map/episode's `samples.jsonl`, not a separate regions log.


Autonomy → Collect → Annotation provides click-guided masks on **saved** camera
frames. The first saved frame in an episode is selected automatically; Saved frame
selects another. Live Observation and Qwen descriptions stay independent.

1. Click **Start episode**, then **Capture frame**. The saved image opens in
   Annotation; Qwen does not need to run. To label a Qwen failure instead, click
   **Save analyzed frame** to preserve the exact analyzed image and result.
2. Select **Mission** once at the top of Annotation, if the mask relates to a mission.
3. Click the object with **Include point**; **Exclude point**, **Undo** and **Reset** correct
   the preview. SAM2 loads automatically. A failed request offers **Retry mask**.
4. Enter **Object name**, inspect the visible overlay, then click **Save mask**.
   This explicit save approves the mask; clicks alone never save training data.
5. **Saved mask** reopens an existing mask or starts a **New mask**. Correct its
   points and save to append a revision. Original images and prior revisions remain.

The initial UI omits reviewer fields, role selection, separate mission relevance,
review checkboxes, rejection and model lifecycle controls. New saves record
`annotator: collect-ui` as a generic UI source, not an identified person's name.
New masks default to `target`; editing a saved mask preserves its role. The shared
Mission selector supplies `mission_id`. Only loaded image/overlay previews can save.
The backend data schema and advanced APIs remain compatible with existing records.

## Image view

- **Expand view** opens the image and point tools fullscreen; **Exit expanded view**
  or Escape returns to Annotation without losing points or the preview.
- **Zoom in / Zoom out** adjust 100–800% relative to the fitted image.
- Drag to pan while zoomed. A drag never adds a SAM2 point.
- **Fit image** restores the full image. Zoom/pan affect display only; saved clicks,
  masks and source images always retain original pixel coordinates and dimensions.

## Runtime

- Official Meta `facebook/sam2.1-hiera-tiny` (SAM 2.1 Tiny, Apache-2.0).
- Model revision: `de431c4043854a71d8101e17995dfe596bf101a5`.
- Official source revision: `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- PyTorch 2.10 / CUDA 12.8, desktop AMD64 NVIDIA GPU, port 7111 on loopback.
- Optional connected-components CUDA extension is disabled; mask hole/sprinkle
  postprocessing thresholds are zero. Point-prompt image inference remains enabled.
- One worker/one predictor; concurrent requests fail with 409. The last image's
  embedding is reused for additional clicks. Frames are checksum verified.
- No camera subscription or robot commands. Supervisor sends only the chosen saved JPEG.
- The unload API releases the model; a later click loads it again. Stopping Qwen does
  not stop SAM2. The full SAM2 container can be stopped via the helper below.

```bash
# From cyclo_intelligence/
docker/container.sh start-sam2 --build
docker/container.sh logs-sam2
docker/container.sh stop-sam2
```

The first build downloads source/dependencies; first inference downloads the pinned
checkpoint into `docker/huggingface`. Preview data is held in supervisor memory
(maximum eight previews, 30-minute TTL); it does not enter the dataset until approval.
If preview expired or the supervisor restarted, Generate mask and review again.

## Data

Root: `docker/workspace/dataset/reasoning/` on host,
`/workspace/dataset/reasoning/` inside the main container.

```text
raw/<episode_id>/
  images/<frame_id>.jpg              # existing immutable source JPEG
  regions.jsonl                     # append-only object/review revisions
  masks/<object_id>/<revision>.png   # opaque black/white RGBA mask, source dimensions
```

Each region includes original image SHA-256, frame ID, object ID/label, role,
optional mission ID, original-pixel include/exclude points, uncompressed COCO RLE
(`size=[height,width]`, column-major counts starting with background), pixel area,
`bbox_xywh`, mask path/hash, annotator, review status/revision, SAM2 model/source/
checkpoint revisions and inference duration. PNG foreground is white, background
black; display overlay is derived separately. Original JPEG bytes are never edited.

Export approved labels also exports approved masks to `regions.jsonl`, `masks/`
and `images/`, including mask-only episodes. Region split follows the same scene-group assignment as decision samples. Rejected regions are excluded. Region data
is a separate artifact: Qwen decision JSONL still uses its original image/context
contract, and no mask is automatically used as attention supervision. A future
region/grounding training adapter can consume these reviewed masks.

## API

- `POST /api/collection/episodes/{id}/segment`: frame_id and points → temporary preview
- `GET/POST /api/collection/episodes/{id}/regions`: list / approve a cached preview
- `POST /api/collection/episodes/{id}/regions/{object_id}/reject`
- `GET /api/collection/episodes/{id}/regions/{object_id}/overlay`
- `POST /api/collection/segmentation/unload`

Save requires `preview_id`, matching `frame_id`, `object_id`, `expected_revision`,
`object_label`, `annotator`, and `reviewed=true`. Server derives mask bytes from its
own validated preview, not from a browser-submitted mask. Revision conflicts are
rejected; reload the frame before retrying. Storage errors return 507.

## Cancel fine-tuning

During fine-tuning, the stop icon next to Fine-tune cancels the current SAM training job. The worker stops at the next feature-cache or optimizer-step boundary and releases the engine before reporting `Cancelled`. Annotation serving, source images, masks, and completed models remain available.

`POST /api/collection/segmentation/training/cancel` accepts `{"job_id":"<current training ID>"}`. The SAM worker handles `/adaptation/cancel`; stale IDs are rejected. The UI keeps training locked while cancellation is pending.

## Verification

- Unit tests cover click scaling/letterboxes, foreground/background prompts, review
  gating, stale/expired previews, image identity, RLE, revisions, disk failure and export.
- A real GPU smoke test used a temporary 320×240 image dataset, generated a SAM2
  mask, saved it and exported one approved region. It did not alter user episodes.
- Browser verification mocks robot/network APIs and uses a real SAM2 overlay image.

Official source: https://github.com/facebookresearch/sam2
Official model: https://huggingface.co/facebook/sam2.1-hiera-tiny
