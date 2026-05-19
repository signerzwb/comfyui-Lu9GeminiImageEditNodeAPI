# lu9-Gimagev1

ComfyUI custom nodes for GPT Image 2 generation and editing over an OpenAI-compatible image API.

Included nodes:

- `lu9-Gimage Config`
- `lu9-Gimage Unified`

Current scope:

- Text-to-image when no reference image is connected
- Image edits when one or more reference images are connected
- Up to 9 reference images
- Optional mask for edit mode
- `1k / 2k / 4k` resolution tiers
- Optional `4k -> 2k` fallback
- Optional `skip_error` to keep workflow running and return parseable error text
- Retry by full rounds
- Manual request URL input in config node; no built-in hardcoded endpoint fallback

Current routing:

- No reference image connected: `/images/generations`
- One or more reference images connected: `/images/edits`

Auto model routing:

- `1k` -> `gpt-image-2` with `api_key_1k`
- `2k / 4k` -> `gpt-image-2-vip` with `api_key_vip`

Supported aspect ratios by tier:

- `1k`: `1:1`, `3:4`, `4:3`, `9:16`, `16:9`
- `2k`: `1:1`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `9:21`, `21:9`
- `4k`: `1:1`, `4:5`, `5:4`, `9:16`, `16:9`, `9:21`, `21:9`

Retry behavior:

- `retry_count` means extra full rounds, not extra single calls.
- When `resolution=4k` and `fallback_4k_to_2k=true`, one round is:
  1. try `4k`
  2. if failed, try `2k`

Special size notes:

- `4k 21:9` -> `3696x1584`
- `4k 9:21` -> `1584x3696`

Skip error behavior:

- When `skip_error=false`, node failures raise an exception and interrupt the workflow.
- When `skip_error=true`, the node returns:
  - a blank placeholder image
  - empty `image_url`
  - `response` JSON containing `status=error`
  - `status_text` containing `ERROR FAILED`
