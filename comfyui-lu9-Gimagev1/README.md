# lu9-Gimagev1

ComfyUI custom nodes for GPT Image 2 generation and editing over an OpenAI-compatible image API.

Included nodes:

- `lu9-Gimage Config`
- `lu9-Gimage Unified`

Current scope:

- Text-to-image when no reference image is connected
- Image edits when one or more reference images are connected
- Up to 12 reference images
- Optional mask for edit mode
- `1k / 2k / 4k` resolution tiers
- `custom` aspect mode with explicit width/height
- `auto` aspect mode that matches the closest supported ratio from `image1`
- `smartauto` aspect mode that tries to preserve `image1` ratio with a legal snapped size first
- Optional `4k -> 2k` fallback
- Optional `skip_error` to keep workflow running and return parseable error text
- `seed` input with ComfyUI `control_after_generate` support
- `output_mode` switch for normal image output or URL-only workflows
- Retry by full rounds
- Manual request URL input in config node; no built-in hardcoded endpoint fallback

Current routing:

- No reference image connected: `/images/generations`
- One or more reference images connected: `/images/edits`
- In edit mode, all connected reference images are uploaded as repeated `image` form fields
- Response parsing now prefers `b64_json` and falls back to `url` download only when needed

Auto model routing:

- `1k` -> `gpt-image-2` with `api_key_1k`
- `2k / 4k` -> `gpt-image-2-vip` with `api_key_vip`

Supported aspect ratios by tier:

- `1k`: `1:1`, `3:4`, `4:3`, `9:16`, `16:9`
- `2k`: `1:1`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `9:21`, `21:9`
- `4k`: `1:1`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `9:21`, `21:9`
- `auto`: picks the nearest supported ratio for the selected resolution based on `image1`
- `smartauto`: first tries a custom size derived from `image1`, then falls back to nearest preset ratio
- `custom`: any size that passes local validation and API acceptance

Retry behavior:

- `retry_count` means extra full rounds, not extra single calls.
- When `resolution=4k` and `fallback_4k_to_2k=true`, one round is:
  1. try `4k`
  2. if failed, try `2k`

Special size notes:

- `4k 3:4` -> `2432x3264`
- `4k 4:3` -> `3264x2432`
- `4k 21:9` -> `3696x1584`
- `4k 9:21` -> `1584x3696`
- tested larger custom `3:4` -> `2432x3264`
- matching custom `4:3` -> `3264x2432`

Custom mode behavior:

- `aspect_ratio=custom` uses `custom_width` and `custom_height`
- custom mode always routes to `api_key_vip`
- custom mode ignores `fallback_4k_to_2k`
- custom width/height must be multiples of `64`
- custom width/height must not exceed `3840`
- custom total pixels must be within `655360..8294400`
- custom aspect ratio must not exceed `3:1`

Auto mode behavior:

- `aspect_ratio=auto` requires `image1`
- auto mode reads the width/height ratio of `image1`
- auto mode picks the closest supported preset ratio for the chosen `resolution`
- the resolved ratio is returned in `response` as `resolved_aspect_ratio`

Smartauto mode behavior:

- `aspect_ratio=smartauto` requires `image1`
- smartauto first tries to keep the source ratio more accurately
- the resolved size is snapped to multiples of `64`
- the resolved size must stay within the current resolution tier pixel/edge limits
- if smartauto cannot derive a legal custom size, it falls back to the nearest preset ratio
- the response includes:
  - `resolved_size_strategy`
  - `resolved_custom_width`
  - `resolved_custom_height`

Skip error behavior:

- When `skip_error=false`, node failures raise an exception and interrupt the workflow.
- When `skip_error=true`, the node returns:
  - a blank placeholder image
  - empty `image_url`
  - `response` JSON containing `status=error`
  - `status_text` containing `ERROR FAILED`

Seed behavior:

- `seed` is exposed as a node input so ComfyUI can treat each run as changed
- `seed` uses `control_after_generate=true`, so the widget supports normal rerun/randomize behavior
- current `llms.best` GPT image docs do not expose a public `seed` request parameter here
- therefore this plugin currently uses `seed` for ComfyUI rerun control, not API determinism

Output mode behavior:

- `image_and_url`:
  - default mode
  - plugin returns normal `IMAGE`
  - plugin prefers `b64_json`, then falls back to `url`
- `url_only`:
  - plugin only requires a usable `url` from the API response
  - plugin does not decode/download the image
  - `IMAGE` output returns a blank placeholder tensor
  - useful when downstream logic only needs `image_url`
