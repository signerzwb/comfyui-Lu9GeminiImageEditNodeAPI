# comfyui-lu9-VisonApi-9images

Independent ComfyUI custom node for OpenAI-compatible vision and image analysis.

Included node:

- `lu9-VisionApi 9images`

Behavior:

- standalone plugin, separate from `comfyui-lu9-Gimagev1`
- supports up to `9` images
- primary provider group:
  - `api_url`
  - `model`
  - `api_key_1`
  - `api_key_2`
  - `api_key_3`
- backup provider group:
  - `backup_api_url`
  - `backup_model`
  - `backup_api_key_1`
  - `backup_api_key_2`
  - `backup_api_key_3`
- all url, model, and key fields default to empty strings
- automatic fallback flow:
  - try the primary group first
  - inside the primary group, try `api_key_1 -> api_key_2 -> api_key_3`
  - if the primary group fails as a whole, switch to the backup group
  - inside the backup group, try `backup_api_key_1 -> backup_api_key_2 -> backup_api_key_3`
- sends requests to `chat/completions`
- each image is sent as `data:image/png;base64,...`
- inserts `[Image 1] ... [Image 9]` markers automatically before each image
- returns:
  - `response`
  - `raw_response`
  - `status_text`
