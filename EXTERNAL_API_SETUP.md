# External classifier API setup

If local TensorFlow model is unavailable, the application can classify images via an external HTTP API.

## Environment variables

- `EXTERNAL_CLASSIFIER_API_URL` (required for external mode)
- `EXTERNAL_CLASSIFIER_API_TOKEN` (optional bearer token)
- `EXTERNAL_CLASSIFIER_TIMEOUT` (optional, default: `20` seconds)

## Expected API contract

`POST` multipart request with file field named `file`.

Response JSON should contain one of:
- `classification`
- `label`

Example response:

```json
{ "classification": "Plastic" }
```

## Example

```bash
export EXTERNAL_CLASSIFIER_API_URL="https://example.com/infer"
export EXTERNAL_CLASSIFIER_API_TOKEN="<token>"
python app.py
```
