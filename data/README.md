# Data

Place `train.jsonl`, `val.jsonl` and `test.jsonl` under the configured `data_dir`. IDs must be unique across the three splits. Relative image paths are resolved from `image_root`.

| Field | Value |
| --- | --- |
| `id` | Unique record ID |
| `text` | News text |
| `image` | Image path |
| `labels` | Object containing `TI`, `T` and `I` labels |
| `valid_text` | Optional text validity flag |
| `media_valid` | Optional image validity flag |

Labels are `0` for real, `1` for fake and `null` for unknown. Training uses complete inputs and their `TI` labels. Unknown view labels are excluded from evaluation.
Unspecified image validity is checked from the image file. An existing `image_validity` report can be selected with `media_report` in the configuration.

`prepare_data.py` accepts the same record fields and can derive labels from the original FineFake numeric `category` or MDSM manipulation `category` when `labels` is absent. It preserves the IDs and ordering in `configs/splits/`, including the selected MDSM training subset. IDs must match these supplied lists. Images can be verified with `--verify-images`.

```bash
python prepare_data.py --dataset finefake --input-dir data/annotations/finefake --output-dir data/finefake --image-root . --verify-images
```
