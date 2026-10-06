# V2 Smoke Test Report

## Dataset
- Load manifest: True
- Class counts (0,1,2,3): {'0': 100, '1': 100, '2': 100, '3': 100}
- All classes present (100 each): True
- All files exist: True (Example missing: None)
- No duplicates: True
- Subject separation (no overlap): True

## Models (96x96 resolution)
- CNN2D: pass
- DenseNet2D: pass
- ViT2D: pass

## Checkpoint Validation
- Save/Load & Metadata Validation: {'save_load': True, 'valid_match': True, 'match_reason': 'metadata matches current configuration', 'valid_reject': True, 'reject_reason': "metadata mismatch on 'image_size': checkpoint has [96, 96], current run has [128, 128]"}

## V1 Integrity Hashes
- final_400.csv: `cd62ce417cc29a7f56682366e2fbf83f337c3ec7dd5efe906a4d4c478e2a0f59`
- split_assignments.csv: `70246e6e39f4694b52c7a80f4871f1f48d831d8d04a92007ca8d52b06a4981b9`
