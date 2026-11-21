# Ma Trận Nhầm Lẫn & Phân Tích Lỗi — VL-P1

## orb_raw
| GT \ Pred | Obj01 | Obj02 | Obj03 | Obj04 | Obj05 | UNKNOWN | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 3 | 0 | 0 | 0 | 1 | 4 | 2 |
| `object_02` | 0 | 0 | 3 | 0 | 1 | 5 | 1 |
| `object_03` | 1 | 0 | 1 | 0 | 1 | 4 | 3 |
| `object_04` | 1 | 0 | 1 | 0 | 1 | 6 | 1 |
| `object_05` | 1 | 1 | 0 | 0 | 0 | 6 | 2 |
| `unknown` | 8 | 0 | 0 | 3 | 0 | 12 | 2 |
- Giày nhầm thành Dép: 0 | Dép nhầm thành Giày: 1
- FAR theo nhóm confuser âm tính: `{"confuser": {"total": 8, "false_accepted": 3, "far": 0.375}, "footwear_confuser": {"total": 3, "false_accepted": 2, "far": 0.6667}, "household_object": {"total": 3, "false_accepted": 1, "far": 0.3333}, "key_metal_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "stick_confuser": {"total": 3, "false_accepted": 2, "far": 0.6667}, "umbrella_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "wallet_confuser": {"total": 6, "false_accepted": 3, "far": 0.5}}`

## orb_adaptive
| GT \ Pred | Obj01 | Obj02 | Obj03 | Obj04 | Obj05 | UNKNOWN | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 0 | 1 | 1 | 2 | 0 | 6 | 0 |
| `object_02` | 0 | 0 | 0 | 0 | 0 | 10 | 0 |
| `object_03` | 0 | 0 | 0 | 1 | 0 | 9 | 0 |
| `object_04` | 0 | 0 | 0 | 0 | 0 | 10 | 0 |
| `object_05` | 0 | 0 | 0 | 1 | 0 | 9 | 0 |
| `unknown` | 1 | 1 | 0 | 1 | 1 | 21 | 0 |
- Giày nhầm thành Dép: 1 | Dép nhầm thành Giày: 0
- FAR theo nhóm confuser âm tính: `{"confuser": {"total": 8, "false_accepted": 1, "far": 0.125}, "footwear_confuser": {"total": 3, "false_accepted": 0, "far": 0.0}, "household_object": {"total": 3, "false_accepted": 0, "far": 0.0}, "key_metal_confuser": {"total": 1, "false_accepted": 1, "far": 1.0}, "stick_confuser": {"total": 3, "false_accepted": 0, "far": 0.0}, "umbrella_confuser": {"total": 1, "false_accepted": 1, "far": 1.0}, "wallet_confuser": {"total": 6, "false_accepted": 1, "far": 0.1667}}`

## mobilenetv3_small__1nn_max
| GT \ Pred | Obj01 | Obj02 | Obj03 | Obj04 | Obj05 | UNKNOWN | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 6 | 0 | 0 | 0 | 0 | 3 | 1 |
| `object_02` | 0 | 4 | 0 | 0 | 0 | 4 | 2 |
| `object_03` | 0 | 0 | 8 | 0 | 0 | 2 | 0 |
| `object_04` | 0 | 0 | 0 | 2 | 1 | 4 | 3 |
| `object_05` | 0 | 0 | 0 | 0 | 4 | 3 | 3 |
| `unknown` | 4 | 0 | 1 | 0 | 0 | 19 | 1 |
- Giày nhầm thành Dép: 0 | Dép nhầm thành Giày: 1
- FAR theo nhóm confuser âm tính: `{"confuser": {"total": 8, "false_accepted": 2, "far": 0.25}, "footwear_confuser": {"total": 3, "false_accepted": 0, "far": 0.0}, "household_object": {"total": 3, "false_accepted": 0, "far": 0.0}, "key_metal_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "stick_confuser": {"total": 3, "false_accepted": 2, "far": 0.6667}, "umbrella_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "wallet_confuser": {"total": 6, "false_accepted": 1, "far": 0.1667}}`

## mobilenetv3_small__prototype
| GT \ Pred | Obj01 | Obj02 | Obj03 | Obj04 | Obj05 | UNKNOWN | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 5 | 0 | 0 | 0 | 0 | 3 | 2 |
| `object_02` | 0 | 5 | 0 | 0 | 0 | 4 | 1 |
| `object_03` | 0 | 0 | 9 | 0 | 0 | 1 | 0 |
| `object_04` | 0 | 0 | 0 | 4 | 1 | 4 | 1 |
| `object_05` | 0 | 0 | 0 | 0 | 7 | 2 | 1 |
| `unknown` | 2 | 0 | 1 | 0 | 0 | 21 | 1 |
- Giày nhầm thành Dép: 0 | Dép nhầm thành Giày: 1
- FAR theo nhóm confuser âm tính: `{"confuser": {"total": 8, "false_accepted": 0, "far": 0.0}, "footwear_confuser": {"total": 3, "false_accepted": 0, "far": 0.0}, "household_object": {"total": 3, "false_accepted": 0, "far": 0.0}, "key_metal_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "stick_confuser": {"total": 3, "false_accepted": 2, "far": 0.6667}, "umbrella_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "wallet_confuser": {"total": 6, "false_accepted": 1, "far": 0.1667}}`

## mobilenetv3_small__hybrid
| GT \ Pred | Obj01 | Obj02 | Obj03 | Obj04 | Obj05 | UNKNOWN | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 6 | 0 | 0 | 0 | 0 | 4 | 0 |
| `object_02` | 0 | 4 | 0 | 0 | 0 | 6 | 0 |
| `object_03` | 0 | 0 | 9 | 0 | 0 | 1 | 0 |
| `object_04` | 0 | 0 | 0 | 4 | 1 | 4 | 1 |
| `object_05` | 0 | 0 | 0 | 0 | 8 | 2 | 0 |
| `unknown` | 3 | 0 | 2 | 0 | 0 | 20 | 0 |
- Giày nhầm thành Dép: 0 | Dép nhầm thành Giày: 1
- FAR theo nhóm confuser âm tính: `{"confuser": {"total": 8, "false_accepted": 1, "far": 0.125}, "footwear_confuser": {"total": 3, "false_accepted": 0, "far": 0.0}, "household_object": {"total": 3, "false_accepted": 0, "far": 0.0}, "key_metal_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "stick_confuser": {"total": 3, "false_accepted": 2, "far": 0.6667}, "umbrella_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "wallet_confuser": {"total": 6, "false_accepted": 2, "far": 0.3333}}`

## efficientnet_b0__1nn_max
| GT \ Pred | Obj01 | Obj02 | Obj03 | Obj04 | Obj05 | UNKNOWN | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 6 | 0 | 0 | 0 | 0 | 4 | 0 |
| `object_02` | 0 | 5 | 0 | 0 | 0 | 5 | 0 |
| `object_03` | 0 | 0 | 8 | 0 | 0 | 2 | 0 |
| `object_04` | 0 | 0 | 0 | 2 | 2 | 6 | 0 |
| `object_05` | 0 | 0 | 0 | 0 | 7 | 3 | 0 |
| `unknown` | 1 | 0 | 0 | 0 | 0 | 24 | 0 |
- Giày nhầm thành Dép: 0 | Dép nhầm thành Giày: 2
- FAR theo nhóm confuser âm tính: `{"confuser": {"total": 8, "false_accepted": 0, "far": 0.0}, "footwear_confuser": {"total": 3, "false_accepted": 0, "far": 0.0}, "household_object": {"total": 3, "false_accepted": 0, "far": 0.0}, "key_metal_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "stick_confuser": {"total": 3, "false_accepted": 1, "far": 0.3333}, "umbrella_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "wallet_confuser": {"total": 6, "false_accepted": 0, "far": 0.0}}`

## efficientnet_b0__prototype
| GT \ Pred | Obj01 | Obj02 | Obj03 | Obj04 | Obj05 | UNKNOWN | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 8 | 0 | 0 | 0 | 0 | 2 | 0 |
| `object_02` | 0 | 9 | 0 | 0 | 0 | 1 | 0 |
| `object_03` | 0 | 0 | 9 | 0 | 0 | 1 | 0 |
| `object_04` | 0 | 0 | 0 | 3 | 2 | 5 | 0 |
| `object_05` | 0 | 0 | 0 | 0 | 8 | 2 | 0 |
| `unknown` | 1 | 2 | 2 | 0 | 0 | 20 | 0 |
- Giày nhầm thành Dép: 0 | Dép nhầm thành Giày: 2
- FAR theo nhóm confuser âm tính: `{"confuser": {"total": 8, "false_accepted": 2, "far": 0.25}, "footwear_confuser": {"total": 3, "false_accepted": 0, "far": 0.0}, "household_object": {"total": 3, "false_accepted": 0, "far": 0.0}, "key_metal_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "stick_confuser": {"total": 3, "false_accepted": 1, "far": 0.3333}, "umbrella_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "wallet_confuser": {"total": 6, "false_accepted": 2, "far": 0.3333}}`

## efficientnet_b0__hybrid
| GT \ Pred | Obj01 | Obj02 | Obj03 | Obj04 | Obj05 | UNKNOWN | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 8 | 0 | 0 | 0 | 0 | 2 | 0 |
| `object_02` | 0 | 9 | 0 | 0 | 0 | 1 | 0 |
| `object_03` | 0 | 0 | 9 | 0 | 0 | 1 | 0 |
| `object_04` | 0 | 0 | 0 | 3 | 2 | 5 | 0 |
| `object_05` | 0 | 0 | 0 | 0 | 8 | 2 | 0 |
| `unknown` | 1 | 2 | 2 | 0 | 0 | 20 | 0 |
- Giày nhầm thành Dép: 0 | Dép nhầm thành Giày: 2
- FAR theo nhóm confuser âm tính: `{"confuser": {"total": 8, "false_accepted": 2, "far": 0.25}, "footwear_confuser": {"total": 3, "false_accepted": 0, "far": 0.0}, "household_object": {"total": 3, "false_accepted": 0, "far": 0.0}, "key_metal_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "stick_confuser": {"total": 3, "false_accepted": 1, "far": 0.3333}, "umbrella_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "wallet_confuser": {"total": 6, "false_accepted": 2, "far": 0.3333}}`

## dinov2_small_int8__1nn_max
| GT \ Pred | Obj01 | Obj02 | Obj03 | Obj04 | Obj05 | UNKNOWN | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 8 | 0 | 0 | 0 | 0 | 2 | 0 |
| `object_02` | 0 | 10 | 0 | 0 | 0 | 0 | 0 |
| `object_03` | 0 | 0 | 9 | 0 | 0 | 1 | 0 |
| `object_04` | 0 | 0 | 0 | 4 | 0 | 0 | 6 |
| `object_05` | 0 | 0 | 0 | 0 | 9 | 0 | 1 |
| `unknown` | 3 | 1 | 3 | 0 | 0 | 16 | 2 |
- Giày nhầm thành Dép: 0 | Dép nhầm thành Giày: 0
- FAR theo nhóm confuser âm tính: `{"confuser": {"total": 8, "false_accepted": 1, "far": 0.125}, "footwear_confuser": {"total": 3, "false_accepted": 0, "far": 0.0}, "household_object": {"total": 3, "false_accepted": 0, "far": 0.0}, "key_metal_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "stick_confuser": {"total": 3, "false_accepted": 3, "far": 1.0}, "umbrella_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "wallet_confuser": {"total": 6, "false_accepted": 3, "far": 0.5}}`

## dinov2_small_int8__prototype
| GT \ Pred | Obj01 | Obj02 | Obj03 | Obj04 | Obj05 | UNKNOWN | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 9 | 0 | 0 | 0 | 0 | 1 | 0 |
| `object_02` | 0 | 10 | 0 | 0 | 0 | 0 | 0 |
| `object_03` | 0 | 0 | 9 | 0 | 0 | 0 | 1 |
| `object_04` | 0 | 0 | 0 | 7 | 0 | 0 | 3 |
| `object_05` | 0 | 0 | 0 | 0 | 9 | 0 | 1 |
| `unknown` | 3 | 2 | 3 | 1 | 0 | 14 | 2 |
- Giày nhầm thành Dép: 0 | Dép nhầm thành Giày: 0
- FAR theo nhóm confuser âm tính: `{"confuser": {"total": 8, "false_accepted": 2, "far": 0.25}, "footwear_confuser": {"total": 3, "false_accepted": 1, "far": 0.3333}, "household_object": {"total": 3, "false_accepted": 0, "far": 0.0}, "key_metal_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "stick_confuser": {"total": 3, "false_accepted": 3, "far": 1.0}, "umbrella_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "wallet_confuser": {"total": 6, "false_accepted": 3, "far": 0.5}}`

## dinov2_small_int8__hybrid
| GT \ Pred | Obj01 | Obj02 | Obj03 | Obj04 | Obj05 | UNKNOWN | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 7 | 0 | 0 | 0 | 0 | 3 | 0 |
| `object_02` | 0 | 7 | 0 | 0 | 0 | 3 | 0 |
| `object_03` | 0 | 0 | 8 | 0 | 0 | 2 | 0 |
| `object_04` | 0 | 0 | 0 | 7 | 0 | 3 | 0 |
| `object_05` | 0 | 0 | 0 | 0 | 10 | 0 | 0 |
| `unknown` | 1 | 0 | 2 | 0 | 0 | 22 | 0 |
- Giày nhầm thành Dép: 0 | Dép nhầm thành Giày: 0
- FAR theo nhóm confuser âm tính: `{"confuser": {"total": 8, "false_accepted": 0, "far": 0.0}, "footwear_confuser": {"total": 3, "false_accepted": 0, "far": 0.0}, "household_object": {"total": 3, "false_accepted": 0, "far": 0.0}, "key_metal_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "stick_confuser": {"total": 3, "false_accepted": 1, "far": 0.3333}, "umbrella_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "wallet_confuser": {"total": 6, "false_accepted": 2, "far": 0.3333}}`

## ensemble_mv3_dino__hybrid
| GT \ Pred | Obj01 | Obj02 | Obj03 | Obj04 | Obj05 | UNKNOWN | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 8 | 0 | 0 | 0 | 0 | 2 | 0 |
| `object_02` | 0 | 7 | 0 | 0 | 0 | 3 | 0 |
| `object_03` | 0 | 0 | 8 | 0 | 0 | 2 | 0 |
| `object_04` | 0 | 0 | 0 | 6 | 0 | 4 | 0 |
| `object_05` | 0 | 0 | 0 | 0 | 9 | 0 | 1 |
| `unknown` | 1 | 0 | 2 | 0 | 0 | 22 | 0 |
- Giày nhầm thành Dép: 0 | Dép nhầm thành Giày: 0
- FAR theo nhóm confuser âm tính: `{"confuser": {"total": 8, "false_accepted": 0, "far": 0.0}, "footwear_confuser": {"total": 3, "false_accepted": 0, "far": 0.0}, "household_object": {"total": 3, "false_accepted": 0, "far": 0.0}, "key_metal_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "stick_confuser": {"total": 3, "false_accepted": 1, "far": 0.3333}, "umbrella_confuser": {"total": 1, "false_accepted": 0, "far": 0.0}, "wallet_confuser": {"total": 6, "false_accepted": 2, "far": 0.3333}}`

