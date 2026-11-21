# Git Baseline Preservation & Worktree Isolation Report — VL-P2

## 1. Protected Commits & Worktree Inventory

All historical milestones and production code remain untouched on their respective branches and worktrees:

| Commit | Worktree / Path | Branch | Status | Scope Preserved |
|---|---|---|---|---|
| `8ae6f85` | Ancestor commit | `hotfix/english-tables-layout-and-tts` | Intact & Immutable | English Tables Layout & OneCore TTS |
| `95a75e0` | Ancestor commit | `hotfix/od-1-foundation` | Intact & Immutable | OD-1 Offline YOLOX-Nano Detection |
| `6e3039f` | `VisionMate_VL_P1` | `research/vl-p1-manual` | Untouched | Static preview hotfix + manual VL-P1 base |
| `65c2a27` | `VisionMateGlasses` | `hotfix/webcam-preview-fps-optimization` | Clean & Unmodified | Production Windows app (`app/`) |
| `988fdf1` | `VisionMate_VL_P1_AG` | `research/vl-p1-antigravity` | Clean & Unmodified | VL-P1 Automated Few-Shot Pilot (`34` tests pass) |
| `dacd245` | `VisionMate_VL_P1b_AG` | `research/vl-p1b-reliability-proposal` | Clean & Unmodified | VL-P1b Proposal & Hard-Neg Pilot (`10` tests pass) |
| `9587ad4` | `VisionMate_VL_P1c_AG` | `research/vl-p1c-calibration-multiproposal` | Clean & Unmodified | VL-P1c Calibration & Multi-Proposal Pilot (`20` tests pass) |
| **VL-P2** | `VisionMate_VL_P2_AG` | `research/vl-p2-white-cane-segmentation` | **Active Research Worktree** | Isolated inside `research/vl_p2/` only |

## 2. Isolation Guarantees

- `git diff --name-only 9587ad4 -- app research/vl_p1 research/vl_p1b research/vl_p1c` returns empty output (`""`).
- No merge into `main`, no branch deletion, no git tag creation, and no remote push were performed.
- Large binary datasets, PNG overlays, contact sheets, and ONNX weights remain excluded from Git history via `research/vl_p2/.gitignore`.
