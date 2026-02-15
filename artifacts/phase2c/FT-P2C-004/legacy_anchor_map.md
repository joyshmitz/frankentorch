# FT-P2C-004 — Legacy Anchor Map

Packet: Autograd engine scheduling  
Legacy root: `legacy_pytorch_code/pytorch`

## Extracted Anchors (Exact)

| Legacy path | Line anchor | Symbol | Porting relevance |
|---|---:|---|---|
| `torch/csrc/autograd/engine.h` | 31 | `MAX_DEPTH` (reentrant depth cap) | explicit reentrant depth policy boundary |
| `torch/csrc/autograd/engine.h` | 51 | `struct NodeTask` | ready-task payload and ordering unit |
| `torch/csrc/autograd/engine.h` | 90 | `CompareNodeTaskTime` | deterministic task ordering/tie-break anchor |
| `torch/csrc/autograd/engine.h` | 210 | `compute_dependencies(Node* root, GraphTask& task, ...)` | dependency graph pre-pass contract |
| `torch/csrc/autograd/engine.h` | 229 | `thread_main(const std::shared_ptr<GraphTask>&)` | scheduler event-loop shape |
| `torch/csrc/autograd/engine.h` | 230 | `reentrant_thread_init()` | reentrant worker bootstrap path |
| `torch/csrc/autograd/engine.cpp` | 232 | `ReadyQueue::push` / `ReadyQueue::pop` | queue semantics and monotone task draining |
| `torch/csrc/autograd/engine.cpp` | 518 | `Engine::thread_main(...)` | pop-evaluate-push scheduler loop |
| `torch/csrc/autograd/engine.cpp` | 618 | `Engine::reentrant_thread_init()` | reentrant-depth specific execution path |
| `torch/csrc/autograd/engine.cpp` | 1248 | `Engine::compute_dependencies(...)` | concrete dependency counting traversal |
| `torch/csrc/autograd/engine.cpp` | 1286 | `Engine::execute(...)` | execution entrypoint and graph-task setup |
| `torch/csrc/autograd/engine.cpp` | 1333 | `GraphTask(... reentrant_depth ...)` | strict/hardened reentrant-depth contract boundary |

## Behavioral Oracle Tests (Scoped)

| Legacy path | Intent |
|---|---|
| `test/cpp/api/autograd.cpp` | reentrant backward priority and ordering behavior |
| `test/test_autograd.py` | backward graph scheduling and deterministic gradient semantics |
| `test/inductor/test_compiled_autograd.py` | deep reentrant behavior probes used as compatibility context only |

## Implemented Rust Mapping

| Rust crate | File | Mapping |
|---|---|---|
| `ft-autograd` | `crates/ft-autograd/src/lib.rs` | deterministic dependency scheduler, ready queue tie-break, strict/hardened reentrant policy, scheduler telemetry |
| `ft-api` | `crates/ft-api/src/lib.rs` | mode-aware `BackwardOptions` propagation and report forwarding |
| `ft-conformance` | `crates/ft-conformance/src/lib.rs` | scheduler conformance fixtures, differential comparators, strict/hardened policy assertions |

## Extraction Schema (Mandatory)

1. `packet_id`: `FT-P2C-004`
2. `legacy_paths`: `torch/csrc/autograd/engine.h`, `torch/csrc/autograd/engine.cpp`
3. `legacy_symbols`: `NodeTask`, `ReadyQueue::push/pop`, `compute_dependencies`, `thread_main`, `reentrant_thread_init`, `Engine::execute`
4. `tensor_storage_contract`: unchanged scalar tensor storage contract inherited from `FT-P2C-001`
5. `dispatch_contract`: unchanged key routing semantics; scheduler consumes already-resolved graph structure
6. `error_contract`: `ReentrantDepthExceeded` and `DependencyUnderflow` remain explicit and fail-closed in strict mode
7. `grad_graph_contract`: dependency-driven deterministic replay with stable execution-order telemetry
8. `serialization_contract`: no packet-local format change
9. `strict_mode_policy`: overflow fails closed (`ReentrantPolicy::StrictFail`)
10. `hardened_mode_policy`: bounded overflow fallback with explicit telemetry (`reentrant_guard_triggered`, `hardened_fallback_used`)
11. `excluded_scope`: multithreaded worker pools, CUDA streams, distributed graph-task futures
12. `oracle_tests`: `test/test_autograd.py`, `test/cpp/api/autograd.cpp` (scoped scheduling semantics)
13. `performance_sentinels`: `queue_pushes`, `queue_pops`, `max_queue_len`, scheduler tail latency
14. `compatibility_risks`: full PyTorch thread/device scheduling behavior is broader than scoped single-thread packet semantics
15. `raptorq_artifacts`: packet parity sidecar + decode proof remain required in final evidence bead (`bd-3v0.15.9`)
