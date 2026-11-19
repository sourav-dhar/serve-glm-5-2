# Serving GLM-5.2 on Modal

## Research question

At what offered load does multi-token prediction (MTP) stop improving the cost per million SLO-compliant output tokens for GLM-5.2? Measure whether a change in prompt length, prefix reuse, or expert precision moves that crossover. A reproducible negative result is a valid outcome. The settings below are proposed experiments. GPU measurements are pending.

## Starting point

The supplied folder is an unpacked MIT-licensed Vizuara AI Labs capstone. Its website identifies the upstream project. Keep the original copyright notice with reused material and describe the specific adaptation in each relevant PR. The portfolio contribution starts with the measurement contract, Modal deployment, controlled comparisons, and resulting engineering decisions. [Upstream project named in the snapshot](https://github.com/VizuaraAI/capstone-serve-glm-5-2)

| Area | Existing behavior | Implication for this project |
| --- | --- | --- |
| `gateway/` | FastAPI checks identity and rate limits, admits requests into bounded capacity, selects an upstream, retries before streaming, and emits Prometheus metrics. Admission and rate state live in the process. | Compare direct engine and gateway on the same replay. One gateway process is enough for the first comparison. Inspect cancellation, truncated streams, and slot release before scaling out. |
| `engine/` | Shell launcher, live/readiness probes, parallelism and quantization notes. The launcher defaults to the base model ID, requests FP8 quantization, and always enables three draft tokens. | Select the published native-FP8 checkpoint explicitly. Give MTP its own switch. Record effective engine configuration and one successful decode. |
| `harness/`, `loadtest/` | Async streaming client, local mock, SLO YAML, fixed-concurrency sweeps and GuideLLM arrival-rate scripts. The Python spike triples concurrency. | Keep closed-loop concurrency and open-loop arrival rate as separate axes. Fixed-concurrency traffic slows its own arrivals when the service saturates. |
| `analysis/` | Computes a concurrency ceiling, goodput plots, and cost per million good output tokens. The ceiling and cost scripts select operating points differently. | Report the highest contiguous full-SLO passing load and separately the best measured cost among passing settings. Name both in outputs. |
| `deploy/`, `envoy/` | Kubernetes workloads, KEDA/HPA, KServe, Ray Serve and an alternate proxy. Ray references a `glm_serve:build_app` entry point absent from the snapshot. | Keep these as reference designs. The first production path has one orchestrator, Modal. |
| `observability/`, root Compose and Docker files | Prometheus rules, Grafana dashboard JSON, and a CPU mock stack. Compose mounts the scrape configuration. The supplied scrape configuration lacks a rule file declaration. | Verify an actual scrape and alert evaluation before treating dashboards or alerts as evidence. |
| `tests/`, `docs/`, `website/`, root specs | CPU checks exercise gateway, shedding and SLO arithmetic. The architecture notes, runbook and public-facing capstone describe the design. | Label mock results as protocol checks. Link portfolio claims to a raw experiment and commit. |

## Architecture and first deployment

Start with one fixed Modal replica on one 8 × H200 node, tensor parallelism 8, and one vLLM endpoint. Use the upstream native FP8 weights and FP8 KV recipe, MTP off, prefix caching off, and an initial 8,192-token context cap. Pin the engine image digest, checkpoint and tokenizer revisions before launch. Confirm that the chosen image and attention backend accept each flag and record the effective settings. The upstream recipe gives a working hardware target. Modal account capacity and this deployment's memory fit still need a smoke test. [vLLM GLM-5.2 recipe](https://recipes.vllm.ai/zai-org/GLM-5.2) · [Native FP8 checkpoint](https://huggingface.co/zai-org/GLM-5.2-FP8) · [Modal GPU types](https://modal.com/docs/guide/gpu)

1. **CPU preparation.** Check shard bytes from the checkpoint index, image compatibility, quota, and the maximum bill at the current rate. Stage the immutable checkpoint on a persistent Modal Volume. Record cold fetch, cached load and first decode separately. Require a run budget and wall-time limit before allocating GPUs.
2. **Engine control.** Run the load generator from an independent CPU client in a recorded location. Send streaming requests directly to an authenticated engine endpoint. Limit the replica count to one and configure idle scale-down. Log per-rank free HBM after load, allocated cache blocks, actual GPU SKU/count, startup time and model identity. Stop on an incorrect response, OOM, missing usage telemetry, or budget expiry.
3. **Gateway treatment.** Route the same workload through one CPU gateway in front of the same warm engine. Measure client-side latency and stream integrity as well as gateway queue time. Keep engine batching independent of gateway admission. Validate disconnect propagation, bounded queueing and recovery after a burst. Add orchestration complexity only when a measured bottleneck motivates a separate issue.

## Measurement contract

The first workload uses tokenizer-verified 512-token inputs and a 256-token generation budget. Record actual generated lengths, reasoning settings, stop reasons and truncation. For a fixed-length throughput trial, explicitly enforce and verify the length in both arms. For task-quality evaluation, allow natural completion. Start with distinct prompts and record prefix-cache state. Use separate SLOs for long prompts.

* Score TTFT at first generated token, tracking first reasoning token and first visible answer separately where applicable. Preserve role frames, terminal signals, usage and upstream error events in protocol fixtures. Time between SSE chunks measures inter-chunk latency. Use token-level engine instrumentation for true inter-token latency or label the approximation. Obtain token counts from final engine usage and reconcile against client records, including the treatment of reasoning tokens.
* For the short workload, adopt provisional targets from the snapshot: p95 TTFT ≤ 2 s, p99 end-to-end ≤ 20 s, valid completions/offered ≥ 99%, and per-request goodput/offered ≥ 95%. A good request completes correctly within both latency bounds. Count 429s in the offered denominator. Report shed, errored, timed out and truncated separately, plus success among admitted requests. Use unrounded values for pass/fail. The first failing full-SLO load bounds the contiguous sustainable range.
* Compute useful tokens/s as generated tokens from good requests divided by measured wall time. Compute cost per million as attributed dollars × 1,000,000 / useful tokens. Report warm steady-state cost and total billed experiment cost including startup and idle time. Zero useful tokens yields no finite cost. Persist the input/output mix for any hosted-price comparison.
* First run a closed-loop sweep at concurrency 1, 2, 4, 8 and upward until failure. Then replay fixed or seeded Poisson arrivals around the observed knee and run a 3× arrival-rate burst followed by recovery. Record intended send time, actual send time, arrival lag and outstanding work so client saturation is visible.
* Predeclare prompts, quality thresholds, warm-up, measurement interval, seeds and stopping rule. Randomize treatment order within each replicate. Start with three independent paired repetitions and report per-run values and uncertainty. Near the boundary extend runs until the p99 has at least 1,000 completed observations per configuration and report tail uncertainty using contiguous time blocks. Mark results inconclusive if the time or cost cap intervenes.
* Each run gets an immutable ID, Git SHA, image digest, model/tokenizer SHA, full flags, hardware/topology, price timestamp, workload hash, raw request records and engine counters. Redact credentials and use redistributable prompts. Attach raw artifacts with checksums and commit a small manifest plus a regeneration command. Retain failures.

## Ablations in dependency order

| Study | Paired comparison | What would change the decision |
| --- | --- | --- |
| MTP and load | Off vs 1, 3 and 5 draft tokens with the same native FP8 weights, cache dtype, workload, TP and node. Sweep short interactive and decode-heavy traffic. | Plot useful tokens/$ against offered rate. Record accepted tokens per draft step, TTFT and decode time. Keep MTP where its benefit survives contention and quality checks. |
| Prefill interference | Sweep chunked-prefill token budget at fixed scheduler limits. Interleave short chats with 4k prompts using the same arrival trace. | Compare short-request p99 and long-request completion. Pick a budget only if both workload classes meet their targets. |
| Prefix reuse | Prefix cache off/on, controlling the fraction of shared token prefixes (0%, 50%, 90%) and warm state. | Report hit rate, prefill time and useful tokens/$ by cohort. This isolates the gain hidden by the starter harness's repeated filler. |
| KV precision | With FP8 weights fixed, compare FP8 KV against a verified higher-precision KV path on the same GPU and backend, if supported. | Measure cache capacity, maximum passing load, retrieval accuracy by context length, and peak HBM. Document an unsupported pairing as a compatibility result. |
| Weight precision | On the same B200 node, rerun native FP8 as a control and compare NVIDIA's expert-only NVFP4 checkpoint at equal TP, KV dtype and context. Record quantized modules and the parent checkpoint. | Test held-out exact-answer tasks, long-context retrieval and reasoning output validity on identical prompts. Set an acceptable paired quality loss before the run. Publish speed/memory and quality deltas together. [NVIDIA checkpoint](https://huggingface.co/nvidia/GLM-5.2-NVFP4) |
| Admission under bursts | Direct engine vs gateway, then change the admission limit in isolation, with one replica throughout. | Select by goodput over all offered requests, admitted p99, 429 fraction, and time to recover. |

BF16 is a candidate for a short quality-reference run once memory fit and budget allow. If it requires another topology, record that comparison separately. AWQ/GPTQ need an architecture-compatible artifact or a reproducible calibration recipe and a verified fused MoE kernel. Merely passing a quantization flag cannot establish a comparison. The NVFP4 study evaluates a published quantization; a later custom quantizer would need calibration data, excluded modules and conversion code of its own.

## GitHub delivery

Keep the initial `main` commit limited to the original license and ignore rules. Open one issue per engineering decision, tie each branch and PR to its issue, and include a reproduction command, expected behavior, actual evidence, and limitations. Squash-merge focused changes after applicable checks pass. Use draft PRs for incomplete experiments. When working solo, record self-review and test evidence. Preserve the record of the actual reviewer.

| Proposed issue / PR | Acceptance evidence |
| --- | --- |
| Establish study design | Review this plan, agree on first workload and budget gates, record provenance. |
| Define streaming correctness and full-SLO accounting | CPU CI passes fixtures for role-only, reasoning-only, grouped deltas, terminal usage, truncated streams and 429s. A synthetic level with 96% goodput and failed p99 must fail the full-SLO ceiling. |
| Bring up the Modal FP8 baseline | Reproducible manifest, cached checkpoint, one valid streamed decode, peak memory, shutdown and billed warm/cold duration. |
| Establish the capacity curve | Release raw measurements plus a one-command figure. Independently reproduce a point near the knee and report uncertainty. |
| Evaluate MTP, scheduling, cache and precision | One study per PR. Include the hypothesis, control, cost, quality check and decision even when an optimization loses. |
| Add gateway overload behavior and publish findings | Demonstrate recovery on a fixed trace. Publish a table mapping each final serving choice to its experiment and operating range. |

Start implementation with the streaming and SLO issue. Its fixtures can catch false wins before the first billable benchmark.
