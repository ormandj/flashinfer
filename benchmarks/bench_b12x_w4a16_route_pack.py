#!/usr/bin/env python3
"""Compare W4A16 route-pack modules in an isolated GPU test process.

Reference and candidate are standalone copies of moe_w4a16_route_pack.py from
the base and proposed FlashInfer revisions. Run with the candidate installed. Integer metadata is checked exactly before
sort; sorted routes are checked against a CPU oracle, including graph replay.
Timings cover the complete pack, with 100 calls captured in each graph to
amortize Python replay overhead. For example, save the base module with
`git show BASE:flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_route_pack.py`
and pass its path as --reference and the proposed module as --candidate.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch
import triton


def load(path, name):
    qualified = f"flashinfer.fused_moe.cute_dsl.blackwell_sm12x.{name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


def workspace(module, ids, block, experts, exact=False):
    capacity = (
        ids.numel()
        if exact
        else module.route_pack_numel_capacity(ids.numel(), topk=ids.shape[-1])
    )
    slots = max(module.max_packed_route_slots(capacity, block, experts), 1)
    sizes = (slots, triton.cdiv(slots, block), 1, experts + 1)
    keys = (
        "packed_route_indices",
        "block_expert_ids",
        "packed_route_count",
        "expert_offsets",
    )
    return {
        k: torch.full((s,), -777, dtype=torch.int32, device="cuda")
        for k, s in zip(keys, sizes, strict=True)
    }


def prefix(module, ids, block, experts, mapping, buffers):
    slots = buffers["packed_route_indices"].numel()
    blocks = buffers["block_expert_ids"].numel()
    capacity = module.route_pack_numel_capacity(ids.numel(), topk=ids.shape[-1])
    if slots < module.max_packed_route_slots(capacity, block, experts):
        capacity = ids.numel()
    kwargs = dict(
        NUMEL_CAPACITY=capacity,
        BLOCK_SIZE=block,
        NUM_EXPERTS=experts,
        MAX_PACKED_ROUTES=slots,
        MAX_ROUTE_BLOCKS=blocks,
        HAS_EXPERT_MAP=mapping is not None,
        BLOCK_E=triton.next_power_of_2(experts),
        BLOCK_T=256,
        BLOCK_ROUTE_INIT=triton.next_power_of_2(slots),
        BLOCK_M=triton.next_power_of_2(blocks),
        num_warps=8,
    )
    if "LOG2_BLOCK_E" in module._pack_topk_routes_small_prefix_kernel.arg_names:
        kwargs["LOG2_BLOCK_E"] = kwargs["BLOCK_E"].bit_length() - 1
    module._pack_topk_routes_small_prefix_kernel[(1,)](
        ids,
        mapping if mapping is not None else ids,
        *buffers.values(),
        ids.numel(),
        **kwargs,
    )


def check(ids, block, experts, mapping, buffers):
    raw = ids.cpu().flatten().long()
    valid = (raw >= 0) & (raw < experts)
    mapped = raw if mapping is None else mapping.cpu().long()[raw.clamp(0, experts - 1)]
    valid &= (mapped >= 0) & (mapped < experts)
    counts = torch.bincount(mapped[valid], minlength=experts)
    padded = ((counts + block - 1) // block) * block
    total = int(padded.sum())
    routes = buffers["packed_route_indices"].cpu().long()
    owners = buffers["block_expert_ids"].cpu().long()
    assert int(buffers["packed_route_count"].item()) == total
    expected_owners = torch.repeat_interleave(torch.arange(experts), padded // block)
    assert torch.equal(owners[: total // block], expected_owners)
    assert (owners[total // block :] == -1).all()
    assert (routes[total:] == raw.numel()).all()
    cursor = 0
    for expert, size in enumerate(padded.tolist()):
        section = routes[cursor : cursor + size]
        assert ((section >= 0) & (section <= raw.numel())).all()
        payload = section[section < raw.numel()].sort().values
        expected = torch.nonzero(valid & (mapped == expert)).flatten()
        assert torch.equal(payload, expected), expert
        cursor += size


def pattern(tokens, experts, dtype, kind, seed):
    generator = torch.Generator().manual_seed(seed)
    if kind == "random":
        x = torch.randint(0, experts, (tokens, 8), generator=generator, dtype=dtype)
    elif kind == "skew":
        x = torch.randint(
            0, min(experts, 3), (tokens, 8), generator=generator, dtype=dtype
        )
    elif kind == "last":
        x = torch.full((tokens, 8), experts - 1, dtype=dtype)
    elif kind == "unique":
        x = torch.arange(tokens * 8, dtype=dtype).reshape(tokens, 8) % experts
    else:
        x = torch.full((tokens, 8), -1, dtype=dtype)
        x.flatten()[::3] = experts
    return x.cuda()


def validate(reference, candidate, quick):
    checked = 0
    cases = [(288, n, 8) for n in (1, 4, 8, 12, 16)]
    if not quick:
        cases += [
            (e, n, b)
            for e in (1, 7, 16, 128, 257, 512)
            for n in (1, 9, 16)
            for b in (8, 48)
        ]
    if not quick:
        cases += [(16, 48, 8), (16, 48, 48)]
    for experts, tokens, block in cases:
        for dtype in (torch.int32, torch.int64):
            for exact in (False, True):
                ids = pattern(tokens, experts, dtype, "random", checked)
                a = workspace(reference, ids, block, experts, exact)
                b = workspace(candidate, ids, block, experts, exact)
                # Bound direct small-prefix checks to the dispatcher's domain.
                small = (
                    triton.next_power_of_2(a["packed_route_indices"].numel()) <= 4096
                    and triton.next_power_of_2(a["block_expert_ids"].numel()) <= 128
                    and triton.next_power_of_2(experts)
                    * triton.next_power_of_2(a["block_expert_ids"].numel())
                    <= 65536
                )
                for has_map in (False, True):
                    mapping = None
                    if has_map:
                        mapping = torch.arange(
                            experts, dtype=torch.int32, device="cuda"
                        ) % max(1, experts // 2)
                        mapping[::3] = -1
                        mapping[1::5] = experts

                    def fn():
                        return candidate.pack_topk_routes_by_expert(
                            ids, block, experts, expert_map=mapping, **b
                        )

                    fn()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        fn()
                    for kind in ("random", "skew", "last", "unique", "invalid"):
                        ids.copy_(pattern(tokens, experts, dtype, kind, checked + 13))
                        if small:
                            prefix(reference, ids, block, experts, mapping, a)
                            prefix(candidate, ids, block, experts, mapping, b)
                            for key in a:
                                assert torch.equal(a[key], b[key]), (
                                    key,
                                    experts,
                                    tokens,
                                    block,
                                    kind,
                                )
                        for tensor in b.values():
                            tensor.fill_(-777)
                        graph.replay()
                        check(ids, block, experts, mapping, b)
                        checked += 1
    return checked


def benchmark(reference, candidate, repeats):
    rows = []
    for tokens in (4, 8, 12, 16, 24, 32):
        for kind in ("random", "skew", "last"):
            ids = pattern(tokens, 288, torch.int32, kind, 20260906 + tokens)
            graphs, owners = {}, []
            for name, module in (("reference", reference), ("candidate", candidate)):
                buffers = workspace(module, ids, 8, 288)
                owners.append(buffers)

                def fn():
                    return module.pack_topk_routes_by_expert(ids, 8, 288, **buffers)

                for _ in range(5):
                    fn()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(100):
                        fn()
                graphs[name] = graph
            samples = {name: [] for name in graphs}
            for rep in range(repeats):
                for name in list(graphs) if rep % 2 == 0 else list(reversed(graphs)):
                    graph = graphs[name]
                    graph.replay()
                    start, end = [
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    ]
                    start.record()
                    for _ in range(10):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(
                        start.elapsed_time(end)
                    )  # ms / 1000 calls == us/call
            row = dict(
                tokens=tokens,
                topk=8,
                experts=288,
                block_size=8,
                pattern=kind,
                samples_us=samples,
                median_us={k: statistics.median(v) for k, v in samples.items()},
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(1)
    reference, candidate = (
        load(args.reference, "route_pack_reference"),
        load(args.candidate, "route_pack_candidate"),
    )
    result = dict(
        torch=torch.__version__,
        triton=triton.__version__,
        gpu=torch.cuda.get_device_name(),
        source_sha256={
            k: hashlib.sha256(p.read_bytes()).hexdigest()
            for k, p in (("reference", args.reference), ("candidate", args.candidate))
        },
        timing="CUDA events, 100 full pack calls per captured graph, 10 replays/sample, alternating order",
    )
    result["integer_and_graph_cases"] = validate(reference, candidate, args.quick)
    print("Validated", result["integer_and_graph_cases"], flush=True)
    if not args.validate_only:
        result["timings"] = benchmark(reference, candidate, args.repeats)
    result["gpu_status"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,temperature.gpu,clocks.sm,power.draw,power.limit",
            "--format=csv",
        ],
        text=True,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
