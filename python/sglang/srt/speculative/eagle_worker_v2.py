import contextlib
import logging
import time
from dataclasses import replace
from typing import List, Optional

import torch

from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess
from sglang.srt.distributed import get_pp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
)
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner import (
    DecodeCudaGraphRunner,
    get_batch_sizes_to_capture,
)
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_model,
    get_parallel,
    get_spec,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import (
    _eagle_prefill_tail_tokens,
    default_tree_mask_mode,
    get_draft_recurrent_hidden_state_spec,
    organize_draft_results,
    per_step_draft_out_cache_loc,
)
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    prepare_for_draft,
    prepare_for_draft_extend,
    run_eagle_verify,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_pp_context,
    draft_tp_context,
    fast_sample,
    get_plan_stream,
    load_token_map,
    renorm_draft_probs,
    sample_draft_proposal,
    select_top_k_tokens,
    spec_stage_span,
)
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)
from sglang.srt.utils.common import (
    MultiprocessingSerializer,
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
    log_info_on_rank0,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_cpu = is_cpu()
_is_npu = is_npu()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_hip = is_hip()
_is_xpu = is_xpu()


logger = logging.getLogger(__name__)


_MTP_SIDECAR_BUFFERS = {}
_MTP_SIDECAR_STREAMS = {}


@contextlib.contextmanager
def _mtp_sidecar_parallel_context(tp_group):
    """Temporarily make the CUDA2 sidecar a coherent TP1/ATTN-TP1 runtime.

    SGLang's normal draft_tp_context only swaps _TP.  Qwen3.5/3.8 attention
    construction and LayerCommunicator also consume the separate attention
    topology, so a TP1 sidecar built with the target's _ATTN_TP becomes a
    permanently half-sharded attention model.  Stack the relevant pointers
    directly instead of nesting patch_tensor_parallel_group; this is safe both
    inside and outside an existing draft context and restores every value.
    """
    import sglang.srt.distributed.parallel_state as _ps
    from sglang.srt.layers import dp_attention as _dp

    _saved_tp = _ps._TP
    _saved_attn_tp = _ps._ATTN_TP
    _saved_attn_cp = _ps._ATTN_CP
    _saved_attn_dp_size = _dp._ATTN_DP_SIZE
    _saved_attn_dp_rank = _dp._ATTN_DP_RANK

    _rank = int(getattr(tp_group, "rank_in_group", 0))
    _size = int(getattr(tp_group, "world_size", 1))
    if _size != 1:
        raise RuntimeError(
            f"MTP sidecar requires a singleton TP group, got world_size={_size}"
        )

    _ps._TP = tp_group
    _ps._ATTN_TP = tp_group
    _ps._ATTN_CP = tp_group
    _dp._ATTN_DP_SIZE = 1
    _dp._ATTN_DP_RANK = 0

    try:
        with (
            get_context().resources.override(
                buffers=_MTP_SIDECAR_BUFFERS,
                streams=_MTP_SIDECAR_STREAMS,
            ),
            get_parallel().override(
                tp_size=1,
                tp_rank=_rank,
                tp_group=tp_group,
                attn_tp_size=1,
                attn_tp_rank=_rank,
                attn_tp_group=tp_group,
                attn_cp_size=1,
                attn_cp_rank=0,
                attn_cp_group=tp_group,
                attn_dp_size=1,
                attn_dp_rank=0,
                dcp_enabled=False,
                dcp_size=1,
                dcp_rank=0,
                attn_dcp_size=1,
                attn_dcp_rank=0,
            ),
        ):
            yield
    finally:
        _dp._ATTN_DP_RANK = _saved_attn_dp_rank
        _dp._ATTN_DP_SIZE = _saved_attn_dp_size
        _ps._ATTN_CP = _saved_attn_cp
        _ps._ATTN_TP = _saved_attn_tp
        _ps._TP = _saved_tp


class EagleDraftWorker(EagleDraftWorkerBase):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
        mtp_authoritative_sidecar: bool = False,
    ):
        super().__init__()

        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self._mtp_authoritative_sidecar = mtp_authoritative_sidecar
        self._mtp_target_gpu_id = target_worker.ps.gpu_id
        # Authoritative CUDA2 state is keyed by target request id.  The first
        # cutover prototype kept one scalar request because max_running_requests=1;
        # parallel serving needs independent Req/KV/Mamba lifetimes per request.
        self._mtp_side_req = None  # legacy compatibility; batch path uses dicts below
        self._mtp_side_rid = None
        self._mtp_side_seq_len = 0
        self._mtp_side_draft_input = None
        self._mtp_side_reqs = {}
        self._mtp_side_seq_lens = {}
        self._mtp_side_target_refs = {}
        # Temporary draft KV belongs to the current synchronous batch, not a req.
        self._mtp_side_last_draft_slots = None
        self._mtp_private_rope_ready = False

        # Args for easy access.  The normal worker historically keeps the generic
        # string ``cuda`` because each TP process already selected its local GPU.
        # A sidecar lives in the *same process* as target TP0, so make its device
        # explicit or later helper allocations silently fall back to CUDA0.
        self.device = (
            f"cuda:{gpu_id}"
            if mtp_authoritative_sidecar and server_args.device == "cuda"
            else server_args.device
        )
        self.topk = server_args.speculative_eagle_topk
        if get_spec().speculative_use_rejection_sampling:
            assert self.topk == 1, "Chain speculative sampling supports only topk=1"
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self._rebuild_topk1_chain_buffers()

        # Load draft model weights only.
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            ctx = draft_tp_context(get_parallel().attn_tp_group)
        else:
            ctx = empty_context()
        with (
            ctx
        ), draft_pp_context(), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                # spec workers don't support pipeline parallelism
                ps=replace(ps, pp_rank=0, pp_size=1),
                nccl_port=nccl_port,
                is_draft_worker=True,
                # The draft runs at absolute target positions.
                context_length=target_worker.model_runner.model_config.context_len,
                random_seed=target_worker.random_seed,
            )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner
        self._init_dsa_index_share_state()
        # Eager draft-extend seed buffer (graph paths use their own static ones).
        self.dsa_extend_topk_buf: Optional[torch.Tensor] = None
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        self.tree_mask_mode = default_tree_mask_mode()

        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        """Allocate draft KV cache pools (called by scheduler)."""
        if self._mtp_authoritative_sidecar:
            from sglang.srt.distributed.parallel_state import get_self_pp_group

            if memory_pool_config is None:
                raise RuntimeError(
                    "MTP cutover requires the target's resolved MemoryPoolConfig"
                )

            target_gpu_id = self._mtp_target_gpu_id
            try:
                torch.cuda.set_device(self.gpu_id)
                with (
                    _mtp_sidecar_parallel_context(get_self_pp_group()),
                    draft_pp_context(),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                ):
                    # Qwen TP1 sidecar is self-contained (full embedding + native
                    # quantized lm_head), so NEVER share target CUDA0 parameters.
                    self.init_token_map()
                    self.draft_worker.alloc_memory_pool(
                        memory_pool_config=memory_pool_config
                    )

                mr = self.draft_runner
                self.req_to_token_pool = mr.req_to_token_pool
                self.token_to_kv_pool_allocator = mr.token_to_kv_pool_allocator

                # A colocated EAGLE draft shares the target HybridReqToTokenPool,
                # whose extra-buffer ping-pong mapping is created by the target
                # pool.  This CUDA2 draft owns an independent draft-worker pool;
                # exact-image builds can therefore omit that mapping even though
                # the process-wide Mamba strategy is extra_buffer_lazy.  Promote
                # the sidecar pool to the same lazy tracking contract before its
                # first Req allocation.  HybridReqToTokenPool.alloc() then owns
                # the actual ping-pong slot allocation and mapping updates.
                if (
                    hasattr(self.req_to_token_pool, "mamba_pool")
                    and not hasattr(
                        self.req_to_token_pool,
                        "req_index_to_mamba_ping_pong_track_buffer_mapping",
                    )
                ):
                    _side_pool = self.req_to_token_pool
                    _side_pool.enable_mamba_extra_buffer = True
                    _side_pool.enable_mamba_extra_buffer_lazy = True
                    if not hasattr(_side_pool, "mamba_ping_pong_track_buffer_size"):
                        _side_pool.mamba_ping_pong_track_buffer_size = 2
                    _side_pool.req_index_to_mamba_ping_pong_track_buffer_mapping = (
                        torch.zeros(
                            (
                                _side_pool.req_to_token.shape[0],
                                _side_pool.mamba_ping_pong_track_buffer_size,
                            ),
                            dtype=torch.int64,
                            device=_side_pool.device,
                        )
                    )
                    logger.info(
                        "[MTP-CUTOVER-MAMBA] CUDA%d installed lazy ping-pong "
                        "tracking rows=%d width=%d mamba_slots=%d",
                        self.gpu_id,
                        _side_pool.req_to_token.shape[0],
                        _side_pool.mamba_ping_pong_track_buffer_size,
                        _side_pool.mamba_pool.size,
                    )
                if int(mr.max_total_num_tokens) < 65536:
                    raise RuntimeError(
                        f"CUDA2 MTP pool too small for cutover: "
                        f"{mr.max_total_num_tokens} < 65536"
                    )
                free_b, total_b = torch.cuda.mem_get_info(self.gpu_id)
                logger.info(
                    "[MTP-CUTOVER-POOL] CUDA%d side_tokens=%d free=%.2f/%.2f GiB",
                    self.gpu_id,
                    mr.max_total_num_tokens,
                    free_b / (1 << 30),
                    total_b / (1 << 30),
                )
            finally:
                torch.cuda.set_device(target_gpu_id)
            return

        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

        # Share target embedding/lm_head after the target pool has been sized,
        # but before the draft pool is allocated. This releases the temporary
        # draft copies and leaves headroom for CUDA graph capture.
        self.init_token_map()
        self.init_lm_head()

        self.draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

        # Probe: allocate an independent draft KV/request pool on the CUDA2
        # TP1 sidecar.  Reuse the resolved target memory-pool configuration so
        # the sidecar capacity tracks the target instead of consuming all free
        # memory on GPU2.
        if getattr(self, "_mtp_sidecar_probe", None) is not None:
            from sglang.srt.distributed.parallel_state import get_self_pp_group

            sidecar_gpu_id = self._mtp_sidecar_probe.gpu_id
            target_gpu_id = self.ps.gpu_id

            try:
                torch.cuda.set_device(sidecar_gpu_id)

                with (
                    _mtp_sidecar_parallel_context(get_self_pp_group()),
                    draft_pp_context(),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                ):
                    self._mtp_sidecar_probe.alloc_memory_pool(
                        memory_pool_config=memory_pool_config,
                    )

                mr = self._mtp_sidecar_probe.model_runner
                free_b, total_b = torch.cuda.mem_get_info(sidecar_gpu_id)

                logger.info(
                    "[MTP-SIDECAR-POOL] CUDA%d local pool ready; "
                    "tokens=%d req_pool=%s kv_allocator=%s "
                    "free=%.2f GiB / total=%.2f GiB",
                    sidecar_gpu_id,
                    mr.max_total_num_tokens,
                    mr.req_to_token_pool.req_to_token.device,
                    type(mr.token_to_kv_pool_allocator).__name__,
                    free_b / (1 << 30),
                    total_b / (1 << 30),
                )
            finally:
                torch.cuda.set_device(target_gpu_id)

        if get_spec().speculative_use_rejection_sampling:
            target_vocab_size = self.target_worker.model_config.vocab_size
            draft_vocab_size = (
                self.hot_token_id.shape[0]
                if self.hot_token_id is not None
                else target_vocab_size
            )
            # FIXME: support reduced (hot) draft vocab by scattering draft probs
            # into the target vocab via the d2t map before the sampling kernel.
            if draft_vocab_size != target_vocab_size:
                raise ValueError(
                    "--speculative-use-rejection-sampling requires the draft and "
                    f"target to share one vocab, but the draft vocab "
                    f"({draft_vocab_size}) != target vocab ({target_vocab_size})."
                )

    def init_attention_backends(self):
        if self._mtp_authoritative_sidecar:
            from sglang.srt.distributed.parallel_state import get_self_pp_group

            target_gpu_id = self._mtp_target_gpu_id
            try:
                torch.cuda.set_device(self.gpu_id)
                with (
                    _mtp_sidecar_parallel_context(get_self_pp_group()),
                    draft_pp_context(),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                ):
                    self.draft_worker.init_attention_backends()
                    self.init_attention_backend()
                logger.info(
                    "[MTP-CUTOVER-ATTN] CUDA%d decode=%s extend=%s",
                    self.gpu_id,
                    type(self.draft_attn_backend).__name__,
                    type(self.draft_extend_attn_backend).__name__,
                )
            finally:
                torch.cuda.set_device(target_gpu_id)
            return

        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker.init_attention_backends()
            self.init_attention_backend()

        # Initialize the independent TP1 MTP sidecar attention backend on
        # CUDA2.  Keep it inside the singleton TP/PP contexts used when the
        # sidecar worker was constructed.
        if getattr(self, "_mtp_sidecar_probe", None) is not None:
            from sglang.srt.distributed.parallel_state import get_self_pp_group

            sidecar_gpu_id = self._mtp_sidecar_probe.gpu_id
            target_gpu_id = self.ps.gpu_id

            try:
                torch.cuda.set_device(sidecar_gpu_id)

                with (
                    _mtp_sidecar_parallel_context(get_self_pp_group()),
                    draft_pp_context(),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                ):
                    self._mtp_sidecar_probe.init_attention_backends()

                mr = self._mtp_sidecar_probe.model_runner

                _side_ws = _MTP_SIDECAR_BUFFERS.get("flashinfer_workspace")
                _target_ws = get_context().resources.buffers.get("flashinfer_workspace")
                if isinstance(_side_ws, torch.Tensor):
                    if _side_ws.device != torch.device("cuda", sidecar_gpu_id):
                        raise RuntimeError(
                            "CUDA2 sidecar FlashInfer workspace landed on the wrong device: "
                            f"{_side_ws.device}"
                        )
                    if isinstance(_target_ws, torch.Tensor) and (
                        _side_ws.data_ptr() == _target_ws.data_ptr()
                    ):
                        raise RuntimeError(
                            "CUDA2 sidecar still aliases the target FlashInfer workspace"
                        )

                logger.info(
                    "[MTP-SIDECAR-RESOURCE] side_flashinfer=%s target_flashinfer=%s separate=%s",
                    (str(_side_ws.device) if isinstance(_side_ws, torch.Tensor) else None),
                    (str(_target_ws.device) if isinstance(_target_ws, torch.Tensor) else None),
                    (
                        not isinstance(_side_ws, torch.Tensor)
                        or not isinstance(_target_ws, torch.Tensor)
                        or _side_ws.data_ptr() != _target_ws.data_ptr()
                    ),
                )
                logger.info(
                    "[MTP-SIDECAR-ATTN] CUDA%d attention backend ready: %s",
                    sidecar_gpu_id,
                    type(mr.attn_backend).__name__,
                )
            finally:
                torch.cuda.set_device(target_gpu_id)

    def init_cuda_graphs(self):
        if self._mtp_authoritative_sidecar:
            from sglang.srt.distributed.parallel_state import get_self_pp_group

            target_gpu_id = self._mtp_target_gpu_id
            try:
                torch.cuda.set_device(self.gpu_id)
                self._mtp_authoritative_ensure_rope()
                with (
                    _mtp_sidecar_parallel_context(get_self_pp_group()),
                    draft_pp_context(),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                ):
                    # This creates EagerRunner and all runner attributes without
                    # capturing a normal decode graph for the draft TpModelWorker.
                    self.draft_worker.init_cuda_graphs(
                        capture_decode_cuda_graph=False
                    )
                self.draft_runner.prefill_cuda_graph_runner = None
                self.draft_runner.decode_cuda_graph_runner = None
                self.cuda_graph_runner = None
                self.cuda_graph_runner_for_draft_extend = None
                logger.info(
                    "[MTP-CUTOVER-GRAPH] CUDA%d sidecar=eager-only", self.gpu_id
                )
            finally:
                torch.cuda.set_device(target_gpu_id)
            return

        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)
            if check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE):
                self.draft_runner.init_prefill_cuda_graph(force_for_draft_worker=True)
            self._capture_cuda_graphs()

        if (c := self.draft_runner.canary_manager) is not None:
            c.mark_init_finished()

    def _init_dsa_index_share_state(self) -> None:
        # Populate DSA index-share fields from the draft runner's hf_config.
        # Reused by the attention unit-test harnesses, which skip __init__.
        hf_config = self.draft_runner.model_config.hf_config
        # Reuse the first draft step's DSA indexer topk across the rest;
        # topk == 1 only (select_top_k_tokens reorders rows, desyncing indices).
        self.index_share_for_mtp_iteration = (
            getattr(hf_config, "index_share_for_mtp_iteration", False)
            and self.topk == 1
        )
        # GLM-5.2 MTP IndexShare: seed reused indexer top-k from draft-extend
        # (last verified token), not draft-decode step 0.
        self.dsa_index_topk = getattr(hf_config, "index_topk", None)
        self.seed_dsa_topk_from_draft_extend = (
            self.index_share_for_mtp_iteration and self.dsa_index_topk is not None
        )

    def init_token_map(self):
        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if get_spec().speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif get_spec().speculative_token_map is not None:
            self.hot_token_id = load_token_map(get_spec().speculative_token_map)
        else:
            self.hot_token_id = None

    def init_lm_head(self):
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        target_lm_head = getattr(self.target_worker.model_runner.model, "lm_head", None)

        def maybe_share_target_lm_head():
            if (
                target_lm_head is not None
                and self.hot_token_id is None
                and getattr(self.draft_runner.model, "hot_token_id", None) is None
                and hasattr(self.draft_runner.model, "set_lm_head_from_target")
            ):
                self.draft_runner.model.set_lm_head_from_target(target_lm_head)

        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
                maybe_share_target_lm_head()
            else:
                self.draft_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None and head is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_runner.model.set_embed_and_head(embed, head)
            maybe_share_target_lm_head()

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners

        self.draft_extend_attn_backend = None

        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
            seed_dsa_topk_from_draft_extend=self.seed_dsa_topk_from_draft_extend,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        if self.draft_extend_attn_backend is not None:
            self.draft_runner.attn_backend = self.draft_extend_attn_backend
        self.tree_mask_mode = default_tree_mask_mode()

    def _capture_cuda_graphs(self):
        """Capture the draft worker's own cuda graphs (decode + draft-extend)."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if _is_cpu or check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            return

        if get_model().model_impl == "mindspore":
            return

        Device2DraftCudaGraphRunner = {
            "xpu": EAGLEDraftCudaGraphRunner,
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        decode_backend = get_exec().graph.cuda_graph_config.decode.backend
        capture_bs, _ = get_batch_sizes_to_capture(self.draft_runner)
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft decode CUDA graph begin. backend={decode_backend}, "
                f"num_tokens_per_req={self.topk}, bs={capture_bs}, "
                f"avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            capture_time = time.perf_counter() - tic
            self._specialized_graph_memory_usage["draft_decode"] = (
                self._specialized_graph_memory_usage.get("draft_decode", 0.0)
                + before_mem
                - after_mem
            )
            self._specialized_graph_time_usage["draft_decode"] = (
                self._specialized_graph_time_usage.get("draft_decode", 0.0)
                + capture_time
            )
            log_info_on_rank0(
                logger,
                "Capture draft decode CUDA graph end. "
                f"elapsed={capture_time:.2f} s, "
                f"mem usage={(before_mem - after_mem):.2f} GB, "
                f"avail mem={after_mem:.2f} GB.",
            )

        Device2ExtendCudaGraphRunner = {
            "xpu": EAGLEDraftExtendCudaGraphRunner,
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        supports_hip_aiter_draft_extend_graph = False
        if _is_hip:
            # Keep import local so non-HIP environments do not require aiter.
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )

            supports_hip_aiter_draft_extend_graph = isinstance(
                self.draft_attn_backend, AiterMultiStepDraftBackend
            )

        graph_supported_backend_types = [
            TritonAttnBackend,
            TRTLLMMLABackend,
            TRTLLMHAAttnBackend,
            TokenspeedMLABackend,
            FlashInferAttnBackend,
        ]
        if _is_cuda or _is_musa:
            # DSA is CUDA-only; import lazily so non-CUDA builds don't pull in
            # deep_gemm and the rest of the sparse-attention stack at import time.
            from sglang.srt.layers.attention.dsa_backend import (
                DeepseekSparseAttnBackend,
            )

            graph_supported_backend_types.append(DeepseekSparseAttnBackend)
            from sglang.srt.layers.attention.deepseek_v4_backend import (
                DeepseekV4AttnBackend,
            )

            graph_supported_backend_types.append(DeepseekV4AttnBackend)
        if _is_cuda:
            # FlashMLA is CUDA-only; import lazily so CPU builds don't pull
            # sgl_kernel.flash_mla at import time.
            from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

            graph_supported_backend_types.append(FlashMLABackend)

        graph_supported_backend = isinstance(
            self.draft_extend_attn_backend,
            tuple(graph_supported_backend_types),
        )
        supports_cuda_draft_extend_graph = (
            _is_cuda or _is_musa
        ) and graph_supported_backend
        # Capture extend
        # TODO: support draft extend cuda graph for more attention backends
        if (
            self.draft_extend_attn_backend
            and not envs.SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH.get()
            and (
                _is_npu
                or _is_xpu
                or supports_cuda_draft_extend_graph
                or supports_hip_aiter_draft_extend_graph
            )
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend CUDA graph begin. backend={decode_backend}, "
                f"num_tokens_per_req={self.speculative_num_draft_tokens}, "
                f"bs={capture_bs}, avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            # draft_extend is the step's last shared-buffer-reading phase; its
            # read-done event is what the scheduler's WAR barrier waits on.
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            capture_time = time.perf_counter() - tic
            self._specialized_graph_memory_usage["draft_extend"] = (
                self._specialized_graph_memory_usage.get("draft_extend", 0.0)
                + before_mem
                - after_mem
            )
            self._specialized_graph_time_usage["draft_extend"] = (
                self._specialized_graph_time_usage.get("draft_extend", 0.0)
                + capture_time
            )
            log_info_on_rank0(
                logger,
                "Capture draft extend CUDA graph end. "
                f"elapsed={capture_time:.2f} s, "
                f"mem usage={(before_mem - after_mem):.2f} GB, "
                f"avail mem={after_mem:.2f} GB.",
            )

    def _mtp_authoritative_to_side(self, value):
        if value is None or not isinstance(value, torch.Tensor):
            return value
        if value.device.type == "cuda":
            value = value.detach().to("cpu")
        return value.to(self.device)

    def _mtp_authoritative_ensure_rope(self):
        if not self._mtp_authoritative_sidecar or self._mtp_private_rope_ready:
            return
        import copy as _copy

        side_device = torch.device(self.device)
        clones = {}
        users = 0
        for _name, mod in self.draft_runner.model.named_modules():
            rope = getattr(mod, "rotary_emb", None)
            if rope is None:
                continue
            private = clones.get(id(rope))
            if private is None:
                private = _copy.deepcopy(rope).to(side_device)
                cache = getattr(private, "cos_sin_cache", None)
                if isinstance(cache, torch.Tensor):
                    private.cos_sin_cache = cache.to(side_device)
                clones[id(rope)] = private
            mod.rotary_emb = private
            users += 1
        self._mtp_private_rope_ready = True
        logger.info(
            "[MTP-CUTOVER-ROPE] CUDA%d users=%d private=%d",
            self.gpu_id,
            users,
            len(clones),
        )

    def _mtp_authoritative_to_side_draft_input(self, draft_input):
        import copy as _copy

        if draft_input is None:
            return None
        side = _copy.copy(draft_input)
        for name in (
            "topk_p",
            "topk_index",
            "draft_probs",
            "hidden_states",
            "dsa_topk_indices",
            "bonus_tokens",
            "kv_indptr",
            "kv_indices",
            "future_indices",
        ):
            value = getattr(side, name, None)
            if isinstance(value, torch.Tensor):
                setattr(side, name, self._mtp_authoritative_to_side(value))
        return side

    def _mtp_authoritative_release_finished(self, keep_rids=()):
        """Reclaim independent CUDA2 Req/KV/Mamba state after target Req finishes."""
        keep = set(keep_rids)
        for rid, req in list(self._mtp_side_reqs.items()):
            if rid in keep:
                continue
            target_ref = self._mtp_side_target_refs.get(rid)
            if target_ref is None or not target_ref.finished():
                continue

            seq_len = int(self._mtp_side_seq_lens.get(rid, 0))
            row = req.req_pool_idx
            if row is not None and seq_len > 0:
                locs = self.req_to_token_pool.req_to_token[row, :seq_len].clone()
                locs = locs[locs > 0]
                if locs.numel() > 0:
                    self.token_to_kv_pool_allocator.free(locs)
            if (
                hasattr(self.req_to_token_pool, "free_mamba_cache")
                and req.mamba_pool_idx is not None
            ):
                self.req_to_token_pool.free_mamba_cache(req)
            if req.req_pool_idx is not None:
                self.req_to_token_pool.free(req)

            self._mtp_side_reqs.pop(rid, None)
            self._mtp_side_seq_lens.pop(rid, None)
            self._mtp_side_target_refs.pop(rid, None)
            logger.info("[MTP-CUTOVER-REQ] CUDA%d rid=%s released", self.gpu_id, rid)

    def _mtp_authoritative_reset_req(self, target_req):
        import copy as _copy

        # Do NOT clear the whole sidecar pool here: other requests may be decoding.
        self._mtp_authoritative_release_finished(keep_rids=(target_req.rid,))

        req = _copy.copy(target_req)
        req.req_pool_idx = None
        req.mamba_pool_idx = None
        req.mamba_ping_pong_track_buffer = None
        req.mamba_next_track_idx = None
        req.mamba_last_track_idx = None
        req.mamba_last_track_seqlen = None
        req.mamba_branching_seqlen = None
        req.mamba_cow_src_index = None
        req.mamba_needs_clear = False
        req.mamba_lazy_is_insert = True
        req.kv_committed_len = 0
        req.kv = None
        req.extend_batch_idx = 0
        req.decode_batch_idx = 0
        req.prefix_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        req.last_node = None
        req.last_host_node = None
        req.best_match_node = None
        req.cache_protected_len = 0
        req.num_matched_prefix_tokens = 0
        req.host_hit_length = 0
        req.swa_host_hit_length = 0
        req.mamba_host_hit_length = 0
        req.storage_hit_length = 0
        req.skip_lock_node_ids = {}

        rows = self.req_to_token_pool.alloc([req])
        if rows is None:
            # A completed target Req may only have become visible since the first GC.
            self._mtp_authoritative_release_finished(keep_rids=(target_req.rid,))
            rows = self.req_to_token_pool.alloc([req])
        if rows is None:
            raise RuntimeError("CUDA2 MTP request-pool allocation failed")

        self._mtp_side_reqs[target_req.rid] = req
        self._mtp_side_seq_lens[target_req.rid] = 0
        self._mtp_side_target_refs[target_req.rid] = target_req
        logger.info(
            "[MTP-CUTOVER-REQ] CUDA%d rid=%s row=%d allocated active=%d",
            self.gpu_id,
            target_req.rid,
            int(rows[0]),
            len(self._mtp_side_reqs),
        )
        return req

    def _mtp_authoritative_get_req(self, target_req):
        req = self._mtp_side_reqs.get(target_req.rid)
        if req is None:
            req = self._mtp_authoritative_reset_req(target_req)
        self._mtp_side_target_refs[target_req.rid] = target_req
        # Mirror scheduler counters used by chunked-prefill assertions.
        req.inflight_middle_chunks = target_req.inflight_middle_chunks
        req.extend_batch_idx = target_req.extend_batch_idx
        req.decode_batch_idx = target_req.decode_batch_idx
        return req

    def _mtp_authoritative_make_batch(self, target_batch, seq_lens):
        import copy as _copy
        from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs

        if len(seq_lens) != len(target_batch.reqs):
            raise RuntimeError(
                f"CUDA2 MTP seq-len batch mismatch: {len(seq_lens)} != {len(target_batch.reqs)}"
            )
        keep_rids = [r.rid for r in target_batch.reqs]
        self._mtp_authoritative_release_finished(keep_rids=keep_rids)
        side_reqs = [self._mtp_authoritative_get_req(r) for r in target_batch.reqs]

        side_batch = _copy.copy(target_batch)
        side_batch.reqs = side_reqs
        side_batch.req_to_token_pool = self.req_to_token_pool
        side_batch.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
        side_batch.device = self.device
        side_batch.req_pool_indices_cpu = torch.tensor(
            [r.req_pool_idx for r in side_reqs], dtype=torch.int64
        )
        side_batch.req_pool_indices = side_batch.req_pool_indices_cpu.to(self.device)
        side_batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        side_batch.seq_lens = side_batch.seq_lens_cpu.to(self.device)
        side_batch.seq_lens_sum = int(sum(seq_lens))
        side_batch.orig_seq_lens = self._mtp_authoritative_to_side(
            getattr(target_batch, "orig_seq_lens", None)
        )
        side_batch.mamba_track_mask = self._mtp_authoritative_to_side(
            getattr(target_batch, "mamba_track_mask", None)
        )
        side_batch.mamba_track_seqlens = self._mtp_authoritative_to_side(
            getattr(target_batch, "mamba_track_seqlens", None)
        )
        side_batch.mamba_lazy_spec_track_positions_cpu = None
        if hasattr(self.req_to_token_pool, "mamba_pool"):
            set_mamba_track_indices_from_reqs(side_batch)
            side_batch._collect_deferred_mamba_cow_and_clear(side_reqs)
        return side_batch

    def mtp_authoritative_prefill(
        self,
        target_batch,
        target_hidden_states,
        next_token_ids,
        mm_input_embeds=None,
    ):
        from sglang.srt.distributed.parallel_state import get_self_pp_group

        bs = len(target_batch.reqs)
        if bs == 0:
            raise RuntimeError("MTP cutover prefill received an empty batch")
        target_gpu_id = self._mtp_target_gpu_id
        try:
            torch.cuda.set_device(self.gpu_id)
            self._mtp_authoritative_ensure_rope()
            side_reqs = [self._mtp_authoritative_get_req(r) for r in target_batch.reqs]

            if target_batch.seq_lens_cpu is not None:
                seq_lens = [int(x) for x in target_batch.seq_lens_cpu.tolist()]
            else:
                seq_lens = [int(x) for x in target_batch.seq_lens.detach().cpu().tolist()]
            extend_lens = [int(x) for x in target_batch.extend_lens]
            if len(extend_lens) != bs:
                raise RuntimeError(f"CUDA2 MTP extend_lens mismatch: {len(extend_lens)} != {bs}")
            old_lens = [int(self._mtp_side_seq_lens.get(r.rid, 0)) for r in target_batch.reqs]
            for req, old_len, seq_len, extend_len in zip(
                target_batch.reqs, old_lens, seq_lens, extend_lens
            ):
                if seq_len - extend_len != old_len:
                    raise RuntimeError(
                        "CUDA2 MTP chunk state diverged: "
                        f"rid={req.rid} old={old_len} seq={seq_len} extend={extend_len}"
                    )

            total_extend = int(sum(extend_lens))
            slots = self.token_to_kv_pool_allocator.alloc(total_extend)
            if slots is None:
                raise RuntimeError(
                    f"CUDA2 MTP prefill KV allocation failed: need={total_extend}"
                )
            off = 0
            for side_req, old_len, seq_len, extend_len in zip(
                side_reqs, old_lens, seq_lens, extend_lens
            ):
                if extend_len:
                    self.req_to_token_pool.write(
                        (side_req.req_pool_idx, slice(old_len, seq_len)),
                        slots[off : off + extend_len],
                    )
                off += extend_len

            side_batch = self._mtp_authoritative_make_batch(target_batch, seq_lens)
            side_batch.prefix_lens = old_lens
            side_batch.extend_lens = extend_lens
            side_batch.extend_num_tokens = total_extend
            side_batch.input_ids = self._mtp_authoritative_to_side(target_batch.input_ids)
            side_batch.out_cache_loc = slots

            with (
                _mtp_sidecar_parallel_context(get_self_pp_group()),
                draft_pp_context(),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                next_draft = self._draft_extend_for_prefill(
                    side_batch,
                    self._mtp_authoritative_to_side(target_hidden_states),
                    self._mtp_authoritative_to_side(next_token_ids),
                    self._mtp_authoritative_to_side(mm_input_embeds),
                )
                torch.cuda.synchronize(self.gpu_id)

            for side_req, target_req, seq_len in zip(
                side_reqs, target_batch.reqs, seq_lens
            ):
                side_req.kv_committed_len = seq_len
                self._mtp_side_seq_lens[target_req.rid] = seq_len
            logger.info(
                "[MTP-CUTOVER-PREFILL] CUDA%d bs=%d seq=%s extend=%s active=%d",
                self.gpu_id,
                bs,
                seq_lens,
                extend_lens,
                len(self._mtp_side_reqs),
            )
            return next_draft
        finally:
            torch.cuda.set_device(target_gpu_id)

    def mtp_authoritative_draft_tokens(self, target_batch):
        from sglang.srt.distributed.parallel_state import get_self_pp_group
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        bs = len(target_batch.reqs)
        if bs == 0:
            raise RuntimeError("MTP cutover draft received an empty batch")
        if target_batch.spec_info is None:
            raise RuntimeError("CUDA2 MTP draft state is missing after prefill/extend")
        target_gpu_id = self._mtp_target_gpu_id
        try:
            torch.cuda.set_device(self.gpu_id)
            side_reqs = [self._mtp_authoritative_get_req(r) for r in target_batch.reqs]
            seq_lens = [int(self._mtp_side_seq_lens[r.rid]) for r in target_batch.reqs]
            side_input = self._mtp_authoritative_to_side_draft_input(target_batch.spec_info)

            future = self.token_to_kv_pool_allocator.alloc(bs * self.speculative_num_steps)
            if future is None:
                raise RuntimeError(
                    "CUDA2 MTP speculative KV reservation failed: "
                    f"need={bs * self.speculative_num_steps}"
                )
            off = 0
            for side_req, seq_len in zip(side_reqs, seq_lens):
                req_future = future[off : off + self.speculative_num_steps]
                self.req_to_token_pool.write(
                    (
                        side_req.req_pool_idx,
                        slice(seq_len, seq_len + self.speculative_num_steps),
                    ),
                    req_future,
                )
                off += self.speculative_num_steps
            self._mtp_side_last_draft_slots = future

            side_batch = self._mtp_authoritative_make_batch(target_batch, seq_lens)
            side_batch.forward_mode = ForwardMode.DECODE
            side_batch.spec_info = side_input
            side_batch.input_ids = side_input.topk_index.reshape(-1)

            with (
                _mtp_sidecar_parallel_context(get_self_pp_group()),
                draft_pp_context(),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                forward_batch, _ = prepare_for_draft(
                    side_input,
                    self.req_to_token_pool,
                    side_batch,
                    None,
                    self.draft_runner,
                    self.topk,
                    self.speculative_num_steps,
                )
                if self.speculative_num_steps > 1:
                    self.draft_attn_backend.init_forward_metadata(forward_batch)
                    forward_batch.mark_forward_metadata_ready()
                _, _, draft_tokens, draft_probs = self.draft_forward(forward_batch)
                if draft_probs is not None:
                    raise RuntimeError(
                        "MTP cutover does not support rejection-sampling draft_probs"
                    )
                torch.cuda.synchronize(self.gpu_id)

            logger.info(
                "[MTP-CUTOVER-DRAFT] CUDA%d bs=%d seq=%s proposal=%s",
                self.gpu_id,
                bs,
                seq_lens,
                draft_tokens.detach().cpu().tolist(),
            )
            return draft_tokens
        finally:
            torch.cuda.set_device(target_gpu_id)

    def mtp_authoritative_extend_after_verify(self, target_batch, target_result):
        import copy as _copy
        from sglang.srt.distributed.parallel_state import get_self_pp_group

        bs = len(target_batch.reqs)
        if bs == 0:
            raise RuntimeError("MTP cutover extend received an empty batch")
        target_gpu_id = self._mtp_target_gpu_id
        accept_cpu = [int(x) for x in target_result.accept_lens.detach().cpu().tolist()]
        old_lens = [int(self._mtp_side_seq_lens[r.rid]) for r in target_batch.reqs]
        try:
            torch.cuda.set_device(self.gpu_id)
            side_reqs = [self._mtp_authoritative_get_req(r) for r in target_batch.reqs]

            # Draft-decode's temporary KV is no longer needed after target verify.
            if self._mtp_side_last_draft_slots is not None:
                self.token_to_kv_pool_allocator.free(self._mtp_side_last_draft_slots)
                self._mtp_side_last_draft_slots = None

            width = self.speculative_num_draft_tokens
            extend_slots = self.token_to_kv_pool_allocator.alloc(bs * width)
            if extend_slots is None:
                raise RuntimeError(
                    f"CUDA2 MTP draft-extend KV allocation failed: need={bs * width}"
                )
            slot_rows = extend_slots.view(bs, width)
            for side_req, old_len, row_slots in zip(side_reqs, old_lens, slot_rows):
                self.req_to_token_pool.write(
                    (side_req.req_pool_idx, slice(old_len, old_len + width)),
                    row_slots,
                )

            side_batch = self._mtp_authoritative_make_batch(target_batch, old_lens)
            side_batch.out_cache_loc = extend_slots

            side_result = _copy.copy(target_result)
            side_logits = _copy.copy(target_result.logits_output)
            side_logits.hidden_states = self._mtp_authoritative_to_side(
                target_result.logits_output.hidden_states
            )
            side_result.logits_output = side_logits
            side_result.next_token_ids = self._mtp_authoritative_to_side(
                target_result.next_token_ids
            )
            side_result.accept_lens = self._mtp_authoritative_to_side(
                target_result.accept_lens
            )
            side_result.next_draft_input = EagleDraftInput(
                bonus_tokens=self._mtp_authoritative_to_side(
                    target_result.next_draft_input.bonus_tokens
                )
            )

            with (
                _mtp_sidecar_parallel_context(get_self_pp_group()),
                draft_pp_context(),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self._draft_extend_for_decode(side_batch, side_result)
                torch.cuda.synchronize(self.gpu_id)

            free_parts = []
            new_lens = []
            for target_req, side_req, old_len, accepted, row_slots in zip(
                target_batch.reqs, side_reqs, old_lens, accept_cpu, slot_rows
            ):
                if accepted < width:
                    free_parts.append(row_slots[accepted:])
                new_len = old_len + accepted
                new_lens.append(new_len)
                side_req.kv_committed_len = new_len
                self._mtp_side_seq_lens[target_req.rid] = new_len
            if free_parts:
                tail = torch.cat([x for x in free_parts if x.numel() > 0])
                if tail.numel() > 0:
                    self.token_to_kv_pool_allocator.free(tail)

            logger.info(
                "[MTP-CUTOVER-EXTEND] CUDA%d bs=%d old=%s accept=%s new=%s",
                self.gpu_id,
                bs,
                old_lens,
                accept_cpu,
                new_lens,
            )
            return side_result.next_draft_input
        finally:
            torch.cuda.set_device(target_gpu_id)

    def draft(self, batch: ScheduleBatch):
        draft_input: EagleDraftInput = batch.spec_info
        forward_batch, can_run_decode_cuda_graph = prepare_for_draft(
            draft_input,
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )
        if (
            can_run_decode_cuda_graph
            and not forward_batch.forward_mode.is_idle()
            and self.seed_dsa_topk_from_draft_extend
            and draft_input.dsa_topk_indices is None
        ):
            can_run_decode_cuda_graph = False

        n_inner = self.speculative_num_steps - 1
        canary_outside_ctx = (
            c.with_ops_outside_graph(
                single_forward_indices=list(range(n_inner)),
                maybe_inaccurate_forward_batch=forward_batch,
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )

        with canary_outside_ctx:
            # Run draft
            if can_run_decode_cuda_graph:
                parent_list, top_scores_index, draft_tokens, draft_probs = (
                    self.cuda_graph_runner.execute(forward_batch)
                )
            else:
                if (
                    not forward_batch.forward_mode.is_idle()
                    and self.speculative_num_steps > 1
                ):
                    # Skip attention backend init for 1-step draft,
                    # `draft_forward` only does sample in this case.
                    self.draft_attn_backend.init_forward_metadata(forward_batch)
                    forward_batch.mark_forward_metadata_ready()
                parent_list, top_scores_index, draft_tokens, draft_probs = (
                    self.draft_forward(forward_batch)
                )

        _side_proposal = getattr(
            self, "_mtp_sidecar_multistep_proposal", None
        )
        if _side_proposal is not None:
            try:
                _stock_proposal = draft_tokens.detach().to("cpu")
                _same_shape = tuple(_side_proposal.shape) == tuple(_stock_proposal.shape)
                _exact = bool(
                    _same_shape and torch.equal(_side_proposal, _stock_proposal)
                )
                _matches = (
                    int((_side_proposal == _stock_proposal).sum().item())
                    if _same_shape
                    else -1
                )
                logger.info(
                    "[MTP-SIDECAR-PROPOSAL] side=%s stock=%s shape_match=%s "
                    "token_matches=%d/%d exact=%s",
                    _side_proposal.tolist(),
                    _stock_proposal.tolist(),
                    _same_shape,
                    _matches,
                    _stock_proposal.numel(),
                    _exact,
                )
            finally:
                self._mtp_sidecar_multistep_proposal = None

        return build_eagle_verify_input(
            batch,
            draft_input,
            parent_list,
            top_scores_index,
            draft_tokens,
            draft_probs,
            target_worker=self.target_worker,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            tree_mask_mode=self.tree_mask_mode,
            device=self.device,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = per_step_draft_out_cache_loc(
            out_cache_loc,
            forward_batch.batch_size,
            self.topk,
            self.speculative_num_steps,
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        if get_spec().speculative_use_rejection_sampling:
            draft_probs_list: List[torch.Tensor] = [spec_info.draft_probs]

        topk1_chain_fits = (
            self.topk == 1
            and topk_index.shape[0] <= self._topk1_parents_prealloc.shape[0]
        )
        # Materialize the chain directly only when the CUDA kernel can write
        # every subsequent column. Other topk=1 paths retain the token list and
        # assemble it with one final cat instead of launching a copy per step.
        draft_tokens_topk1 = None
        if (
            topk1_chain_fits
            and _is_cuda
            and self.hot_token_id is None
            and not get_spec().speculative_use_rejection_sampling
        ):
            draft_tokens_topk1 = torch.empty(
                (topk_index.shape[0], self.speculative_num_steps),
                dtype=topk_index.dtype,
                device=topk_index.device,
            )
            draft_tokens_topk1[:, :1].copy_(topk_index)

        # Forward multiple steps
        scores = None
        with IndexTopKShareState.mtp_iteration(
            forward_batch,
            enabled=self.index_share_for_mtp_iteration,
            keep_carry_seed=self.seed_dsa_topk_from_draft_extend,
        ):
            for i in range(self.speculative_num_steps):
                if draft_tokens_topk1 is not None:
                    input_ids = topk_index.flatten()
                else:
                    input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                        i, topk_p, topk_index, hidden_states, scores, self.topk
                    )
                    score_list.append(tree_info[0])
                    token_list.append(tree_info[1])
                    parents_list.append(tree_info[2])

                if i == self.speculative_num_steps - 1:
                    break

                forward_batch.input_ids = input_ids
                # Qwen3-MoE MTP uses a fused RoPE + KV-store path whose cache_loc
                # argument must be contiguous.
                if (
                    self.draft_runner.model_config.hf_config.architectures[0]
                    == "Qwen3MoeForCausalLMMTP"
                ):
                    out_cache_loc = out_cache_loc.contiguous()
                forward_batch.out_cache_loc = out_cache_loc[i]
                spec_info.hidden_states = hidden_states

                canary_index_ctx = (
                    c.with_active_single_forward_manager(i)
                    if (c := self.draft_runner.canary_manager) is not None
                    else contextlib.nullcontext()
                )
                with (
                    forward_context(
                        ForwardContext(
                            attn_backend=self.draft_attn_backend.attn_backends[i]
                        )
                    ),
                    canary_index_ctx,
                ):
                    logits_output = self.draft_runner.forward(
                        forward_batch
                    ).logits_output
                maybe_detect_nan(
                    logits_output.next_token_logits, f"draft_forward step {i}"
                )
                maybe_detect_inf(
                    logits_output.next_token_logits, f"draft_forward step {i}"
                )
                if get_spec().speculative_use_rejection_sampling:
                    probs, topk_p, topk_index = sample_draft_proposal(
                        logits_output.next_token_logits,
                        forward_batch.sampling_info.temperatures,
                    )
                    draft_probs_list.append(probs)
                    forward_batch.positions.add_(1)
                elif self.topk == 1 and not _is_hip:
                    if _is_cuda:
                        topk_p, topk_index = draft_topk1_postprocess(
                            logits_output.next_token_logits,
                            forward_batch.positions,
                            draft_tokens_topk1,
                            i + 1,
                        )
                    else:
                        topk_index = torch.argmax(
                            logits_output.next_token_logits, dim=-1, keepdim=True
                        )
                        topk_p = torch.ones_like(topk_index, dtype=torch.float32)
                        forward_batch.positions.add_(1)
                else:
                    probs = renorm_draft_probs(
                        logits_output.next_token_logits,
                        forward_batch.sampling_info,
                        get_spec().speculative_use_rejection_sampling,
                    )
                    topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
                    forward_batch.positions.add_(1)
                maybe_detect_oob(
                    topk_index,
                    0,
                    logits_output.next_token_logits.shape[-1],
                    f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
                )
                if self.hot_token_id is not None:
                    topk_index = self.hot_token_id[topk_index]
                hidden_states = logits_output.hidden_states

        draft_probs = (
            torch.stack(draft_probs_list, dim=1)
            if get_spec().speculative_use_rejection_sampling
            else None
        )

        # Organize the results
        if draft_tokens_topk1 is not None:
            bs = draft_tokens_topk1.shape[0]
            top_scores_index = self._topk1_score_indices_prealloc[:bs]
            parent_list = self._topk1_parents_prealloc[:bs]
            return parent_list, top_scores_index, draft_tokens_topk1, draft_probs

        if topk1_chain_fits:
            bs = token_list[0].shape[0]
            draft_tokens = torch.cat(token_list, dim=1)
            top_scores_index = self._topk1_score_indices_prealloc[:bs]
            parent_list = self._topk1_parents_prealloc[:bs]
            return parent_list, top_scores_index, draft_tokens, draft_probs

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        return parent_list, top_scores_index, draft_tokens, draft_probs

    def draft_extend(self):
        pass

    def _mtp_sidecar_shadow_prefill(
        self,
        batch,
        target_hidden_states,
        next_token_ids,
        mm_input_embeds=None,
    ):
        """One-shot eager prefill probe on the CUDA2 TP1 MTP sidecar.

        This is deliberately shadow-only: the normal colocated draft result
        remains authoritative.  Failure here must not break serving.
        """
        sidecar = getattr(self, "_mtp_sidecar_probe", None)
        if sidecar is None:
            return

        if getattr(self, "_mtp_sidecar_shadow_prefill_attempted", False):
            return

        if batch.forward_mode.is_idle() or len(batch.reqs) != 1:
            return

        # For the first proof, require no target prefix-cache hit.  The sidecar
        # has an independent cache and therefore cannot reuse target KV.
        if int(batch.prefix_lens[0]) != 0:
            return

        self._mtp_sidecar_shadow_prefill_attempted = True

        import copy as _copy

        from sglang.srt.distributed.parallel_state import get_self_pp_group
        from sglang.srt.managers.schedule_batch import (
            set_mamba_track_indices_from_reqs,
        )

        sidecar_gpu_id = sidecar.gpu_id
        target_gpu_id = self.ps.gpu_id
        sidecar_device = f"cuda:{sidecar_gpu_id}"

        def _to_sidecar(t):
            if t is None:
                return None
            if not isinstance(t, torch.Tensor):
                return t
            # Avoid relying on CUDA peer access between the target 5070 and
            # the 5060.  This is a correctness probe, so stage through CPU.
            if t.device.type == "cuda":
                t = t.detach().to("cpu")
            return t.to(sidecar_device)

        try:
            torch.cuda.set_device(sidecar_gpu_id)

            mr = sidecar.model_runner

            if not getattr(self, "_mtp_sidecar_topology_checked", False):
                _attn_sizes = set()
                _comm_attn_sizes = set()
                _comm_tp_sizes = set()
                for _name, _mod in mr.model.named_modules():
                    _attn_size = getattr(_mod, "attn_tp_size", None)
                    if _attn_size is not None:
                        _attn_sizes.add(int(_attn_size))
                    _comm = getattr(_mod, "layer_communicator", None)
                    _ctx = getattr(_comm, "_context", None)
                    if _ctx is not None:
                        _comm_attn_sizes.add(int(_ctx.attn_tp_size))
                        _comm_tp_sizes.add(int(_ctx.tp_size))

                if _attn_sizes and _attn_sizes != {1}:
                    raise RuntimeError(
                        f"CUDA2 sidecar attention was not built TP1: {_attn_sizes}"
                    )
                if _comm_attn_sizes and _comm_attn_sizes != {1}:
                    raise RuntimeError(
                        "CUDA2 sidecar LayerCommunicator captured non-TP1 "
                        f"attention topology: {_comm_attn_sizes}"
                    )
                if _comm_tp_sizes and _comm_tp_sizes != {1}:
                    raise RuntimeError(
                        "CUDA2 sidecar LayerCommunicator captured non-TP1 "
                        f"generic topology: {_comm_tp_sizes}"
                    )

                self._mtp_sidecar_topology_checked = True
                logger.info(
                    "[MTP-SIDECAR-TOPO] static attention=%s communicator_attn=%s communicator_tp=%s",
                    sorted(_attn_sizes),
                    sorted(_comm_attn_sizes),
                    sorted(_comm_tp_sizes),
                )

            # get_rope() uses a process-wide module cache.  The target model
            # therefore leaves the shared RoPE cache on CUDA0 even though this
            # sidecar model itself lives on CUDA2.  Never move that shared
            # object in-place: doing so would also break the target model.
            #
            # Give the sidecar private RotaryEmbedding instances and migrate
            # only those copies to the sidecar device.
            if not getattr(self, "_mtp_sidecar_rope_isolated", False):
                _side_device = torch.device("cuda", sidecar_gpu_id)
                _rope_clones = {}
                _num_rope_users = 0

                for _mod_name, _mod in mr.model.named_modules():
                    _rope = getattr(_mod, "rotary_emb", None)
                    if _rope is None:
                        continue

                    _key = id(_rope)
                    _private_rope = _rope_clones.get(_key)

                    if _private_rope is None:
                        _private_rope = _copy.deepcopy(_rope)
                        _private_rope = _private_rope.to(_side_device)

                        # Be explicit in case this exact image keeps the cache
                        # as an ordinary tensor rather than a registered buffer.
                        _cache = getattr(_private_rope, "cos_sin_cache", None)
                        if isinstance(_cache, torch.Tensor):
                            _private_rope.cos_sin_cache = _cache.to(_side_device)

                        _rope_clones[_key] = _private_rope

                    _mod.rotary_emb = _private_rope
                    _num_rope_users += 1

                self._mtp_sidecar_rope_isolated = True

                logger.info(
                    "[MTP-SIDECAR-ROPE] isolated %d users into %d private CUDA%d RoPE object(s)",
                    _num_rope_users,
                    len(_rope_clones),
                    sidecar_gpu_id,
                )

            # A standalone TpModelWorker does not pass through the normal
            # EAGLE cuda-graph initialization path.  init_cuda_graphs() also
            # creates the EagerRunner and the runner attributes expected by
            # ModelRunner.forward().  Initialize through the official path,
            # then force this shadow probe back to eager-only execution.
            if (
                not hasattr(mr, "eager_runner")
                or not hasattr(mr, "prefill_cuda_graph_runner")
                or not hasattr(mr, "decode_cuda_graph_runner")
            ):
                with (
                    _mtp_sidecar_parallel_context(get_self_pp_group()),
                    draft_pp_context(),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                ):
                    mr.init_cuda_graphs(capture_decode_cuda_graph=False)

                mr.prefill_cuda_graph_runner = None
                mr.decode_cuda_graph_runner = None

                logger.info(
                    "[MTP-SIDECAR-SHADOW] CUDA%d eager runner initialized",
                    sidecar_gpu_id,
                )

            side_req_pool = mr.req_to_token_pool
            side_kv_alloc = mr.token_to_kv_pool_allocator

            # Persistent target Req objects must never be handed to the
            # sidecar pool because pool.alloc() mutates req_pool_idx and
            # Mamba ownership fields.  Use an isolated request clone.
            target_req = batch.reqs[0]
            side_req = _copy.copy(target_req)

            side_req.req_pool_idx = None
            side_req.mamba_pool_idx = None
            side_req.mamba_ping_pong_track_buffer = None
            side_req.mamba_next_track_idx = None
            side_req.mamba_last_track_seqlen = None
            side_req.mamba_branching_seqlen = None
            side_req.mamba_cow_src_index = None
            side_req.mamba_needs_clear = False
            side_req.mamba_lazy_is_insert = True

            side_req.kv_committed_len = 0
            side_req.kv = None
            side_req.extend_batch_idx = 0
            side_req.decode_batch_idx = 0

            side_req.prefix_indices = torch.empty(
                (0,), dtype=torch.int64, device=sidecar_device
            )
            side_req.last_node = None
            side_req.last_host_node = None
            side_req.best_match_node = None
            side_req.cache_protected_len = 0
            side_req.num_matched_prefix_tokens = 0
            side_req.host_hit_length = 0
            side_req.swa_host_hit_length = 0
            side_req.mamba_host_hit_length = 0

            rows = side_req_pool.alloc([side_req])
            if rows is None:
                raise RuntimeError("CUDA2 sidecar request pool allocation failed")

            side_req_idx = int(rows[0])

            extend_num_tokens = int(batch.extend_num_tokens)
            side_out_cache_loc = side_kv_alloc.alloc(extend_num_tokens)
            if side_out_cache_loc is None:
                raise RuntimeError(
                    f"CUDA2 sidecar KV allocation failed: need={extend_num_tokens}"
                )

            prefix_len = int(batch.prefix_lens[0])
            seq_len = int(batch.seq_lens_cpu[0])
            extend_len = int(batch.extend_lens[0])

            if extend_len != extend_num_tokens:
                raise RuntimeError(
                    f"unexpected probe shape: extend_len={extend_len}, "
                    f"extend_num_tokens={extend_num_tokens}"
                )

            # page_size == 1: map this extend directly into the independent
            # CUDA2 request row.
            side_req_pool.write(
                (side_req_idx, slice(prefix_len, seq_len)),
                side_out_cache_loc[:extend_len],
            )
            side_req.kv_committed_len = seq_len

            # Build an isolated ScheduleBatch view.  Never mutate the target
            # batch because the normal colocated draft still runs afterwards.
            side_batch = _copy.copy(batch)
            side_batch.reqs = [side_req]
            side_batch.req_to_token_pool = side_req_pool
            side_batch.token_to_kv_pool_allocator = side_kv_alloc
            side_batch.device = sidecar_device

            side_batch.req_pool_indices_cpu = torch.tensor(
                [side_req_idx], dtype=torch.int64
            )
            side_batch.req_pool_indices = side_batch.req_pool_indices_cpu.to(
                sidecar_device
            )

            side_batch.seq_lens_cpu = batch.seq_lens_cpu.clone()
            side_batch.seq_lens = _to_sidecar(batch.seq_lens)
            side_batch.orig_seq_lens = _to_sidecar(batch.orig_seq_lens)
            side_batch.out_cache_loc = side_out_cache_loc

            # Recreate the shifted MTP prefill token stream without touching
            # batch.input_ids.
            tail_tokens = _eagle_prefill_tail_tokens(batch, next_token_ids)
            new_input_ids = torch.empty_like(batch.input_ids)
            pt = 0
            for i, cur_extend_len in enumerate(batch.extend_lens):
                input_ids = batch.input_ids[pt : pt + cur_extend_len]
                new_input_ids[pt : pt + cur_extend_len].copy_(
                    torch.cat(
                        (input_ids[1:], tail_tokens[i].reshape(1))
                    )
                )
                pt += cur_extend_len

            side_batch.input_ids = _to_sidecar(new_input_ids)

            side_batch.spec_info = EagleDraftExtendInput(
                hidden_states=_to_sidecar(target_hidden_states),
                num_tokens_per_req=1,
                num_tokens_for_logprob_per_req=1,
            )

            # Shape-dependent tracking data can be mirrored; slot identities
            # must be rebuilt from the CUDA2 HybridReqToTokenPool.
            side_batch.mamba_track_mask = _to_sidecar(
                getattr(batch, "mamba_track_mask", None)
            )
            side_batch.mamba_track_seqlens = _to_sidecar(
                getattr(batch, "mamba_track_seqlens", None)
            )
            side_batch.mamba_lazy_spec_track_positions_cpu = None

            if hasattr(side_req_pool, "mamba_pool"):
                set_mamba_track_indices_from_reqs(side_batch)
                side_batch._collect_deferred_mamba_cow_and_clear([side_req])

            capture_hidden_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.LAST
            )

            with (
                _mtp_sidecar_parallel_context(get_self_pp_group()),
                draft_pp_context(),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                forward_batch = ForwardBatch.init_new(
                    side_batch,
                    mr,
                    capture_hidden_mode=capture_hidden_mode,
                    return_hidden_states_before_norm=False,
                )
                forward_batch.return_logprob = False

                if mm_input_embeds is not None:
                    forward_batch.mm_input_embeds = _to_sidecar(mm_input_embeds)

                # Diagnose any target/CPU tensors that survived ForwardBatch.init_new.
                for _name in (
                    "input_ids",
                    "positions",
                    "seq_lens",
                    "req_pool_indices",
                    "out_cache_loc",
                    "mamba_track_indices",
                    "mamba_track_mask",
                    "mamba_track_seqlens",
                ):
                    _v = getattr(forward_batch, _name, None)
                    if isinstance(_v, torch.Tensor):
                        logger.info(
                            "[MTP-SIDECAR-DEV] %s device=%s dtype=%s shape=%s",
                            _name,
                            _v.device,
                            _v.dtype,
                            tuple(_v.shape),
                        )

                _spec = getattr(forward_batch, "spec_info", None)
                if _spec is not None:
                    _h = getattr(_spec, "hidden_states", None)
                    if isinstance(_h, torch.Tensor):
                        logger.info(
                            "[MTP-SIDECAR-DEV] spec.hidden_states device=%s dtype=%s shape=%s",
                            _h.device,
                            _h.dtype,
                            tuple(_h.shape),
                        )

                # Inspect the persistent tensors passed directly into
                # fused_qk_gemma_rmsnorm_rope_gate.
                for _mod_name, _mod in mr.model.named_modules():
                    if not (
                        hasattr(_mod, "q_norm")
                        and hasattr(_mod, "k_norm")
                        and hasattr(_mod, "rotary_emb")
                        and hasattr(_mod, "forward_prepare_cuda_fused")
                    ):
                        continue

                    _qnorm = getattr(_mod.q_norm, "weight", None)
                    _knorm = getattr(_mod.k_norm, "weight", None)
                    _rope = getattr(_mod.rotary_emb, "cos_sin_cache", None)

                    logger.info(
                        "[MTP-SIDECAR-PTR] module=%s qnorm=%s knorm=%s rope=%s",
                        _mod_name,
                        (
                            str(_qnorm.device)
                            if isinstance(_qnorm, torch.Tensor)
                            else type(_qnorm).__name__
                        ),
                        (
                            str(_knorm.device)
                            if isinstance(_knorm, torch.Tensor)
                            else type(_knorm).__name__
                        ),
                        (
                            str(_rope.device)
                            if isinstance(_rope, torch.Tensor)
                            else type(_rope).__name__
                        ),
                    )

                # ForwardBatch.init_new and the model forward are already
                # enclosed by _mtp_sidecar_parallel_context, so do not nest
                # SGLang's non-reentrant draft TP/PP patchers here.
                side_logits = mr.forward(forward_batch).logits_output

                # Surface asynchronous CUDA faults on CUDA2 here.  Otherwise a
                # shadow-side illegal access can be reported later by the
                # authoritative target draft and produce a misleading traceback.
                torch.cuda.synchronize(sidecar_gpu_id)

            if side_logits.next_token_logits is None:
                raise RuntimeError("CUDA2 sidecar returned no next_token_logits")

            probe_token = int(
                torch.argmax(
                    side_logits.next_token_logits[-1],
                    dim=-1,
                ).item()
            )

            # Exercise the real TOPK=1 EAGLE multi-step chain on CUDA2 before
            # the stock draft runs.  Reserve the sidecar's future draft KV slots
            # explicitly: prepare_for_draft() reads them from req_to_token at
            # [seq_len : seq_len + topk*num_steps].  The scheduler normally
            # pre-populates those slots for the colocated draft; our independent
            # sidecar pool must do that itself.
            if self.topk != 1:
                raise RuntimeError(
                    f"CUDA2 multistep shadow currently requires topk=1, got {self.topk}"
                )

            _num_steps = int(self.speculative_num_steps)
            _future_slots = side_kv_alloc.alloc(_num_steps)
            if _future_slots is None:
                raise RuntimeError(
                    f"CUDA2 sidecar draft KV reservation failed: need={_num_steps}"
                )
            side_req_pool.write(
                (side_req_idx, slice(seq_len, seq_len + _num_steps)),
                _future_slots,
            )

            _side_initial_token = torch.argmax(
                side_logits.next_token_logits, dim=-1, keepdim=True
            ).to(torch.int64)
            _side_draft_input = EagleDraftInput(
                topk_p=torch.ones_like(_side_initial_token, dtype=torch.float32),
                topk_index=_side_initial_token,
                draft_probs=None,
                hidden_states=side_logits.hidden_states,
                bonus_tokens=_to_sidecar(next_token_ids),
                num_tokens_per_req=1,
                num_tokens_for_logprob_per_req=1,
            )

            # prepare_for_draft mutates the ScheduleBatch view only.  Reuse the
            # already-isolated side_batch, whose request row and pools belong to
            # CUDA2, and create a sidecar-specific multi-step attention backend
            # once.  Do not borrow the colocated draft backend: it owns CUDA0
            # buffers and TP2 construction state.
            side_batch.spec_info = _side_draft_input
            from sglang.srt.model_executor.forward_batch_info import ForwardMode as _ForwardMode

            side_batch.forward_mode = _ForwardMode.DECODE
            side_batch.input_ids = _side_initial_token.reshape(-1)
            side_batch.seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)
            side_batch.seq_lens = side_batch.seq_lens_cpu.to(sidecar_device)
            side_batch.seq_lens_sum = seq_len
            logger.info(
                "[MTP-SIDECAR-DECODE-VIEW] mode=%s input_shape=%s seq_len=%d",
                side_batch.forward_mode,
                tuple(side_batch.input_ids.shape),
                seq_len,
            )

            with (
                _mtp_sidecar_parallel_context(get_self_pp_group()),
                draft_pp_context(),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                if not hasattr(self, "_mtp_sidecar_draft_attn_backend"):
                    # This WIP file is mounted into an exact Docker image while
                    # the fork around it can be a newer SGLang revision.
                    # DraftBackendFactory changed constructor shape between
                    # those revisions, so resolve it from the runtime class
                    # instead of hard-coding either ABI.
                    import inspect as _inspect

                    _factory_params = _inspect.signature(
                        DraftBackendFactory.__init__
                    ).parameters
                    _factory_kwargs = {
                        "draft_model_runner": mr,
                        "topk": self.topk,
                        "speculative_num_steps": self.speculative_num_steps,
                        "seed_dsa_topk_from_draft_extend": False,
                    }
                    if "server_args" in _factory_params:
                        _factory_kwargs["server_args"] = self.server_args

                    logger.info(
                        "[MTP-SIDECAR-FACTORY] DraftBackendFactory params=%s server_args=%s",
                        list(_factory_params),
                        "server_args" in _factory_params,
                    )
                    _side_factory = DraftBackendFactory(**_factory_kwargs)
                    self._mtp_sidecar_draft_attn_backend = (
                        _side_factory.create_decode_backend()
                    )

                _side_fb, _ = prepare_for_draft(
                    _side_draft_input,
                    side_req_pool,
                    side_batch,
                    None,
                    mr,
                    self.topk,
                    self.speculative_num_steps,
                )
                _side_fb.return_logprob = False
                _side_attn = self._mtp_sidecar_draft_attn_backend
                if self.speculative_num_steps > 1:
                    _side_attn.init_forward_metadata(_side_fb)
                    _side_fb.mark_forward_metadata_ready()

                _out_cache = per_step_draft_out_cache_loc(
                    _side_fb.out_cache_loc,
                    _side_fb.batch_size,
                    self.topk,
                    self.speculative_num_steps,
                )
                _cur_token = _side_initial_token
                _cur_hidden = side_logits.hidden_states
                _proposal = [_cur_token.reshape(-1)]

                # num_steps=3 means two actual MTP forwards after the initial
                # token from prefill logits.  This mirrors draft_forward's
                # topk1 fast path without touching the stock worker's state.
                for _i in range(self.speculative_num_steps - 1):
                    _side_fb.input_ids = _cur_token.reshape(-1)
                    _side_fb.out_cache_loc = _out_cache[_i]
                    _side_fb.spec_info.hidden_states = _cur_hidden
                    with forward_context(
                        ForwardContext(attn_backend=_side_attn.attn_backends[_i])
                    ):
                        _step_logits = mr.forward(_side_fb).logits_output
                    torch.cuda.synchronize(sidecar_gpu_id)
                    _cur_token = torch.argmax(
                        _step_logits.next_token_logits, dim=-1, keepdim=True
                    ).to(torch.int64)
                    _proposal.append(_cur_token.reshape(-1))
                    _cur_hidden = _step_logits.hidden_states
                    _side_fb.positions.add_(1)

            self._mtp_sidecar_multistep_proposal = (
                torch.stack(_proposal, dim=1).detach().cpu()
            )
            logger.info(
                "[MTP-SIDECAR-MULTISTEP] CUDA%d proposal=%s",
                sidecar_gpu_id,
                self._mtp_sidecar_multistep_proposal.tolist(),
            )

            # Save a tiny one-shot correctness snapshot on host memory.  The
            # authoritative colocated draft runs immediately after this helper
            # on the same logical prefill.  Host staging deliberately avoids
            # CUDA2->CUDA0 P2P/IPC assumptions while we validate equivalence.
            self._mtp_sidecar_shadow_compare = {
                "logits": side_logits.next_token_logits.detach().float().cpu(),
                "hidden": (
                    side_logits.hidden_states.detach().float().cpu()
                    if side_logits.hidden_states is not None
                    else None
                ),
                "argmax": probe_token,
            }

            logger.info(
                "[MTP-SIDECAR-SHADOW] eager prefill SUCCESS "
                "CUDA%d req_row=%d seq_len=%d extend=%d "
                "logits=%s hidden=%s argmax=%d",
                sidecar_gpu_id,
                side_req_idx,
                seq_len,
                extend_len,
                tuple(side_logits.next_token_logits.shape),
                (
                    tuple(side_logits.hidden_states.shape)
                    if side_logits.hidden_states is not None
                    else None
                ),
                probe_token,
            )

        except Exception:
            logger.exception(
                "[MTP-SIDECAR-SHADOW] eager prefill FAILED"
            )
        finally:
            torch.cuda.set_device(target_gpu_id)

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor] = None,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # One-shot CUDA2 shadow execution.  It never replaces the normal
        # colocated draft result at this stage.
        self._mtp_sidecar_shadow_prefill(
            batch,
            target_hidden_states,
            next_token_ids,
            mm_input_embeds,
        )

        # Construct input_ids
        if not batch.forward_mode.is_idle():
            # Chunked-prefill-aware tail tokens (see PR #26329).
            tail_tokens = _eagle_prefill_tail_tokens(batch, next_token_ids)
            new_input_ids = torch.empty_like(batch.input_ids)
            pt = 0
            for i, extend_len in enumerate(batch.extend_lens):
                input_ids = batch.input_ids[pt : pt + extend_len]
                new_input_ids[pt : pt + extend_len].copy_(
                    torch.cat((input_ids[1:], tail_tokens[i].reshape(1)))
                )
                pt += extend_len
            assert pt == batch.input_ids.numel()
            batch.input_ids = new_input_ids

        # Draft-extend spec_info for the extend forward; carries only
        # hidden_states + shape info.
        batch.spec_info = EagleDraftExtendInput(
            hidden_states=target_hidden_states,
            # draft mode is same with decode mode, only 1 token per req
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        # Run forward (LAST mode: only the final hidden state per request,
        # to feed the next draft step which expects [bs, hidden_dim]).
        # STANDALONE skips hidden states end-to-end.
        capture_hidden_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        forward_batch = ForwardBatch.init_new(
            batch,
            self.draft_runner,
            capture_hidden_mode=capture_hidden_mode,
            return_hidden_states_before_norm=False,
        )
        forward_batch.return_logprob = False
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds

        # Seed the first draft-decode loop from each request's last prefill
        # position. Gather last-per-req before the copy (prefill can be long).
        seed_from_extend = (
            self.seed_dsa_topk_from_draft_extend
            and not forward_batch.forward_mode.is_idle()
        )
        if seed_from_extend:
            bs = forward_batch.batch_size
            forward_batch.spec_info.dsa_seed_topk_capture = (
                self._get_dsa_extend_topk_buf(bs)
            )
            forward_batch.spec_info.dsa_seed_topk_select = (
                torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            ).long()

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            logits_output = self.draft_runner.forward(forward_batch).logits_output
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")
        maybe_detect_inf(logits_output.next_token_logits, "draft_extend_for_prefill")

        # One-shot numerical comparison between the CUDA2 TP1 sidecar and the
        # authoritative colocated draft.  Both consumed the same shifted token
        # stream and target hidden states.  MTP uses top-k=1 here, so top1
        # agreement is the primary semantic check; cosine / absolute deltas
        # expose subtler loader or TP-sharding mistakes before replacement.
        _side_cmp = getattr(self, "_mtp_sidecar_shadow_compare", None)
        if _side_cmp is not None:
            try:
                _stock_logits = logits_output.next_token_logits.detach().float().cpu()
                _side_logits_cpu = _side_cmp["logits"]

                if _stock_logits.shape != _side_logits_cpu.shape:
                    logger.error(
                        "[MTP-SIDECAR-COMPARE] SHAPE_MISMATCH side=%s stock=%s",
                        tuple(_side_logits_cpu.shape),
                        tuple(_stock_logits.shape),
                    )
                else:
                    _delta = (_side_logits_cpu - _stock_logits).abs()
                    _side_flat = _side_logits_cpu.reshape(-1)
                    _stock_flat = _stock_logits.reshape(-1)
                    _logit_cos = torch.nn.functional.cosine_similarity(
                        _side_flat, _stock_flat, dim=0
                    ).item()

                    _side_argmax = int(torch.argmax(_side_logits_cpu[-1]).item())
                    _stock_argmax = int(torch.argmax(_stock_logits[-1]).item())
                    _k = min(5, _stock_logits.shape[-1])
                    _side_top5 = set(
                        torch.topk(_side_logits_cpu[-1], k=_k).indices.tolist()
                    )
                    _stock_top5 = set(
                        torch.topk(_stock_logits[-1], k=_k).indices.tolist()
                    )

                    _hidden_cos = None
                    _hidden_max_abs = None
                    _side_hidden = _side_cmp.get("hidden")
                    _stock_hidden = (
                        logits_output.hidden_states.detach().float().cpu()
                        if logits_output.hidden_states is not None
                        else None
                    )
                    if (
                        _side_hidden is not None
                        and _stock_hidden is not None
                        and _side_hidden.shape == _stock_hidden.shape
                    ):
                        _hidden_cos = torch.nn.functional.cosine_similarity(
                            _side_hidden.reshape(-1),
                            _stock_hidden.reshape(-1),
                            dim=0,
                        ).item()
                        _hidden_max_abs = (
                            (_side_hidden - _stock_hidden).abs().max().item()
                        )

                    logger.info(
                        "[MTP-SIDECAR-COMPARE] "
                        "shape=%s side_argmax=%d stock_argmax=%d top1_match=%s "
                        "top5_overlap=%d logit_cosine=%.9f "
                        "max_abs=%.7g mean_abs=%.7g "
                        "hidden_cosine=%s hidden_max_abs=%s",
                        tuple(_stock_logits.shape),
                        _side_argmax,
                        _stock_argmax,
                        _side_argmax == _stock_argmax,
                        len(_side_top5 & _stock_top5),
                        _logit_cos,
                        _delta.max().item(),
                        _delta.mean().item(),
                        (
                            f"{_hidden_cos:.9f}"
                            if _hidden_cos is not None
                            else None
                        ),
                        (
                            f"{_hidden_max_abs:.7g}"
                            if _hidden_max_abs is not None
                            else None
                        ),
                    )
            finally:
                self._mtp_sidecar_shadow_compare = None

        prefill_dsa_topk = None
        if seed_from_extend:
            prefill_dsa_topk = self.dsa_extend_topk_buf[:bs].clone()

        # Assemble the next-iter draft spec_info from the extend output.
        use_rejection_sampling = get_spec().speculative_use_rejection_sampling
        probs = renorm_draft_probs(
            logits_output.next_token_logits,
            batch.sampling_info,
            use_rejection_sampling,
        )
        if use_rejection_sampling:
            topk_p, topk_index = fast_sample(probs, num_samples=1)
        else:
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        return EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            draft_probs=probs if use_rejection_sampling else None,
            hidden_states=logits_output.hidden_states,
            bonus_tokens=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
            dsa_topk_indices=prefill_dsa_topk,
        )

    def _get_dsa_extend_topk_buf(self, num_tokens: int) -> torch.Tensor:
        """Lazily-grown int32 [num_tokens, index_topk] eager draft-extend seed buffer."""
        buf = self.dsa_extend_topk_buf
        if buf is None or buf.shape[0] < num_tokens:
            buf = torch.full(
                (num_tokens, self.dsa_index_topk),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            self.dsa_extend_topk_buf = buf
        return buf[:num_tokens]

    def _draft_extend_for_decode(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        # Batch 2: Draft extend
        draft_extend_input = EagleDraftExtendInput(
            hidden_states=batch_result.logits_output.hidden_states,
            # accept_lens includes the bonus token; correct drafts exclude it.
            num_correct_drafts=batch_result.accept_lens - 1,
            num_accept_tokens=batch_result.accept_lens,
            # Draft-extend fills the whole tree width (num_draft_tokens) per req,
            # not num_steps + 1, so DP MLP-sync padding stays consistent for topk > 1.
            num_tokens_per_req=self.speculative_num_draft_tokens,
            num_tokens_for_logprob_per_req=self.speculative_num_draft_tokens,
        )
        select_index = (
            torch.arange(
                0,
                len(batch.seq_lens) * self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens,
                device=self.device,
            )
            + batch_result.accept_lens
            - 1
        )

        # Cast to int64 before entering plan stream to avoid cross-stream
        # synchronization issues with .to() inside the plan stream context.
        next_token_ids = batch_result.next_token_ids.to(torch.int64)

        # Prepare for draft extend in a separate stream
        with self.plan_stream_ctx:
            forward_batch = prepare_for_draft_extend(
                draft_extend_input,
                batch,
                next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner,
                self.cuda_graph_runner_for_draft_extend,
                return_hidden_states_before_norm=False,
            )

        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

        # Run draft extend batch in the main compute stream
        can_run_decode_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run_graph(forward_batch)
        )

        # Eager path publishes the indexer top-k into a worker buffer (the graph
        # path uses the runner's static buffer). Gathered at select_index below.
        if self.seed_dsa_topk_from_draft_extend and not can_run_decode_cuda_graph:
            forward_batch.spec_info.dsa_seed_topk_capture = (
                self._get_dsa_extend_topk_buf(forward_batch.input_ids.shape[0])
            )

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            if can_run_decode_cuda_graph:
                draft_logits_output = self.cuda_graph_runner_for_draft_extend.execute(
                    forward_batch
                )
            else:
                draft_logits_output = self.draft_runner.forward(
                    forward_batch
                ).logits_output

        maybe_detect_nan(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_run_decode_cuda_graph})",
        )
        maybe_detect_inf(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_run_decode_cuda_graph})",
        )

        # Gather the per-request last-position indexer top-k as the next loop's
        # seed (select_index already picks the last accepted position per req).
        dsa_seed_topk_indices = None
        if self.seed_dsa_topk_from_draft_extend:
            if can_run_decode_cuda_graph:
                dsa_extend_topk_capture = (
                    self.cuda_graph_runner_for_draft_extend.buffers.dsa_seed_topk_capture
                )
            else:
                dsa_extend_topk_capture = forward_batch.spec_info.dsa_seed_topk_capture
            # Fancy indexing returns a fresh tensor (detached from the buffer).
            dsa_seed_topk_indices = dsa_extend_topk_capture[select_index]

        # Reorganize the spec info for the next batch
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        if draft_logits_output.hidden_states is not None:
            draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                select_index
            ]
        # The draft-extend graph only anchors full logits; selected-row topk is
        # owned by the worker for both graph and eager paths.
        if get_spec().speculative_use_rejection_sampling:
            ret_draft_probs, ret_topk_p, ret_topk_index = sample_draft_proposal(
                draft_logits_output.next_token_logits,
                batch.sampling_info.temperatures,
            )
        elif self.topk == 1 and not _is_hip:
            # Gated to CUDA: see #26358 — ROCm's argmax tie-break corrupts
            # MTP draft selection on FP8 logits.
            ret_topk_index = torch.argmax(
                draft_logits_output.next_token_logits, dim=-1, keepdim=True
            )
            ret_topk_p = torch.ones_like(ret_topk_index, dtype=torch.float32)
            ret_draft_probs = None
        else:
            probs = renorm_draft_probs(
                draft_logits_output.next_token_logits,
                batch.sampling_info,
                get_spec().speculative_use_rejection_sampling,
            )
            ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
            ret_draft_probs = None
        ret_hidden_states = draft_logits_output.hidden_states

        # Construct the return values
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            ret_topk_p,
            ret_topk_index,
            ret_hidden_states,
        )
        if get_spec().speculative_use_rejection_sampling:
            next_draft_input.draft_probs = ret_draft_probs
        if self.seed_dsa_topk_from_draft_extend:
            next_draft_input.dsa_topk_indices = dsa_seed_topk_indices


class EAGLEWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()

        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.ps = ps
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Authoritative local 3-GPU MTP cutover.
        # Target stays TP2 on CUDA0/CUDA1.  Only target TP0 constructs a draft,
        # and that draft is a self-contained TP1 worker on CUDA2.  TP1 is a
        # relay/verify participant and therefore carries no colocated MTP weights.
        self._hosts_draft = get_pp_group().is_last_rank
        self._mtp_sidecar_authoritative = self._hosts_draft

        if self._mtp_sidecar_authoritative:
            if ps.pp_size != 1:
                raise RuntimeError("MTP cutover currently requires PP=1")
            if ps.tp_size != 2:
                raise RuntimeError(
                    f"MTP cutover requires target TP=2, got {ps.tp_size}"
                )
            if self.topk != 1:
                raise RuntimeError(
                    f"MTP cutover currently requires EAGLE topk=1, got {self.topk}"
                )
            if server_args.speculative_adaptive:
                raise RuntimeError("MTP cutover does not yet support adaptive speculative")
            if get_spec().speculative_use_rejection_sampling:
                raise RuntimeError("MTP cutover does not yet support rejection sampling")
            if not server_args.disable_radix_cache:
                raise RuntimeError(
                    "MTP cutover gate requires --disable-radix-cache so CUDA2 can "
                    "mirror every target prefill chunk without a separate radix tree"
                )
            if torch.cuda.device_count() < 3:
                raise RuntimeError(
                    f"MTP cutover requires 3 visible GPUs, got {torch.cuda.device_count()}"
                )

        self._draft_worker = None
        if self._mtp_sidecar_authoritative and ps.tp_rank == 0:
            from sglang.srt.distributed.parallel_state import get_self_pp_group

            sidecar_gpu_id = 2
            target_gpu_id = ps.gpu_id
            sidecar_ps = ParallelState.trivial(gpu_id=sidecar_gpu_id)
            logger.info(
                "[MTP-CUTOVER] constructing authoritative TP1 draft on CUDA%d (%s)",
                sidecar_gpu_id,
                torch.cuda.get_device_name(sidecar_gpu_id),
            )
            try:
                torch.cuda.set_device(sidecar_gpu_id)
                # EagleDraftWorker.__init__ owns draft_pp_context and the
                # speculative MoE contexts itself.  The PP patcher is deliberately
                # non-reentrant, so the outer worker must establish only the TP1 /
                # attention topology required while the sidecar modules are built.
                with _mtp_sidecar_parallel_context(get_self_pp_group()):
                    self._draft_worker = EagleDraftWorker(
                        server_args,
                        sidecar_gpu_id,
                        sidecar_ps,
                        nccl_port,
                        target_worker,
                        mtp_authoritative_sidecar=True,
                    )
            finally:
                torch.cuda.set_device(target_gpu_id)
        elif self._mtp_sidecar_authoritative:
            logger.info(
                "[MTP-CUTOVER] target TP rank %d is relay-only; no local draft model",
                ps.tp_rank,
            )

        # Adaptive speculative
        self.adaptive_controller: Optional[AdaptiveController] = None
        if server_args.speculative_adaptive and self._hosts_draft:
            self.adaptive_controller = AdaptiveController(
                self,
                config_path=server_args.speculative_adaptive_config,
            )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        super().alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        if self._mtp_sidecar_authoritative:
            target_tokens = int(self.target_worker.model_runner.max_total_num_tokens)
            if target_tokens < 65536:
                raise RuntimeError(
                    f"MTP cutover target KV pool too small: "
                    f"{target_tokens} < 65536"
                )
            side_tokens = (
                int(self._draft_worker.draft_runner.max_total_num_tokens)
                if self._draft_worker is not None
                else -1
            )
            logger.info(
                "[MTP-CUTOVER-POOL] target_rank=%d target_tokens=%d side_tokens=%d",
                self.ps.tp_rank,
                target_tokens,
                side_tokens,
            )

    @property
    def war_fastpath_runner(self):
        if self._mtp_sidecar_authoritative:
            # Sidecar work completes synchronously before publish/return, so the
            # target runner is the last owner of scheduler-shared CUDA buffers.
            return self._target_worker.model_runner
        return self._draft_worker.draft_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        if self._mtp_sidecar_authoritative:
            # Keep target-side scheduling decisions rank-uniform. CUDA2 batches
            # build their own CPU seq-len mirrors explicitly.
            return (self._target_worker.model_runner.attn_backend,)
        return (
            self._target_worker.model_runner.attn_backend,
            self._draft_worker.draft_attn_backend,
            self._draft_worker.draft_extend_attn_backend
            or self._draft_worker.draft_runner.attn_backend,
        )

    def init_cuda_graphs(self):
        super().init_cuda_graphs()
        # Build adaptive runtime states after target and draft backends exist.
        if self.adaptive_controller is not None:
            with (
                self._draft_worker.draft_tp_context(
                    self._draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.adaptive_controller.register(
                    SpecRuntimeState(
                        speculative_num_steps=self.speculative_num_steps,
                        speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                        draft_attn_backend=self._draft_worker.draft_attn_backend,
                        cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                        target_attn_backend=self._target_worker.model_runner.attn_backend,
                        target_graph_runner=self._target_worker.model_runner.decode_cuda_graph_runner,
                        draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                        cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
                    )
                )
                self.adaptive_controller.init_states(
                    cuda_graph_bs=(
                        None
                        if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)
                        else get_exec().graph.cuda_graph_bs_decode
                    ),
                )

    def _mtp_cutover_relay_tensor(self, side_tensor, shape, dtype):
        """CUDA2 -> CPU -> target CUDA0, then TP0 -> TP1 target broadcast."""
        target_device = torch.device("cuda", self.gpu_id)
        torch.cuda.set_device(self.gpu_id)
        if self.ps.tp_rank == 0:
            if side_tensor is None:
                raise RuntimeError("TP0 relay source tensor is missing")
            cpu = side_tensor.detach().to("cpu")
            out = cpu.to(device=target_device, dtype=dtype)
            if tuple(out.shape) != tuple(shape):
                raise RuntimeError(
                    f"TP0 relay shape mismatch: {tuple(out.shape)} != {tuple(shape)}"
                )
        else:
            out = torch.empty(shape, dtype=dtype, device=target_device)
        self.target_worker.model_runner.tp_group.broadcast(out, src=0)
        return out

    def _mtp_cutover_relay_draft_input(self, side_input, bs: int):
        cfg = getattr(
            self.target_worker.model_config.hf_config,
            "text_config",
            self.target_worker.model_config.hf_config,
        )
        hidden_size = int(cfg.hidden_size)
        topk_p = self._mtp_cutover_relay_tensor(
            None if side_input is None else side_input.topk_p,
            (bs, self.topk),
            torch.float32,
        )
        topk_index = self._mtp_cutover_relay_tensor(
            None if side_input is None else side_input.topk_index,
            (bs, self.topk),
            torch.int64,
        )
        hidden = self._mtp_cutover_relay_tensor(
            None if side_input is None else side_input.hidden_states,
            (bs, hidden_size),
            torch.bfloat16,
        )
        bonus = self._mtp_cutover_relay_tensor(
            None if side_input is None else side_input.bonus_tokens,
            (bs,),
            torch.int32,
        )
        return EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden,
            bonus_tokens=bonus,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def _mtp_cutover_relay_proposal(self, side_tokens, bs: int):
        return self._mtp_cutover_relay_tensor(
            side_tokens,
            (bs, self.speculative_num_steps),
            torch.int64,
        )

    def _mtp_cutover_forward(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ):
        # Idle overlap batches do not carry a real request or sidecar state.
        if batch.forward_mode.is_idle():
            batch.spec_info = EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                self.device,
            )
            return self.verify(batch, grammar_barrier=grammar_barrier)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            target_capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.FULL
            )
            batch_output = self.target_worker.forward_batch_generation(
                batch,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=target_capture_mode,
            )
            batch_output.new_seq_lens = batch.seq_lens

            side_next = None
            if self.ps.tp_rank == 0:
                side_next = self._draft_worker.mtp_authoritative_prefill(
                    batch,
                    batch_output.logits_output.hidden_states,
                    batch_output.next_token_ids,
                    batch_output.logits_output.mm_input_embeds,
                )
            # Rank1 waits here while TP0 finishes CUDA2; afterwards both ranks
            # hold a shape-valid target-device EagleDraftInput.
            batch_output.next_draft_input = self._mtp_cutover_relay_draft_input(
                side_next, len(batch.reqs)
            )
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            return batch_output

        if batch.spec_info is None:
            raise RuntimeError(
                "MTP cutover decode entered without relayed EagleDraftInput"
            )

        bs = len(batch.reqs)
        side_tokens = None
        if self.ps.tp_rank == 0:
            side_tokens = self._draft_worker.mtp_authoritative_draft_tokens(batch)
        draft_tokens = self._mtp_cutover_relay_proposal(side_tokens, bs)

        # topk=1 has a deterministic chain topology; only tokens need crossing
        # ranks. Build each target rank's verify metadata locally.
        parent_list = torch.arange(
            -1,
            self.speculative_num_steps - 1,
            dtype=torch.long,
            device=draft_tokens.device,
        ).repeat(bs, 1)
        top_scores_index = torch.arange(
            self.speculative_num_steps,
            dtype=torch.long,
            device=draft_tokens.device,
        ).repeat(bs, 1)
        verify_input = build_eagle_verify_input(
            batch,
            batch.spec_info,
            parent_list,
            top_scores_index,
            draft_tokens,
            None,
            target_worker=self.target_worker,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            tree_mask_mode=default_tree_mask_mode(),
            device=self.device,
        )
        batch.spec_info = verify_input
        batch_output = self.verify(batch, grammar_barrier=grammar_barrier)

        side_next = None
        if self.ps.tp_rank == 0:
            side_next = self._draft_worker.mtp_authoritative_extend_after_verify(
                batch, batch_output
            )
        batch_output.next_draft_input = self._mtp_cutover_relay_draft_input(
            side_next, bs
        )
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)
        return batch_output

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ):
        if self._mtp_sidecar_authoritative:
            return self._mtp_cutover_forward(
                batch,
                on_publish=on_publish,
                grammar_barrier=grammar_barrier,
                pp_proxy_tensors=pp_proxy_tensors,
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # Target prefill
            target_capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.FULL
            )
            batch_output = self.target_worker.forward_batch_generation(
                batch,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=target_capture_mode,
            )

            # Spec_v2 convention: batch.seq_lens = length BEFORE this iter's tokens.
            # Extend processed L prompt tokens; next verify iter expects same L.
            batch_output.new_seq_lens = batch.seq_lens
            # Publish before draft_extend so the fence is at target-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            # A rank that does not host the draft (prefill-side PP builds it only on
            # the last stage) forwards the target's proxy tensors and stops here.
            if self._draft_worker is None:
                return batch_output

            # Draft prefill
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft_extend"),
            ):
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                        batch_output.logits_output.mm_input_embeds,
                    )
                )
                return batch_output
        else:
            self.activate_step_by_batch(batch.seq_lens.shape[0])

            if batch.spec_info is None:
                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.speculative_algorithm.is_standalone()
                    else CaptureHiddenMode.LAST
                )
                hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
                    self.draft_worker.draft_runner
                )
                batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=hidden_size,
                    dtype=hidden_dtype,
                    topk=self.topk,
                    capture_hidden_mode=capture_mode,
                    vocab_size=self.target_worker.model_config.vocab_size,
                )
            if self.speculative_num_steps == 0:
                # Drafting disabled (high batch size). _draft_extend below still
                # runs, keeping draft KV warm for when the batch shrinks.
                verify_input = self._build_trivial_verify_input(batch)
            else:
                with (
                    self.draft_worker.draft_tp_context(
                        self.draft_worker.draft_runner.tp_group
                    ),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                    spec_stage_span("draft"),
                ):
                    verify_input: EagleVerifyInput = self.draft_worker.draft(batch)
            assert verify_input.is_verify_input()
            batch.spec_info = verify_input
            batch_output = self.verify(batch, grammar_barrier=grammar_barrier)
            # Publish before draft_extend so the fence is at verify-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            if (
                self.speculative_num_steps == 0
                and envs.SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND.get()
            ):
                self._stub_skipped_draft_extend(batch, batch_output)
            else:
                with (
                    self.draft_worker.draft_tp_context(
                        self.draft_worker.draft_runner.tp_group
                    ),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                    spec_stage_span("draft_extend"),
                ):
                    self.draft_worker._draft_extend_for_decode(batch, batch_output)

            return batch_output

    def _build_trivial_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        """Build a 1-node EagleVerifyInput rooted at the previous bonus token.

        Used when ``speculative_num_steps == 0`` to skip drafting while still
        routing through the existing TARGET_VERIFY graph captured at
        ``draft_token_num=1``: the kernel always accepts the root and samples
        one new bonus token from target logits -- functionally a plain decode.
        """
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                topk=self.topk, spec_steps=0, num_verify_tokens=1, device=self.device
            )

        draft_input: EagleDraftInput = batch.spec_info
        bs = batch.seq_lens.shape[0]
        device = self.device

        retrieve_index = torch.arange(bs, dtype=torch.long, device=device).unsqueeze(1)
        retrieve_next_token = torch.full((bs, 1), -1, dtype=torch.long, device=device)
        retrieve_next_sibling = torch.full((bs, 1), -1, dtype=torch.long, device=device)

        attn_backend = self._target_worker.model_runner.attn_backend
        verify_mask = attn_backend.verify_mask
        # Every position in a 1-node tree is visible, so an all-True fill is
        # correct under either layout.
        if verify_mask is not None and verify_mask.fits(bs):
            custom_mask = verify_mask.buffer
            custom_mask.fill_(True)
        else:
            if batch.seq_lens_sum is not None:
                seq_lens_sum = batch.seq_lens_sum
            elif batch.seq_lens_cpu is not None:
                seq_lens_sum = int(batch.seq_lens_cpu.sum())
            else:
                seq_lens_sum = bs * attn_backend.max_context_len
            custom_mask = torch.ones(seq_lens_sum + bs, dtype=torch.bool, device=device)

        positions = batch.seq_lens.to(torch.int64)

        return EagleVerifyInput(
            draft_token=draft_input.bonus_tokens,
            custom_mask=custom_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=0,
            topk=self.topk,
            draft_token_num=1,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def _stub_skipped_draft_extend(
        self, batch: ScheduleBatch, batch_output: GenerationBatchResult
    ) -> None:
        """Fill shape-valid stubs on next_draft_input when draft_extend is skipped.

        ``verify`` already set ``bonus_tokens`` (the only field the next steps=0
        verify reads). The overlap FutureMap still stashes topk_p/topk_index/
        hidden_states, so provide zeroed tensors of the right shape. They are never
        consumed while at steps=0; an upshift to steps>0 would draft from this stale
        state (cold recovery), which is the documented cost of this experimental flag.
        """
        next_draft_input: EagleDraftInput = batch_output.next_draft_input
        bs = batch.seq_lens.shape[0]
        device = self.device
        next_draft_input.topk_p = torch.zeros(
            (bs, self.topk), dtype=torch.float32, device=device
        )
        next_draft_input.topk_index = torch.zeros(
            (bs, self.topk), dtype=torch.int64, device=device
        )
        hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
            self.draft_worker.draft_runner
        )
        if hidden_size is not None:
            next_draft_input.hidden_states = torch.zeros(
                (bs, hidden_size),
                dtype=hidden_dtype,
                device=device,
            )

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(
                num_correct_drafts_per_req, batch_size=batch_size
            )

    def activate_step_by_batch(self, batch_size: int) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.activate_step_by_batch(batch_size)

    # -- Adaptive speculative decoding protocol --

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs=None,
    ) -> SpecRuntimeState:
        """Build a SpecRuntimeState for the given step configuration."""
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)

        with self._override_worker_state(
            speculative_num_steps,
            speculative_num_draft_tokens,
            cuda_graph_bs=cuda_graph_bs,
        ):
            self._draft_worker.init_attention_backend()
            self._draft_worker._capture_cuda_graphs()

            # Build target attention backend and CUDA graph runner
            target_model_runner = self._target_worker.model_runner
            backup_init = target_model_runner.init_new_workspace
            try:
                target_attn_backend = target_model_runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                target_model_runner.init_new_workspace = backup_init

            target_graph_runner = None
            if not check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
                TargetGraphRunnerCls = (
                    NPUGraphRunner if _is_npu else DecodeCudaGraphRunner
                )
                target_graph_before_mem = get_available_gpu_memory(
                    self.device, self.gpu_id
                )
                target_graph_tic = time.perf_counter()
                target_graph_runner = TargetGraphRunnerCls(
                    target_model_runner,
                    attn_backend=target_attn_backend,
                    speculative_num_steps=speculative_num_steps,
                    speculative_num_draft_tokens=speculative_num_draft_tokens,
                )
                target_graph_after_mem = get_available_gpu_memory(
                    self.device, self.gpu_id
                )
                target_graph_time = time.perf_counter() - target_graph_tic
                self._additional_graph_memory_usage["target_verify"] = (
                    self._additional_graph_memory_usage.get("target_verify", 0.0)
                    + target_graph_before_mem
                    - target_graph_after_mem
                )
                self._additional_graph_time_usage["target_verify"] = (
                    self._additional_graph_time_usage.get("target_verify", 0.0)
                    + target_graph_time
                )

            state = SpecRuntimeState(
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                draft_attn_backend=self._draft_worker.draft_attn_backend,
                cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                target_attn_backend=target_attn_backend,
                target_graph_runner=target_graph_runner,
                draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
            )

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        log_info_on_rank0(
            logger,
            f"Built adaptive runtime state steps={speculative_num_steps}: "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB",
        )

        return state

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        """Apply a pre-built runtime state to this worker."""
        if self.speculative_num_steps == state.speculative_num_steps:
            return

        log_info_on_rank0(
            logger,
            "Switch adaptive runtime state: "
            f"steps {self.speculative_num_steps} -> {state.speculative_num_steps}, "
            f"draft_tokens {self.speculative_num_draft_tokens} -> "
            f"{state.speculative_num_draft_tokens}",
        )

        # Top-level
        self.speculative_num_steps = state.speculative_num_steps
        self.speculative_num_draft_tokens = state.speculative_num_draft_tokens

        # Draft side
        dw = self._draft_worker
        dw.speculative_num_steps = state.speculative_num_steps
        dw.speculative_num_draft_tokens = state.speculative_num_draft_tokens
        dw.draft_attn_backend = state.draft_attn_backend
        dw.draft_runner.draft_attn_backend = state.draft_attn_backend
        dw.cuda_graph_runner = state.cuda_graph_runner
        dw.draft_extend_attn_backend = state.draft_extend_attn_backend
        # Keep the runner's attn_backend in step with the active draft-extend
        # backend (the draft-extend forward reads draft_runner.attn_backend);
        # mirrors init_attention_backend. When None, the runner keeps its
        # initialized backend (consistent across step configs).
        if state.draft_extend_attn_backend is not None:
            dw.draft_runner.attn_backend = state.draft_extend_attn_backend
        dw.cuda_graph_runner_for_draft_extend = state.cuda_graph_runner_for_draft_extend
        dw._rebuild_topk1_chain_buffers()

        # Target side
        self._target_worker.model_runner.attn_backend = state.target_attn_backend
        self._target_worker.model_runner.decode_cuda_graph_runner = (
            state.target_graph_runner
        )

        # Sync server_args
        get_context().override(
            "adaptive_spec.restore",
            speculative_num_steps=state.speculative_num_steps,
            speculative_num_draft_tokens=state.speculative_num_draft_tokens,
        )

    @contextlib.contextmanager
    def _override_worker_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ):
        """Temporarily override server_args and worker attributes for graph capture."""
        dw = self._draft_worker
        backup = (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            dw.speculative_num_steps,
            dw.speculative_num_draft_tokens,
            dw.draft_attn_backend,
            dw.draft_extend_attn_backend,
            dw.draft_runner.draft_attn_backend,
            dw.draft_runner.attn_backend,
            dw.cuda_graph_runner,
            dw.cuda_graph_runner_for_draft_extend,
            get_spec().speculative_num_steps,
            get_spec().speculative_num_draft_tokens,
            get_exec().graph.cuda_graph_bs_decode,
            get_exec().graph.disable_cuda_graph,
        )

        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        dw.speculative_num_steps = speculative_num_steps
        dw.speculative_num_draft_tokens = speculative_num_draft_tokens
        get_context().override(
            "adaptive_spec.capture_override",
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        if cuda_graph_bs is not None:
            # BS-aware adaptive spec may prune cuda_graph_bs to an empty list
            # for steps that no BS range uses (e.g. step=1). Disable graph
            # capture for those steps; restore in finally so subsequent steps
            # are not affected.
            get_context().override(
                "adaptive_spec.capture_override",
                cuda_graph_bs_decode=cuda_graph_bs,
                **({"disable_cuda_graph": True} if not cuda_graph_bs else {}),
            )
        dw._rebuild_topk1_chain_buffers()

        try:
            yield
        finally:
            (
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                dw.speculative_num_steps,
                dw.speculative_num_draft_tokens,
                dw.draft_attn_backend,
                dw.draft_extend_attn_backend,
                dw.draft_runner.draft_attn_backend,
                dw.draft_runner.attn_backend,
                dw.cuda_graph_runner,
                dw.cuda_graph_runner_for_draft_extend,
            ) = backup[:10]
            get_context().override(
                "adaptive_spec.capture_restore",
                speculative_num_steps=backup[10],
                speculative_num_draft_tokens=backup[11],
                cuda_graph_bs_decode=backup[12],
                disable_cuda_graph=backup[13],
            )
            dw._rebuild_topk1_chain_buffers()

    def verify(self, batch: ScheduleBatch, grammar_barrier=None):
        return run_eagle_verify(
            batch,
            target_worker=self.target_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            plan_stream=self.plan_stream,
            plan_stream_ctx=self.plan_stream_ctx,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,
            grammar_barrier=grammar_barrier,
        )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.ps.tp_rank]
        )
        success, message = (
            self.draft_worker.draft_runner.weight_updater.update_weights_from_tensor(
                named_tensors=named_tensors,
                load_format=recv_req.load_format,
            )
        )
        if not success:
            return success, message

        success, message = (
            self.target_worker.model_runner.weight_updater.update_weights_from_tensor(
                named_tensors=named_tensors,
                load_format=recv_req.load_format,
            )
        )
        return success, message
