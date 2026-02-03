import torch
import torch.distributed as dist
from typing import Callable, List, Tuple, Optional, Union

# noinspection PyUnresolvedReferences
import mori
import mori.ops
import mori.shmem
# noinspection PyUnresolvedReferences
from .utils import EventOverlap

# Type aliases for compatibility
Config = mori.ops.EpDispatchCombineConfig  # Use mori's config as Config


class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports:
        - high-throughput intranode all-to-all (dispatch and combine, using NVLink)
        - high-throughput internode all-to-all (dispatch and combine, using RDMA and NVLink)
        - low-latency all-to-all (dispatch and combine, using RDMA)

    Attributes:
        num_sms: the SMs used in high-throughput kernels.
        rank: the local rank number.
        group_size: the number of ranks in the group.
        group: the communication group.
        num_nvl_bytes: the buffer size for intranode NVLink communication.
        num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
        runtime: the C++ runtime.
    """

    num_sms: int = 20
    MAX_GPU_PER_NODE: int = 8

    def __init__(self,
                 group: Optional[dist.ProcessGroup],
                 num_nvl_bytes: int = 0,
                 num_rdma_bytes: int = 0,
                 low_latency_mode: bool = False,
                 num_qps_per_rank: int = 24,
                 allow_nvlink_for_low_latency_mode: bool = True,
                 allow_mnnvl: bool = False,
                 use_fabric: bool = False,
                 explicitly_destroy: bool = False,
                 enable_shrink: bool = False,
                 comm: Optional["mpi4py.MPI.Comm"] = None) -> None:  # noqa: F821
        """
        Initialize the communication buffer.

        Arguments:
            group: the communication group.
            num_nvl_bytes: the buffer size for intranode NVLink communication.
            num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
            low_latency_mode: whether to enable low-latency mode.
            num_qps_per_rank: the number of QPs for RDMA, the low-latency mode requires that this number equals
                to the number of local experts.
            allow_nvlink_for_low_latency_mode: whether allow NVLink traffic for low-latency mode, you should notice
                this is somehow incompatible with the hook-based overlapping.
                Warning: PCIe connections may lead to errors due to memory ordering issues,
                please make sure all connections are via NVLink.
            allow_mnnvl: whether to allow MNNVL
            use_fabric: whether to use fabric API for memory buffers.
            enable_shrink: whether to enable shrink mode. The enable mode allocates a mask buffer to support masking ranks dynamically.
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
                Note: Releasing resources in the destructor may cause Python's exception handling process to hang.
            comm: the `mpi4py.MPI.Comm` communicator to use in case the group parameter is absent.
        """
        # check_nvlink_connections(group)

        # Initialize basic attributes
        if group is not None:
            self.rank = group.rank()
            self.group = group
            self.group_size = group.size()

            def all_gather_object(obj):
                object_list = [None] * self.group_size
                dist.all_gather_object(object_list, obj, group)
                return object_list
        elif comm is not None:
            self.rank = comm.Get_rank()
            self.group = comm
            self.group_size = comm.Get_size()

            def all_gather_object(obj):
                return comm.allgather(obj)
        else:
            raise ValueError("Either 'group' or 'comm' must be provided.")
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.num_qps_per_rank = num_qps_per_rank
        self.low_latency_mode = low_latency_mode
        self.explicitly_destroy = explicitly_destroy
        self.enable_shrink = enable_shrink
        self.multi_node = self.group_size > Buffer.MAX_GPU_PER_NODE

        # Register default process group for mori shmem (required)
        assert dist.is_initialized(), "torch.distributed must be initialized before shmem init"
        world_group = dist.group.WORLD
        if world_group is not None:
            torch._C._distributed_c10d._register_process_group("default", world_group)
        # Initialize SHMEM
        mori.shmem.shmem_torch_process_group_init("default")

        # Create mori config and op using saved parameters from get_low_latency_rdma_size_hint
        self.mori_op = None
        self.mori_config = None
        self._mori_config_key = None

    def _ensure_mori_op(self, num_tokens: int, hidden: int, num_experts: int, num_topk: int, kernel_type, block_num: int,
                        warp_num_per_block: int, rdma_block_num: int) -> None:
        config_key = (num_tokens, hidden, num_experts, num_topk, kernel_type, block_num, warp_num_per_block, rdma_block_num)
        if self.mori_op is not None and self._mori_config_key == config_key:
            return

        num_local_experts = num_experts // self.group_size
        gpu_per_node = min(self.group_size, Buffer.MAX_GPU_PER_NODE)

        self.mori_config = mori.ops.EpDispatchCombineConfig(
            data_type=torch.bfloat16,
            rank=self.rank,
            world_size=self.group_size,
            hidden_dim=hidden,
            scale_dim=0,
            scale_type_size=0,
            max_token_type_size=2,
            max_num_inp_token_per_rank=num_tokens,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=num_topk,
            warp_num_per_block=warp_num_per_block,
            block_num=block_num,
            use_external_inp_buf=True,
            kernel_type=kernel_type,
            gpu_per_node=gpu_per_node,
            rdma_block_num=rdma_block_num,
            num_qp_per_pe=self.num_qps_per_rank,
        )

        self.mori_op = mori.ops.EpDispatchCombineOp(self.mori_config)
        self._mori_config_key = config_key

    def destroy(self):
        """
        Destroy the runtime and release resources.
        """
        assert self.explicitly_destroy, '`explicitly_destroy` flag must be set'

        # Clean up mori resources
        self.mori_op = None
        self.mori_config = None

    @staticmethod
    def is_sm90_compiled():
        """Check if SM90 is compiled"""
        return False  # ROCm doesn't use SM90

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """
        Set the number of SMs to use in high-throughput kernels.

        Arguments:
            new_num_sms: the new number to be set.
        """

        # assert new_num_sms % 2 == 0, 'The SM count must be even'
        Buffer.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event: the captured event.
        """
        return EventOverlap()

    @staticmethod
    def get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int) -> int:
        """
        Get a minimum size requirement for the RDMA buffer. The size calculation will be done with BF16.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_ranks: the number of EP group ranks.
            num_experts: the number of all experts.

        Returns:
            size: the RDMA buffer size recommended.
        """
        # For mori, we don't need to return a size hint
        return 0

        # Conservative estimate for mori
        return (num_max_dispatch_tokens_per_rank * hidden * num_ranks * 4) * 4  # 4 bytes per element

    def get_comm_stream(self) -> torch.Stream:
        """
        Get the communication stream.

        Returns:
            stream: the communication stream.
        """
        # Return the current CUDA stream for mori
        return torch.cuda.current_stream()

    def get_local_buffer_tensor(self,
                                dtype: torch.dtype,
                                size: Optional[torch.Size] = None,
                                offset: int = 0,
                                use_rdma_buffer: bool = False) -> torch.Tensor:
        """
        Get the raw buffer (slice supported) as a PyTorch tensor.

        Argument:
            dtype: the data type (PyTorch `dtype`) for the tensor.
            size: the slice size (by elements) to get from the buffer.
            offset: the offset of the beginning element.
            use_rdma_buffer: whether to return the RDMA buffer.
        """
        # Not implemented for mori
        raise NotImplementedError("get_local_buffer_tensor not implemented for mori backend")

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """
        # Not implemented for mori backend
        raise NotImplementedError("get_dispatch_config not implemented for mori backend")

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """
        # Not implemented for mori backend
        raise NotImplementedError("get_combine_config not implemented for mori backend")

    # noinspection PyTypeChecker
    def get_dispatch_layout(self, topk_idx: torch.Tensor, num_experts: int,
                            previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                            allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap]:
        """
        Calculate the layout required for later communication.

        Arguments:
            topk_idx: `[num_tokens, num_topk]`, dtype must be `deep_ep.topk_idx_t` (typically `torch.int64`), the expert
                indices selected by each token, `-1` means no selections.
            num_experts: the number of experts.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Not implemented for mori - placeholder
        raise NotImplementedError("get_dispatch_layout not implemented for mori backend")

    # noinspection PyTypeChecker
    def dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 handle: Optional[Tuple] = None,
                 num_tokens_per_rank: Optional[torch.Tensor] = None, num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                 is_token_in_rank: Optional[torch.Tensor] = None, num_tokens_per_expert: Optional[torch.Tensor] = None,
                 topk_idx: Optional[torch.Tensor] = None, topk_weights: Optional[torch.Tensor] = None,
                 expert_alignment: int = 1, num_worst_tokens: int = 0,
                 config: Optional[Config] = None,
                 previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                 allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],
                  Optional[torch.Tensor], List[int], Tuple, EventOverlap]:
        """
        Dispatch tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
                and type must be `torch.bfloat16`; for the second type, the first element of the tuple must be shaped as
                `[num_tokens, hidden]` with type `torch.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                 (requiring divisible) with type `torch.float`.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            topk_idx: `[num_tokens, num_topk]` with `deep_ep.topk_idx_t` (typically `torch.int64`), the expert indices
                selected by each token, `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_worst_tokens: the worst number of tokens to receive, if specified, there will be no CPU sync, and it
                will be CUDA-graph compatible. Please also notice that this flag is for intranode only.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
                received token count.
            recv_topk_idx: received expert indices.
            recv_topk_weights: received expert weights.
            num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`. If `num_worst_tokens` is specified, the list
                will be empty.
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Not implemented for mori - use low_latency_dispatch instead
        raise NotImplementedError("dispatch not implemented for mori backend, use low_latency_dispatch instead")

        # Placeholder to prevent syntax error
        # Default config
        config = self.get_dispatch_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_dispatch(x, handle, num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank,
                                           num_tokens_per_expert, topk_idx, topk_weights, expert_alignment, num_worst_tokens, config,
                                           previous_event, async_finish, allocate_on_comm_stream)

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head = handle
            num_recv_tokens = recv_src_idx.size(0)
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, event = self.runtime.intranode_dispatch(
                x, x_scales, None, None, None, is_token_in_rank, None, num_recv_tokens, rank_prefix_matrix, channel_prefix_matrix,
                expert_alignment, num_worst_tokens, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            return (recv_x, recv_x_scales) if x_scales is not None else recv_x, None, None, None, None, EventOverlap(event)
        else:
            assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None
            recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, send_head, event = \
                self.runtime.intranode_dispatch(x, x_scales, topk_idx, topk_weights,
                                                num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, 0, None, None,
                                                expert_alignment, num_worst_tokens, config,
                                                getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            handle = (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)
            return (
                recv_x, recv_x_scales
            ) if x_scales is not None else recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, EventOverlap(
                event)

    # noinspection PyTypeChecker
    def combine(self, x: torch.Tensor, handle: Tuple,
                topk_weights: Optional[torch.Tensor] = None,
                bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                config: Optional[Config] = None,
                previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to its original ranks.
            bias: 0, 1 or 2 `[num_tokens, hidden]` with `torch.bfloat16` final bias to the output.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: the reduced token from its dispatched ranks.
            recv_topk_weights: the reduced top-k weights from its dispatch ranks.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Not implemented for mori - use low_latency_combine instead
        raise NotImplementedError("combine not implemented for mori backend, use low_latency_combine instead")

        # Placeholder to prevent syntax error
        # Default config
        config = self.get_combine_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_combine(x, handle, topk_weights, bias, config, previous_event, async_finish, allocate_on_comm_stream)

        # NOTES: the second `_` is for the sending side, so we should use the third one
        rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle
        bias_0, bias_1 = Buffer._unpack_bias(bias)

        # Launch the kernel
        recv_x, recv_topk_weights, event = self.runtime.intranode_combine(x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix,
                                                                          channel_prefix_matrix, send_head, config,
                                                                          getattr(previous_event, 'event',
                                                                                  None), async_finish, allocate_on_comm_stream)
        return recv_x, recv_topk_weights, EventOverlap(event)

    # noinspection PyTypeChecker
    def internode_dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                           handle: Optional[Tuple] = None,
                           num_tokens_per_rank: Optional[torch.Tensor] = None, num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                           is_token_in_rank: Optional[torch.Tensor] = None, num_tokens_per_expert: Optional[torch.Tensor] = None,
                           topk_idx: Optional[torch.Tensor] = None, topk_weights: Optional[torch.Tensor] = None, expert_alignment: int = 1,
                           num_worst_tokens: int = 0, config: Optional[Config] = None,
                           previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                           allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],
            Optional[torch.Tensor], List[int], Tuple, EventOverlap]:
        """
        Internode dispatch implementation, for more details, please refer to the `dispatch` docs.
        Normally, you should not directly call this function.
        """
        # Not implemented for mori
        raise NotImplementedError("internode_dispatch not implemented for mori backend")

        # Placeholder
        assert config is not None

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            is_token_in_rank, \
                rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, \
                recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
                recv_src_meta, send_rdma_head, send_nvl_head = handle
            num_recv_tokens = recv_src_meta.size(0)
            num_rdma_recv_tokens = send_nvl_head.size(0)
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, _, _, _, _, event = self.runtime.internode_dispatch(
                x, x_scales, topk_idx, topk_weights, None, None, is_token_in_rank, None, num_recv_tokens, num_rdma_recv_tokens,
                rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
                expert_alignment, num_worst_tokens, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            return (recv_x, recv_x_scales) if x_scales is not None else recv_x, None, None, None, None, EventOverlap(event)
        else:
            assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None
            recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, \
                rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, \
                recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, \
                recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
                recv_src_meta, send_rdma_head, send_nvl_head, event = self.runtime.internode_dispatch(
                x, x_scales, topk_idx, topk_weights,
                num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank, num_tokens_per_expert,
                0, 0, None, None, None, None,
                expert_alignment, num_worst_tokens, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            handle = (is_token_in_rank, rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, recv_rdma_channel_prefix_matrix,
                      recv_rdma_rank_prefix_sum, recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, recv_src_meta, send_rdma_head,
                      send_nvl_head)
            return (
                recv_x, recv_x_scales
            ) if x_scales is not None else recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, EventOverlap(
                event)

    # noinspection PyTypeChecker
    def internode_combine(self, x: torch.Tensor, handle: Union[tuple, list],
                          topk_weights: Optional[torch.Tensor] = None,
                          bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                          config: Optional[Config] = None,
                          previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                          allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Internode combine implementation, for more details, please refer to the `combine` docs.
        Normally, you should not directly call this function.
        """
        # Not implemented for mori
        raise NotImplementedError("internode_combine not implemented for mori backend")

        # Placeholder
        assert config is not None

        # Unpack handle and bias
        is_combined_token_in_rank, \
            _, _, \
            rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix, gbl_rank_prefix_sum, \
            src_meta, send_rdma_head, send_nvl_head = handle
        bias_0, bias_1 = Buffer._unpack_bias(bias)

        # Launch the kernel
        combined_x, combined_topk_weights, event = self.runtime.internode_combine(x, topk_weights, bias_0, bias_1, src_meta,
                                                                                  is_combined_token_in_rank, rdma_channel_prefix_matrix,
                                                                                  rdma_rank_prefix_sum, gbl_channel_prefix_matrix,
                                                                                  send_rdma_head, send_nvl_head, config,
                                                                                  getattr(previous_event, 'event',
                                                                                          None), async_finish, allocate_on_comm_stream)
        return combined_x, combined_topk_weights, EventOverlap(event)

    def clean_low_latency_buffer(self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int) -> None:
        """
        As low-latency kernels require part of the buffer to be zero-initialized, so it is vital to clean the buffer
            if the buffer is dirty at some time.
        For example, after running the normal dispatch/combine, you must run this function before executing any
            low-latency kernel.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_experts: the number of all experts.
        """
        # For mori, nothing to clean
        pass

    # noinspection PyTypeChecker
    def low_latency_dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor,
                             num_max_dispatch_tokens_per_rank: int, num_experts: int,
                             cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                             dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
                             use_fp8: bool = True, round_scale: bool = False, use_ue8m0: bool = False,
                             async_finish: bool = False, return_recv_hook: bool = False,
                             topk_weights: Optional[torch.Tensor] = None) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable]:
        """
        Low-latency dispatch using mori backend.
        Note: mori must be built with ENABLE_STANDARD_MOE_ADAPT=ON for this path.
        """
        # Mori backend only supports FP16/BF16 synchronous path
        unsupported = []
        if use_fp8:
            unsupported.append('use_fp8')
        if round_scale:
            unsupported.append('round_scale')
        if use_ue8m0:
            unsupported.append('use_ue8m0')
        if async_finish:
            unsupported.append('async_finish')
        if return_recv_hook:
            unsupported.append('return_recv_hook')
        if unsupported:
            unsupported_str = ', '.join(unsupported)
            raise NotImplementedError(f'Mori backend does not support: {unsupported_str}. '
                                      'Only FP16/BF16 synchronous dispatch is supported.')

        num_tokens, hidden = x.shape
        num_topk = topk_idx.shape[1]
        if not self.multi_node:
            kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
            block_num, warp_num_per_block = 64, 16
            rdma_block_num = 0
        else:
            kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1LL
            block_num, warp_num_per_block = 64, 8
            rdma_block_num = 32

        self._ensure_mori_op(
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
            num_topk,
            kernel_type,
            block_num,
            warp_num_per_block,
            rdma_block_num,
        )

        convert_stand_alone = True
        if convert_stand_alone:
            # Call mori dispatch (topk_weights can be None)
            dispatch_out_x, dispatch_out_topk_weights, _, dispatch_out_topk_idx, dispatch_out_count = \
                self.mori_op.dispatch(x, topk_weights, None, topk_idx)
            packed_recv_x, packed_recv_count, packed_recv_src_info, packed_recv_layout_range = \
                self.mori_op.convert_dispatch_output(
                    dispatch_out_x, dispatch_out_topk_idx, 80, 16
                )
        else:
            packed_recv_x, packed_recv_count, packed_recv_src_info, packed_recv_layout_range = \
                self.mori_op.dispatch_standard_moe(x, topk_weights, None, topk_idx, 64, 16)

        # Create handle for combine (store necessary info)
        handle = (packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, x.size(1), num_experts)
        tensors_to_record = (x, topk_idx, packed_recv_x, packed_recv_count, packed_recv_src_info, packed_recv_layout_range)

        # Create event and hook
        event = None

        def hook():
            return None  # Dummy hook for compatibility

        return packed_recv_x, packed_recv_count, handle, EventOverlap(event, tensors_to_record if async_finish else None), hook

    # noinspection PyTypeChecker
    def low_latency_combine(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                            handle: tuple, use_logfmt: bool = False, zero_copy: bool = False, async_finish: bool = False,
                            return_recv_hook: bool = False, out: Optional[torch.Tensor] = None,
                            combine_wait_recv_cost_stats: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, EventOverlap, Callable]:
        """
        Low-latency combine using mori backend.
        Note: mori must be built with ENABLE_STANDARD_MOE_ADAPT=ON for this path.
        """
        unsupported = []
        if use_logfmt:
            unsupported.append('use_logfmt')
        if zero_copy:
            unsupported.append('zero_copy')
        if async_finish:
            unsupported.append('async_finish')
        if return_recv_hook:
            unsupported.append('return_recv_hook')
        if out is not None:
            unsupported.append('out')
        if combine_wait_recv_cost_stats is not None:
            unsupported.append('combine_wait_recv_cost_stats')
        if unsupported:
            unsupported_str = ', '.join(unsupported)
            raise NotImplementedError(f'Mori backend does not support: {unsupported_str}. '
                                      'Only synchronous combine without extra outputs is supported.')

        # Convert packed 3D input into mori combine input (2D)
        packed_recv_src_info = handle[0]
        packed_recv_layout_range = handle[1]
        if not self.multi_node:
            combine_block_num, combine_warp_per_block = 64, 4
        else:
            combine_block_num, combine_warp_per_block = 64, 8

        convert_stand_alone = True
        if convert_stand_alone:
            combine_input = self.mori_op.convert_combine_input(x, packed_recv_src_info, packed_recv_layout_range, 80, 16)
            combined_x, _ = self.mori_op.combine(combine_input, None, topk_idx, combine_block_num, combine_warp_per_block)
        else:
            combined_x, _ = self.mori_op.combine_standard_moe(x, None, topk_idx, 64, 8)

        # Create event and hook
        def hook():
            return None  # Dummy hook for compatibility

        return combined_x, None, hook

    # noinspection PyTypeChecker
    def low_latency_dispatch_rocm(self, x: torch.Tensor, topk_idx: torch.Tensor,
                             num_max_dispatch_tokens_per_rank: int, num_experts: int,
                             cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                             dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
                             use_fp8: bool = True, round_scale: bool = False, use_ue8m0: bool = False,
                             async_finish: bool = False, return_recv_hook: bool = False,
                             topk_weights: Optional[torch.Tensor] = None) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable]:
        """
        A ROCm-adapted low-latency dispatch variant for fused MoE (AITER).
        This function mirrors `low_latency_dispatch` but adapts inputs/outputs to match
        AITER fused MoE expectations.

        Differences vs `low_latency_dispatch`:
            - Returns: explicitly returns `recv_topk_idx` and `recv_topk_weights` (the base API does not return these).

        Note:
            - Requires mori to be compiled with `ENABLE_STANDARD_MOE_ADAPT=OFF`.

        See also:
            - `low_latency_dispatch` for the base CUDA variant and full documentation.
        """
        # Mori backend only supports FP16/BF16 synchronous path
        unsupported = []
        if use_fp8:
            unsupported.append('use_fp8')
        if round_scale:
            unsupported.append('round_scale')
        if use_ue8m0:
            unsupported.append('use_ue8m0')
        if async_finish:
            unsupported.append('async_finish')
        if return_recv_hook:
            unsupported.append('return_recv_hook')
        if unsupported:
            unsupported_str = ', '.join(unsupported)
            raise NotImplementedError(f'Mori backend does not support: {unsupported_str}. '
                                      'Only FP16/BF16 synchronous dispatch is supported.')

        num_tokens, hidden = x.shape
        num_topk = topk_idx.shape[1]
        if not self.multi_node:
            kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
            block_num, warp_num_per_block = 64, 16
            rdma_block_num = 0
        else:
            kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1LL
            block_num, warp_num_per_block = 64, 8
            rdma_block_num = 32

        self._ensure_mori_op(
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
            num_topk,
            kernel_type,
            block_num,
            warp_num_per_block,
            rdma_block_num,
        )

        # dispatch_out_x, dispatch_out_topk_weights, _, dispatch_out_topk_idx, dispatch_out_count = \
        #     self.mori_op.dispatch(x, topk_weights, None, topk_idx)
        packed_recv_x, packed_recv_topk_weights, _, packed_recv_topk_idx, packed_recv_count = \
            self.mori_op.dispatch(x, topk_weights, None, topk_idx)

        # Create handle for combine (store necessary info)
        handle = (num_max_dispatch_tokens_per_rank, x.size(1), num_experts)
        tensors_to_record = (x, topk_idx, packed_recv_x, packed_recv_count, packed_recv_topk_idx, packed_recv_topk_weights)

        # Create event and hook
        event = None

        def hook():
            return None  # Dummy hook for compatibility

        return packed_recv_x, packed_recv_topk_idx, packed_recv_topk_weights, packed_recv_count, handle, EventOverlap(
            event, tensors_to_record if async_finish else None), hook

    # noinspection PyTypeChecker
    def low_latency_combine_rocm(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                            handle: tuple, use_logfmt: bool = False, zero_copy: bool = False, async_finish: bool = False,
                            return_recv_hook: bool = False, out: Optional[torch.Tensor] = None,
                            combine_wait_recv_cost_stats: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, EventOverlap, Callable]:
        """
        A ROCm-adapted low-latency combine variant for fused MoE (AITER).

        Differences vs `low_latency_combine`:
            - `topk_weights` is Optional and can be omitted (passed as None); the base API requires it.

        Note:
            - Requires mori to be compiled with `ENABLE_STANDARD_MOE_ADAPT=OFF`.

        See also:
            - `low_latency_combine` for the base CUDA variant and full documentation.
        """
        unsupported = []
        if use_logfmt:
            unsupported.append('use_logfmt')
        if zero_copy and self.multi_node:
            unsupported.append('zero_copy')
        if async_finish:
            unsupported.append('async_finish')
        if return_recv_hook:
            unsupported.append('return_recv_hook')
        if out is not None:
            unsupported.append('out')
        if combine_wait_recv_cost_stats is not None:
            unsupported.append('combine_wait_recv_cost_stats')
        if unsupported:
            unsupported_str = ', '.join(unsupported)
            raise NotImplementedError(f'Mori backend does not support: {unsupported_str}. '
                                      'Only synchronous combine without extra outputs is supported.')

        if not self.multi_node:
            if zero_copy:
                combine_block_num, combine_warp_per_block = 64, 4
            else:
                combine_block_num, combine_warp_per_block = 64, 16
        else:
            combine_block_num, combine_warp_per_block = 64, 8
        use_external_inp_buf = 0 if zero_copy else 1
        combined_x, _ = self.mori_op.combine(x, None, topk_idx, combine_block_num, combine_warp_per_block, use_external_inp_buf)

        # Create event and hook
        def hook():
            return None  # Dummy hook for compatibility

        return combined_x, None, hook

    def low_latency_update_mask_buffer(self, rank_to_mask: int, mask: bool = False):
        """
        Mask (unmask) a rank during communication (dispatch, combine, and clean)

        Arguments:
            rank: the rank to mask (unmask).
            mask: if True, will mask the rank (do not recvfrom/sendto the rank), otherwise will unmask the rank.

        Not implemented for mori backend.
        """
        raise NotImplementedError("low_latency_update_mask_buffer not implemented for mori backend")

    def low_latency_query_mask_buffer(self, mask_status: torch.Tensor):
        """
        Query the mask status of all ranks

        Arguments:
            mask_status: `[num_ranks]` with `torch.int`, the mask status of each rank. `1` means mask and `0` means unmasked.

        Not implemented for mori backend.
        """
        raise NotImplementedError("low_latency_query_mask_buffer not implemented for mori backend")

    def low_latency_clean_mask_buffer(self):
        """
        Clean the mask buffer.
        Not implemented for mori backend.
        """
        raise NotImplementedError("low_latency_clean_mask_buffer not implemented for mori backend")

    def get_next_low_latency_combine_buffer(self, handle: object):
        """
        Get the raw registered RDMA buffer tensor for next low-latency combine, so that the next combine kernel can skip the copying.

        Arguments:
            handle: the communication handle given by the `dispatch` function.

        Returns:
            buffer: the raw RDMA low-latency buffer as a BF16 PyTorch tensor with shape
                `[num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden]`, you should fill this buffer
                by yourself.
        """
        return self.mori_op.get_registered_combine_input_buffer(self.mori_config.data_type)
