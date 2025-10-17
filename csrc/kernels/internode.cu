#include "configs.cuh"
#include "buffer.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"
#include "ibgda_device.cuh"

namespace deep_ep {

namespace internode {

extern nvshmem_team_t cpu_rdma_team;

struct SourceMeta {
    int src_rdma_rank, is_token_in_nvl_rank_bits;

    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "Invalid number of maximum NVL peers");

    __forceinline__ SourceMeta() = default;

    // TODO: faster encoding
    __device__ __forceinline__ SourceMeta(int rdma_rank, const bool* is_token_in_nvl_ranks) {
        src_rdma_rank = rdma_rank;
        is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];
        #pragma unroll
        for (int i = 1; i < NUM_MAX_NVL_PEERS; ++ i)
            is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;
    }

    __device__ __forceinline__ bool is_token_in_nvl_rank(int nvl_rank) const {
        return (is_token_in_nvl_rank_bits >> nvl_rank) & 1;
    }
};

EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

int get_source_meta_bytes() {
    return sizeof(SourceMeta);
}

__host__ __device__ __forceinline__
int get_num_bytes_per_token(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights) {
    return static_cast<int>(align(hidden_int4 * sizeof(int4) + sizeof(SourceMeta) + num_scales * sizeof(float) + num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float), sizeof(int4)));
}

__host__ __device__ __forceinline__
std::pair<int, int> get_rdma_clean_meta(int hidden_int4, int num_scales, int num_topk_idx,
                                        int num_topk_weights, int num_rdma_ranks, int num_rdma_recv_buffer_tokens,
                                        int num_channels) {
    // Return `int32_t` offset and count to clean
    return {
        (get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_rdma_recv_buffer_tokens * num_rdma_ranks * 2 * num_channels) / sizeof(int),
        (NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_channels
    };
}

__host__ __device__ __forceinline__
std::pair<int, int> get_nvl_clean_meta(int hidden_int4, int num_scales, int num_topk_idx,
                                       int num_topk_weights, int num_rdma_ranks, int num_nvl_ranks,
                                       int num_nvl_recv_buffer_tokens, int num_channels, bool is_dispatch) {
    // Return `int32_t` offset and to clean
    EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");
   
    return {
        (num_nvl_recv_buffer_tokens * get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_nvl_ranks * num_channels) / sizeof(int),
        num_nvl_ranks * (2 * num_rdma_ranks + 2) * num_channels,
    };
}

template <bool kLowLatencyMode>
__forceinline__ __device__ int translate_dst_rdma_rank(const int dst_rdma_rank, const int nvl_rank) {
    return kLowLatencyMode ? (dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank) : dst_rdma_rank;
}

template <bool kLowLatencyMode>
__forceinline__ __device__ void nvshmem_sync_with_same_gpu_idx(const nvshmem_team_t& rdma_team) {
    kLowLatencyMode ? void(nvshmem_sync(rdma_team)) : nvshmem_sync_all();
}

template <bool kLowLatencyMode, int kNumRDMARanks>
__global__ void
notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                const int* num_tokens_per_rdma_rank, int* moe_recv_rdma_counter_mapped,
                const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                const bool* is_token_in_rank, int num_tokens, int num_channels, int expert_alignment,
                const int rdma_clean_offset, const int rdma_num_int_clean,
                const int nvl_clean_offset, const int nvl_num_int_clean,
                int* rdma_channel_prefix_matrix, int* recv_rdma_rank_prefix_sum,
                int* gbl_channel_prefix_matrix, int* recv_gbl_rank_prefix_sum,
                void* rdma_buffer_ptr,
                void** buffer_ptrs, int** barrier_signal_ptrs, int rank,
                const nvshmem_team_t rdma_team) {// rdma_buffer_ptr 是的nvshmem_align开出来的， buffer_ptrs 是buffer_ptrs_gpu
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;

    auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    auto num_rdma_experts = num_experts / kNumRDMARanks, num_nvl_experts = num_rdma_experts / NUM_MAX_NVL_PEERS;

    if (sm_id == 0) {
        // Communication with others
        // Global barrier: the first warp does intra-node sync, the second warp does internode sync
        EP_DEVICE_ASSERT(num_warps > 1);
        EP_DEVICE_ASSERT(kNumRDMARanks <= num_threads);
        if (thread_id == 32)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);//同号卡barrier
        barrier_block<NUM_MAX_NVL_PEERS, true>(barrier_signal_ptrs, nvl_rank);

        // Send numbers of tokens per rank/expert to RDMA ranks
        auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
        auto rdma_recv_num_tokens_mixed = SymBuffer<int>(rdma_buffer_ptr, NUM_MAX_NVL_PEERS + num_rdma_experts + 1, kNumRDMARanks); // 给传入的指针加偏移

        // Clean up for later data dispatch
        EP_DEVICE_ASSERT(rdma_recv_num_tokens_mixed.total_bytes <= rdma_clean_offset * sizeof(int));
        #pragma unroll
        for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
            rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;

        // Copy to send buffer get_dispatch_layout算出来的那些？
        #pragma unroll
        for (int i = thread_id; i < num_ranks; i += num_threads)
            rdma_recv_num_tokens_mixed.send_buffer(i / NUM_MAX_NVL_PEERS)[i % NUM_MAX_NVL_PEERS] = num_tokens_per_rank[i];
        #pragma unroll
        for (int i = thread_id; i < num_experts; i += num_threads)
            rdma_recv_num_tokens_mixed.send_buffer(i / num_rdma_experts)[NUM_MAX_NVL_PEERS + i % num_rdma_experts] = num_tokens_per_expert[i]; // num_rdma_experts 才是节点的EP数
        if (thread_id < kNumRDMARanks)
            rdma_recv_num_tokens_mixed.send_buffer(thread_id)[NUM_MAX_NVL_PEERS + num_rdma_experts] = num_tokens_per_rdma_rank[thread_id];
        __syncthreads();

        // Issue send
        // TODO: more light fence or barrier or signaling
        // TODO: overlap EP barrier and NVL cleaning
        for (int i = warp_id; i < kNumRDMARanks; i += num_warps) {
            if (i != rdma_rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank)),
                                                reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.send_buffer(i)),
                                                (NUM_MAX_NVL_PEERS + num_rdma_experts + 1) * sizeof(int),
                                                translate_dst_rdma_rank<kLowLatencyMode>(i, nvl_rank), 0, lane_id, 0);
            } else { 
                UNROLLED_WARP_COPY(1, lane_id, NUM_MAX_NVL_PEERS + num_rdma_experts + 1, 
                                    rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank), 
                                    rdma_recv_num_tokens_mixed.send_buffer(i), 
                                    ld_volatile_global, st_na_global);
            }
        }
        __syncthreads();

        // Wait previous operations to be finished
        if (thread_id < kNumRDMARanks and thread_id != rdma_rank)
            nvshmemi_ibgda_quiet(translate_dst_rdma_rank<kLowLatencyMode>(thread_id, nvl_rank), 0);
        __syncthreads();

        // Barrier
        if (thread_id == 0)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        __syncthreads();

        // NVL buffers
        auto nvl_send_buffer = thread_id < NUM_MAX_NVL_PEERS ? buffer_ptrs[thread_id] : nullptr;
        auto nvl_recv_buffer = buffer_ptrs[nvl_rank];
        auto nvl_reduced_num_tokens_per_expert = Buffer<int>(nvl_recv_buffer, num_rdma_experts).advance_also(nvl_send_buffer); // Buffer的类成员ptr不变，nvl_recv_buffer往后偏移num_rdma_experts，另外nvl_send_buffer也偏移这么多
        auto nvl_send_num_tokens_per_rank = AsymBuffer<int>(nvl_send_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
        auto nvl_send_num_tokens_per_expert = AsymBuffer<int>(nvl_send_buffer, num_nvl_experts, NUM_MAX_NVL_PEERS);
        auto nvl_recv_num_tokens_per_rank = AsymBuffer<int>(nvl_recv_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
        auto nvl_recv_num_tokens_per_expert = AsymBuffer<int>(nvl_recv_buffer, num_nvl_experts, NUM_MAX_NVL_PEERS);

        // Clean up for later data dispatch
        auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);
        EP_DEVICE_ASSERT(nvl_reduced_num_tokens_per_expert.total_bytes + nvl_send_num_tokens_per_rank.total_bytes +
                         nvl_send_num_tokens_per_expert.total_bytes <= nvl_clean_offset * sizeof(int));
        #pragma unroll
        for (int i = thread_id; i < nvl_num_int_clean; i += num_threads)
            nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;

        // Reduce number of tokens per expert into the NVL send buffer
        // TODO: may use NVSHMEM reduction
        EP_DEVICE_ASSERT(num_rdma_experts <= num_threads);
        if (thread_id < num_rdma_experts) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++ i)
                sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + thread_id];
            nvl_reduced_num_tokens_per_expert[thread_id] = sum;//1. 计算每个EP，需要收的token总数（是节点内的所有EP）
        }
        __syncthreads();

        // Reduce RDMA received tokens
        if (thread_id == 0) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++ i) {
                sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + num_rdma_experts];
                recv_rdma_rank_prefix_sum[i] = sum;//从每个rdma rank收token总数的前缀和，存在buffer上收到每个rdma rank的起始位置
            }
            while (ld_volatile_global(moe_recv_rdma_counter_mapped) != -1);
            *moe_recv_rdma_counter_mapped = sum;//输出：从所有rdma rank收到的总数
        }

        // Send numbers of tokens per rank/expert to NVL ranks
        EP_DEVICE_ASSERT(NUM_MAX_NVL_PEERS <= num_threads);
        if (thread_id < NUM_MAX_NVL_PEERS) {
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++ i)
                nvl_send_num_tokens_per_rank.buffer(nvl_rank)[i] = rdma_recv_num_tokens_mixed.recv_buffer(i)[thread_id];//2. 每个nvl rank要收的token总数，也是就是机内转发要发的数量？
            #pragma unroll
            for (int i = 0; i < num_nvl_experts; ++ i)
                nvl_send_num_tokens_per_expert.buffer(nvl_rank)[i] = nvl_reduced_num_tokens_per_expert[thread_id * num_nvl_experts + i];//3. 
        }
        barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);

        // Reduce the number of tokens per rank/expert
        EP_DEVICE_ASSERT(num_nvl_experts <= num_threads);
        if (thread_id == 0) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < num_ranks; ++ i) {
                int src_rdma_rank = i / NUM_MAX_NVL_PEERS, src_nvl_rank = i % NUM_MAX_NVL_PEERS;
                sum += nvl_recv_num_tokens_per_rank.buffer(src_nvl_rank)[src_rdma_rank];
                recv_gbl_rank_prefix_sum[i] = sum;//每个位置表示nvl_rank分别从所有node收的token数
            }
            while (ld_volatile_global(moe_recv_counter_mapped) != -1);
            *moe_recv_counter_mapped = sum; // 输出：disp的output的总大小（是转发之后包含重复的，会比moe_recv_rdma_counter_mapped大）
        }
        if (thread_id < num_nvl_experts) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++ i)
                sum += nvl_recv_num_tokens_per_expert.buffer(i)[thread_id];
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
            while (ld_volatile_global(moe_recv_expert_counter_mapped + thread_id) != -1);
            moe_recv_expert_counter_mapped[thread_id] = sum;
        }

        // Finally barrier
        if (thread_id == 32)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);
    } else {
        // Calculate meta data
        int dst_rdma_rank = sm_id - 1;
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over tokens
            int total_count = 0, per_nvl_rank_count[NUM_MAX_NVL_PEERS] = {0};
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32) {
                EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
                auto is_token_in_rank_uint64 = *reinterpret_cast<const uint64_t*>(is_token_in_rank + i * num_ranks + dst_rdma_rank * NUM_MAX_NVL_PEERS);
                auto is_token_in_rank_values = reinterpret_cast<const bool*>(&is_token_in_rank_uint64);
                #pragma unroll
                for (int j = 0; j < NUM_MAX_NVL_PEERS; ++ j)
                    per_nvl_rank_count[j] += is_token_in_rank_values[j];
                total_count += (is_token_in_rank_uint64 != 0);
            }

            // Warp reduce
            total_count = warp_reduce_sum(total_count);
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++ i)
                per_nvl_rank_count[i] = warp_reduce_sum(per_nvl_rank_count[i]);

            // Write into channel matrix
            if (lane_id == 0) {
                #pragma unroll
                for (int i = 0; i < NUM_MAX_NVL_PEERS; ++ i)
                    gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + i) * num_channels + channel_id] = per_nvl_rank_count[i]; // 每个rank在每个channel中需要处理的token数量
                rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] = total_count;//每个node在每个channel中需要处理的token总数
            }
        }

        // Calculate prefix sum
        __syncthreads();
        if (thread_id == 0) {
            auto prefix_row = rdma_channel_prefix_matrix + dst_rdma_rank * num_channels;
            #pragma unroll
            for (int i = 1; i < num_channels; ++ i)
                prefix_row[i] += prefix_row[i - 1];
        }

        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
        if (thread_id < NUM_MAX_NVL_PEERS) {
            auto prefix_row = gbl_channel_prefix_matrix + (dst_rdma_rank * NUM_MAX_NVL_PEERS + thread_id) * num_channels;
            #pragma unroll
            for (int i = 1; i < num_channels; ++ i)
                prefix_row[i] += prefix_row[i - 1];
        }
    }
}

void notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                     const int* num_tokens_per_rdma_rank, int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                     const bool* is_token_in_rank, int num_tokens, int num_channels,
                     int hidden_int4, int num_scales, int num_topk, int expert_alignment,
                     int* rdma_channel_prefix_matrix, int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix, int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs, int num_max_nvl_chunked_recv_tokens,
                     int** barrier_signal_ptrs, int rank,
                     cudaStream_t stream, int64_t num_rdma_bytes, int64_t num_nvl_bytes,
                     bool low_latency_mode) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(num_rdma_ranks) { \
    auto notify_dispatch_func = low_latency_mode ? \
        notify_dispatch<true, num_rdma_ranks> : notify_dispatch<false, num_rdma_ranks>; \//这里有low_latency_mode？
    LAUNCH_KERNEL(&cfg, notify_dispatch_func, \
                  num_tokens_per_rank, moe_recv_counter_mapped, num_ranks, \
                  num_tokens_per_rdma_rank, moe_recv_rdma_counter_mapped, \
                  num_tokens_per_expert, moe_recv_expert_counter_mapped, num_experts, \
                  is_token_in_rank, num_tokens, num_channels, expert_alignment, \
                  rdma_clean_meta.first, rdma_clean_meta.second, \
                  nvl_clean_meta.first, nvl_clean_meta.second, \
                  rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, \
                  gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
                  rdma_buffer_ptr, \
                  buffer_ptrs, barrier_signal_ptrs, rank, \
                  cpu_rdma_team); } break

    constexpr int kNumThreads = 512;
    const auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

    // Get clean meta
    auto rdma_clean_meta = get_rdma_clean_meta(hidden_int4, num_scales, num_topk, num_topk, num_rdma_ranks, num_max_rdma_chunked_recv_tokens, num_channels);
    auto nvl_clean_meta = get_nvl_clean_meta(hidden_int4, num_scales, num_topk, num_topk, num_rdma_ranks, NUM_MAX_NVL_PEERS, num_max_nvl_chunked_recv_tokens, num_channels, true);
    EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
    EP_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes);
    EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_nvl_bytes < std::numeric_limits<int>::max());

    // Launch kernel
    SETUP_LAUNCH_CONFIG(1 + num_rdma_ranks, kNumThreads, stream);
    SWITCH_RDMA_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

// At most 8 RDMA ranks to be sent
constexpr int get_num_topk_rdma_ranks(int num_rdma_ranks) {
    return num_rdma_ranks < 8 ? num_rdma_ranks : 8;
}

template <bool kLowLatencyMode, int kNumRDMARanks, bool kCachedMode, int kNumTMABytesPerWarp,
          int kNumDispatchRDMASenderWarps, int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks)>
__global__ void __launch_bounds__(((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32), 1)
dispatch(int4* recv_x, float* recv_x_scales, int64_t* recv_topk_idx, float* recv_topk_weights, SourceMeta* recv_src_meta,
         const int4* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
         int* send_rdma_head, int* send_nvl_head,
         int* recv_rdma_channel_prefix_matrix, int* recv_gbl_channel_prefix_matrix,
         const int* rdma_channel_prefix_matrix, const int* recv_rdma_rank_prefix_sum,
         const int* gbl_channel_prefix_matrix, const int* recv_gbl_rank_prefix_sum,
         const bool* is_token_in_rank,
         int num_tokens, int hidden_int4, int num_scales, int num_topk, int num_experts,
         int scale_token_stride, int scale_hidden_stride,
         void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
         void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
         int rank, int num_ranks) {
    enum class WarpRole {
        kRDMASender,
        kRDMASenderCoordinator,
        kRDMAAndNVLForwarder,
        kForwarderCoordinator,
        kNVLReceivers
    };

    const auto num_sms = static_cast<int>(gridDim.x);
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_channels = num_sms / 2, channel_id = sm_id / 2;
    const bool is_forwarder = sm_id % 2 == 0;//偶数sm 做fwd
    const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;

    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe == num_channels or ibgda_get_state()->num_rc_per_pe >= num_sms);

    const auto role_meta = [=]() -> std::pair<WarpRole, int> {
        if (is_forwarder) {
            if (warp_id < NUM_MAX_NVL_PEERS) {
                return {WarpRole::kRDMAAndNVLForwarder, (warp_id + channel_id) % NUM_MAX_NVL_PEERS};//前8个fwd，TODO 没太懂为啥要+channel_id
            } else {
                return {WarpRole::kForwarderCoordinator, warp_id - NUM_MAX_NVL_PEERS};//第8个控制
            }
        } else if (warp_id < kNumDispatchRDMASenderWarps) {
            return {WarpRole::kRDMASender, -1};
        } else if (warp_id == kNumDispatchRDMASenderWarps) {
            return {WarpRole::kRDMASenderCoordinator, -1};
        } else {
            return {WarpRole::kNVLReceivers, (warp_id + channel_id - kNumDispatchRDMASenderWarps) % NUM_MAX_NVL_PEERS};
        }
    }();
    auto warp_role = role_meta.first;
    auto target_rank = role_meta.second; // Not applicable for RDMA senders
    EP_DEVICE_ASSERT(num_warps == kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS);

    // Data checks
    EP_DEVICE_ASSERT(num_topk <= 32);

    // RDMA symmetric layout
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
    auto hidden_bytes = hidden_int4 * sizeof(int4);
    auto scale_bytes = num_scales * sizeof(float);
    // TODO: rename `num_bytes_per_rdma_token` after combine refactor
    auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, num_scales, num_topk, num_topk); // 注意这里包含了所有东西(包括SourceMeta)，不光是token
    auto rdma_channel_data = SymBuffer<uint8_t>(rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_meta = SymBuffer<int>(rdma_buffer_ptr, NUM_MAX_NVL_PEERS * 2 + 2, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

    // NVL buffer layouts
    // NOTES: `rs_wr_buffer_ptr` means "Read for Senders, Write for Receivers", `ws_rr_buffer_ptr` means "Write for Senders, Read for Receivers"
    void *rs_wr_buffer_ptr = nullptr, *ws_rr_buffer_ptr = nullptr;
    int rs_wr_rank = 0, ws_rr_rank = 0;
    if (warp_role == WarpRole::kRDMAAndNVLForwarder)
        rs_wr_buffer_ptr = buffer_ptrs[nvl_rank], ws_rr_buffer_ptr = buffer_ptrs[target_rank], rs_wr_rank = nvl_rank, ws_rr_rank = target_rank;
    if (warp_role == WarpRole::kNVLReceivers)
        rs_wr_buffer_ptr = buffer_ptrs[target_rank], ws_rr_buffer_ptr = buffer_ptrs[nvl_rank], rs_wr_rank = target_rank, ws_rr_rank = nvl_rank;//这两个role的rank配置很妙啊。。。

    // Allocate buffers
    auto nvl_channel_x = AsymBuffer<uint8_t>(ws_rr_buffer_ptr, num_max_nvl_chunked_recv_tokens * num_bytes_per_token, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_prefix_start = AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_prefix_end = AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_head = AsymBuffer<int>(rs_wr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, ws_rr_rank).advance_also(ws_rr_buffer_ptr);
    auto nvl_channel_tail = AsymBuffer<int>(ws_rr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);

    // RDMA sender warp synchronization
    // NOTES: `rdma_send_channel_tail` means the latest released tail
    // NOTES: `rdma_send_channel_window` means the ongoing 32 transactions' status
    __shared__ int rdma_send_channel_lock[kNumRDMARanks];
    __shared__ int rdma_send_channel_tail[kNumRDMARanks];
    __shared__ uint32_t rdma_send_channel_window[kNumRDMARanks];
    auto sync_rdma_sender_smem = []() { asm volatile("bar.sync 0, %0;" :: "r"((kNumDispatchRDMASenderWarps + 1) * 32)); };

    // TMA stuffs
    extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
    auto tma_buffer = smem_tma_buffer + target_rank * kNumTMABytesPerWarp;
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + hidden_bytes);
    uint32_t tma_phase = 0;
    if ((warp_role == WarpRole::kRDMAAndNVLForwarder or warp_role == WarpRole::kNVLReceivers) and lane_id == 0) {
        mbarrier_init(tma_mbarrier, 1);
        fence_view_async_shared();
        fence_barrier_init();
        EP_DEVICE_ASSERT(num_bytes_per_token + sizeof(uint64_t) <= kNumTMABytesPerWarp);
    }
    __syncwarp();

    // Forward warp synchronization
    __shared__ volatile int forward_channel_head[NUM_MAX_NVL_PEERS][kNumRDMARanks];
    __shared__ volatile bool forward_channel_retired[NUM_MAX_NVL_PEERS];
    auto sync_forwarder_smem = []() { asm volatile("bar.sync 1, %0;" :: "r"((NUM_MAX_NVL_PEERS + 1) * 32)); };

    if (warp_role == WarpRole::kRDMASender) {//作用：将token数据打包并放入RDMA发送缓冲区；输出：RDMA缓冲区中的token数据（包含hidden states、scales、topk信息等）
        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        // Send number of tokens in this channel by `-value - 1`
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * 2 + 2 <= 32, "Invalid number of NVL peers");
        for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
            auto dst_ptr = dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) : rdma_channel_meta.send_buffer(dst_rdma_rank);
            if (lane_id < NUM_MAX_NVL_PEERS) {
                dst_ptr[lane_id] = -(channel_id == 0 ? 0 : gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) * num_channels + channel_id - 1]) - 1;//发到每个rank的前缀和，其实就是offset
            } else if (lane_id < NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] = -gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id - NUM_MAX_NVL_PEERS) * num_channels + channel_id] - 1;//发到每个rank的token数
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] = -(channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1]) - 1; // 发到dst_rdma_rank的offset
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2 + 1) {
                dst_ptr[lane_id] = -rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] - 1;// 发到dst_rdma_rank的token数
            }//当前channel发给8个rank的offset，8个rank的size，一个到dst node的offset，一个到dst node的总size
            __syncwarp();

            // Issue RDMA for non-local ranks
            if (dst_rdma_rank != rdma_rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
                                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
                                                  sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2),
                                                  translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                  channel_id, lane_id, 0);
            }
        } // kRDMAAndNVLForwarder接收方获取，不断去poll
        sync_rdma_sender_smem(); // 和 kRDMASenderCoordinator warp做同步，保证对方已经清空再更新

        // Iterate over tokens and copy into buffer
        int64_t token_idx;
        int cached_rdma_channel_head = 0, global_rdma_tail_idx = 0;
        auto send_buffer = lane_id == rdma_rank ? rdma_channel_data.recv_buffer(lane_id) : rdma_channel_data.send_buffer(lane_id);
        for (token_idx = token_start_idx; token_idx < token_end_idx; ++ token_idx) {
            // Read RDMA rank existence
            uint64_t is_token_in_rank_uint64 = 0;
            if (lane_id < kNumRDMARanks) {//一个线程处理一个同号卡
                is_token_in_rank_uint64 = __ldg(reinterpret_cast<const uint64_t *>(is_token_in_rank + token_idx * num_ranks + lane_id * NUM_MAX_NVL_PEERS)); // is_token_in_rank是bool类型，大小1B，转为8B的mask，不为0表示需要给该node发数据；
                global_rdma_tail_idx += (is_token_in_rank_uint64 != 0);
            }
            __syncwarp();

            // Skip the token which does not belong to this warp
            if ((token_idx - token_start_idx) % kNumDispatchRDMASenderWarps != warp_id)
                continue;//只处理warp_id对应的token，为什么循环不直接这样stride，为什么在这个位置跳过：因为global_rdma_tail_idx是全局的，land_id线程要遍历所有token，获取到token在node上的slot
            auto rdma_tail_idx = is_token_in_rank_uint64 == 0 ? -1 : global_rdma_tail_idx - 1;//当前的lane thr给对应的node累计发送的token数

            // Wait the remote buffer to be released
            auto start_time = clock64();
            while (is_token_in_rank_uint64 != 0 and rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {// num_max_rdma_chunked_recv_tokens 应该是一个channel的最大能发的数量；确认 rdma buffer ready
                cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(lane_id)));//851行更新的

                // Timeout check
                if (clock64() - start_time >= NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch RDMA sender timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA lane: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, lane_id, cached_rdma_channel_head, rdma_tail_idx);
                    trap();
                }
            }
            __syncwarp();

            // Store RDMA head for combine
            if (lane_id < kNumRDMARanks and not kCachedMode)
                send_rdma_head[token_idx * kNumRDMARanks + lane_id] = rdma_tail_idx;// TODO

            // Broadcast tails
            SourceMeta src_meta;
            int num_topk_ranks = 0, topk_ranks[kNumTopkRDMARanks]; // num_topk_ranks是实际需要发的node数；
            void* dst_send_buffers[kNumTopkRDMARanks];
            #pragma unroll
            for (int i = 0, slot_idx; i < kNumRDMARanks; ++ i) if ((slot_idx = __shfl_sync(0xffffffff, rdma_tail_idx, i)) >= 0) {//整个warp都可以给topk_ranks等几个变量赋值
                slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;
                topk_ranks[num_topk_ranks] = i;
                auto recv_is_token_in_rank_uint64 = broadcast(is_token_in_rank_uint64, i); //__shfl_sync只能int float 可能只能32位
                auto recv_is_token_in_rank_values = reinterpret_cast<const bool*>(&recv_is_token_in_rank_uint64);//uint64转为bool指针
                if (lane_id == num_topk_ranks)
                    src_meta = SourceMeta(rdma_rank, recv_is_token_in_rank_values);//8B转为一个int表示？
                dst_send_buffers[num_topk_ranks ++] = reinterpret_cast<uint8_t*>(broadcast(send_buffer, i)) + slot_idx * num_bytes_per_token;//发给不同node的send staging是不同的
            }
            EP_DEVICE_ASSERT(num_topk_ranks <= kNumTopkRDMARanks);

            // Copy `x` into symmetric send buffer
            auto st_broadcast = [=](const int key, const int4& value) {
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++ j)//同一个token拷贝多次
                    st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + key, value);
            };
            UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, ld_nc_global, st_broadcast);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++ i)
                dst_send_buffers[i] = reinterpret_cast<int4*>(dst_send_buffers[i]) + hidden_int4;//加偏移

            // Copy `x_scales` into symmetric send buffer
            #pragma unroll
            for (int i = lane_id; i < num_scales; i += 32) {
                auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                auto value = ld_nc_global(x_scales + offset);
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++ j)
                    st_na_global(reinterpret_cast<float*>(dst_send_buffers[j]) + i, value);
            }
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++ i)
                dst_send_buffers[i] = reinterpret_cast<float*>(dst_send_buffers[i]) + num_scales;

            // Copy source metadata into symmetric send buffer
            if (lane_id < num_topk_ranks)
                st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);//对方可以直接知道本地哪些rank要收这个token
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++ i)
                dst_send_buffers[i] = reinterpret_cast<SourceMeta*>(dst_send_buffers[i]) + 1;

            // Copy `topk_idx` and `topk_weights` into symmetric send buffer
            #pragma unroll
            for (int i = lane_id; i < num_topk * num_topk_ranks; i += 32) {
                auto rank_idx = i / num_topk, copy_idx = i % num_topk;
                auto idx_value = static_cast<int>(ld_nc_global(topk_idx + token_idx * num_topk + copy_idx));
                auto weight_value = ld_nc_global(topk_weights + token_idx * num_topk + copy_idx);
                st_na_global(reinterpret_cast<int*>(dst_send_buffers[rank_idx]) + copy_idx, idx_value);
                st_na_global(reinterpret_cast<float*>(dst_send_buffers[rank_idx]) + num_topk + copy_idx, weight_value);
            }
            __syncwarp();

            // Release the transaction in the window
            if (is_token_in_rank_uint64 != 0) {//多个warp对应的token会发给同一个node
                // Acquire lock first
                acquire_lock(rdma_send_channel_lock + lane_id);
                auto latest_tail = rdma_send_channel_tail[lane_id];
                auto offset = rdma_tail_idx - latest_tail;
                while (offset >= 32) {
                    release_lock(rdma_send_channel_lock + lane_id);
                    acquire_lock(rdma_send_channel_lock + lane_id);
                    latest_tail = rdma_send_channel_tail[lane_id];
                    offset = rdma_tail_idx - latest_tail;
                }

                // Release the transaction slot
                // Add the bit and move the ones if possible
                auto window = rdma_send_channel_window[lane_id] | (1u << offset);
                if (offset == 0) {
                    auto num_empty_slots = (~window) == 0 ? 32 : __ffs(~window) - 1;                //__ffs 是最低位的1的位置，从1开始记数；其实也就是windows里从前到后一段连续的1的个数
                    st_release_cta(rdma_send_channel_tail + lane_id, latest_tail + num_empty_slots);//主要是更新tail？窗口起点往前移了
                    window >>= num_empty_slots;//更新tail之后窗口前移
                }
                rdma_send_channel_window[lane_id] = window;

                // Release lock
                release_lock(rdma_send_channel_lock + lane_id);
            }
            __syncwarp();
        }
    } else if (warp_role == WarpRole::kRDMASenderCoordinator) {//作用：协调RDMA发送操作，管理发送队列和流控；输出：控制实际的RDMA网络传输，更新发送进度
        // NOTES: in case of splitting, the issued put at the end of the buffer
        EP_DEVICE_ASSERT(num_max_rdma_chunked_recv_tokens % num_max_rdma_chunked_send_tokens == 0);//前者是单笔收发大小，后者是ibgda put一次发送数量

        // Clean shared memory
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA ranks");
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_lock[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_tail[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_window[lane_id] = 0) : 0;

        // Synchronize shared memory
        sync_rdma_sender_smem();

        // Get number of tokens to send for each RDMA rank
        int num_tokens_to_send = 0;
        if (lane_id < kNumRDMARanks) {
            num_tokens_to_send = rdma_channel_prefix_matrix[lane_id * num_channels + channel_id];
            if (channel_id > 0)
                num_tokens_to_send -= rdma_channel_prefix_matrix[lane_id * num_channels + channel_id - 1];//前缀和相减得size
        }

        // Iterate all RDMA ranks
        int last_issued_tail = 0;
        auto start_time = clock64();
        while (__any_sync(0xffffffff, num_tokens_to_send > 0)) {
            // Timeout check
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                printf("DeepEP RDMA sender coordinator timeout, channel: %d, IB: %d, nvl %d, dst IB: %d, tail: %d, remaining: %d\n",
                       channel_id, rdma_rank, nvl_rank, lane_id, last_issued_tail, num_tokens_to_send);
                trap();
            } // 等待RDMASender拷贝完更新rdma_send_channel_tail，拷贝时间也要检查TO？

            // TODO: try thread-level `put_nbi`?
            for (int i = 0, synced_num_tokens_to_send; i < kNumRDMARanks; ++ i) {
                // To mitigate incast congestion, shuffle the starting index of target rank for different ranks and channels
                int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks; // PR只加了rdma_rank，不加rdma_rank导致同一时间channel0所有node都发往node0
                synced_num_tokens_to_send = __shfl_sync(0xffffffff, num_tokens_to_send, dst_rdma_rank);
                if (synced_num_tokens_to_send == 0)
                    continue;

                // Read the latest progress
                // NOTES: `rdma_send_channel_tail` does not need to be protected by lock
                auto processed_tail = __shfl_sync(0xffffffff, ld_acquire_cta(const_cast<const int*>(rdma_send_channel_tail + dst_rdma_rank)), 0);//获取tail
                auto synced_last_issued_tail = __shfl_sync(0xffffffff, last_issued_tail, dst_rdma_rank);
                auto num_tokens_processed = processed_tail - synced_last_issued_tail;
                if (num_tokens_processed != synced_num_tokens_to_send and num_tokens_processed < num_max_rdma_chunked_send_tokens)
                    continue; // 637行的还是不好说，没凑够就发下一个了。。。这里在等凑够num_max_rdma_chunked_send_tokens个token再发

                // Issue RDMA send
                auto num_tokens_to_issue = min(num_tokens_processed, num_max_rdma_chunked_send_tokens);
                EP_DEVICE_ASSERT(num_tokens_to_issue >= 0 and num_tokens_to_issue <= synced_num_tokens_to_send);
                if (dst_rdma_rank != rdma_rank) {
                    auto dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;
                    EP_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens);
                    const size_t num_bytes_per_msg = num_bytes_per_token * num_tokens_to_issue; // num_tokens_to_issue看起来也聚合了token
                    const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + dst_slot_idx * num_bytes_per_token);
                    const auto src_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + dst_slot_idx * num_bytes_per_token);
                    nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, num_bytes_per_msg,
                                                      translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id, lane_id, 0);
                } else {
                    // Lighter fence for local RDMA rank
                    memory_fence();
                }
                __syncwarp();

                // Update tails
                if (lane_id == dst_rdma_rank) {
                    last_issued_tail += num_tokens_to_issue;
                    num_tokens_to_send -= num_tokens_to_issue;
                    nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank), num_tokens_to_issue,
                                                    translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id, dst_rdma_rank == rdma_rank);
                }
                __syncwarp();
            }
        }
    } else if (warp_role == WarpRole::kRDMAAndNVLForwarder) {//作用：从RDMA缓冲区接收数据，转发到NVL缓冲区；输出：NVL缓冲区中的token数据，实现从跨节点到节点内的数据转发
        // RDMA consumers and NVL producers
        const auto dst_nvl_rank = target_rank;

        // Wait counters to arrive
        int num_tokens_to_recv_from_rdma = 0, src_rdma_channel_prefix = 0;//获取要接受的token，和staging offset
        EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
        auto start_time = clock64();
        if (lane_id < kNumRDMARanks) {
            while (true) {
                auto meta_0 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
                auto meta_1 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS + dst_nvl_rank);
                auto meta_2 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2);
                auto meta_3 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2 + 1);
                if (meta_0 < 0 and meta_1 < 0 and meta_2 < 0 and meta_3 < 0) {
                    // Notify NVL ranks
                    int start_sum = -meta_0 - 1, end_sum = -meta_1 - 1;
                    EP_DEVICE_ASSERT(start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum);
                    st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id, -start_sum - 1);//第lane_id个node的同号卡要发给当前nvl rank的token数
                    st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + lane_id, -end_sum - 1);

                    // Save RDMA channel received token count
                    src_rdma_channel_prefix = -meta_2 - 1;
                    auto src_rdma_channel_prefix_1 = -meta_3 - 1;
                    num_tokens_to_recv_from_rdma = src_rdma_channel_prefix_1 - src_rdma_channel_prefix;
                    if (not kCachedMode)
                        recv_rdma_channel_prefix_matrix[lane_id * num_channels + channel_id] = src_rdma_channel_prefix_1;
                    src_rdma_channel_prefix += lane_id == 0 ? 0 : recv_rdma_rank_prefix_sum[lane_id - 1];
                    EP_DEVICE_ASSERT(num_tokens_to_recv_from_rdma >= 0);
                    break;
                }

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch forwarder timeout (RDMA meta), channel: %d, RDMA: %d, nvl: %d, src RDMA lane: %d, dst NVL: %d, meta: %d, %d, %d, %d\n",
                           channel_id, rdma_rank, nvl_rank, lane_id, dst_nvl_rank, meta_0, meta_1, meta_2, meta_3);
                    trap();
                }
            }
        }
        __syncwarp();

        // Shift cached head
        send_nvl_head += src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank; // TODO src_rdma_channel_prefix是从不同node 收token的offset

        // Wait shared memory to be cleaned
        sync_forwarder_smem();

        // Forward tokens from RDMA buffer
        // NOTES: always start from the local rank
        int src_rdma_rank = sm_id % kNumRDMARanks;//
        int cached_rdma_channel_head = 0, cached_rdma_channel_tail = 0;
        int cached_nvl_channel_head = 0, cached_nvl_channel_tail = 0, rdma_nvl_token_idx = 0;
        while (__any_sync(0xffffffff, num_tokens_to_recv_from_rdma > 0)) {
            // Check destination queue emptiness, or wait a buffer to be released
            start_time = clock64();
            while (true) {
                const int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;
                if (num_max_nvl_chunked_recv_tokens - num_used_slots >= num_max_nvl_chunked_send_tokens) // num_max_nvl_chunked_recv_tokens是实际的nvl buffer大小，num_max_nvl_chunked_send_tokens是一次要发送的tokens数 TODO 只要判断有num_max_nvl_chunked_send_tokens的空余即可，为什么要求空出num_max_nvl_chunked_recv_tokens-num_max_nvl_chunked_send_tokens
                    break;//检查 nvl buffer ready
                cached_nvl_channel_head = __shfl_sync(0xffffffffu, ld_volatile_global(nvl_channel_head.buffer()), 0); // nvl_channel_head应该是dst_nvl_rank独有的head

                // Timeout check
                if (lane_id == 0 and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch forwarder timeout (NVL check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank, ld_volatile_global(nvl_channel_head.buffer()), cached_nvl_channel_tail);
                    trap();
                }
            }

            // Find next source RDMA rank (round-robin)
            start_time = clock64();
            while (true) {//轮询检查 rdma data ready 的node
                src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;
                if (__shfl_sync(0xffffffff, num_tokens_to_recv_from_rdma, src_rdma_rank) > 0) {//表示需要从src_rdma_rank收
                    if (lane_id == src_rdma_rank and cached_rdma_channel_head == cached_rdma_channel_tail)
                        cached_rdma_channel_tail = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(src_rdma_rank))); // 更新 cached_rdma_channel_tail，为什么要转为int
                    if (__shfl_sync(0xffffffff, cached_rdma_channel_tail > cached_rdma_channel_head, src_rdma_rank))
                        break;//说明有数据从 RDMASenderCoordinator 那边来
                }

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    printf("DeepEP dispatch forwarder timeout (RDMA check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, src RDMA lane: %d, head: %d, tail: %d, expected: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank, lane_id, cached_rdma_channel_head, cached_rdma_channel_tail, num_tokens_to_recv_from_rdma);
                    trap();
                }
            }
            auto src_rdma_head = __shfl_sync(0xffffffff, cached_rdma_channel_head, src_rdma_rank);
            auto src_rdma_tail = __shfl_sync(0xffffffff, cached_rdma_channel_tail, src_rdma_rank);

            // Iterate over every token from the RDMA buffer
            for (int i = src_rdma_head, num_tokens_sent = 0; i < src_rdma_tail; ++ i) {
                auto rdma_slot_idx = i % num_max_rdma_chunked_recv_tokens;
                auto shifted = rdma_channel_data.recv_buffer(src_rdma_rank) + rdma_slot_idx * num_bytes_per_token;
                auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));
                lane_id == src_rdma_rank ? (num_tokens_to_recv_from_rdma -= 1) : 0;
                bool is_in_dst_nvl_rank = src_meta.is_token_in_nvl_rank(dst_nvl_rank);//在这个role里，每个warp对应一个nvl rank
                if (lane_id == src_rdma_rank) {
                    auto cached_head = is_in_dst_nvl_rank ? rdma_nvl_token_idx : -1;
                    rdma_nvl_token_idx += is_in_dst_nvl_rank;//累计了所有rdma rank发给dst_nvl_rank的所有tokens
                    if (not kCachedMode)
                        send_nvl_head[i * NUM_MAX_NVL_PEERS] = cached_head; // TODO 记录这个token是否要发给dst_nvl_rank，记下转发到dst_nvl_rank的index；（前面send_nvl_head加了偏移，只会写入第dst_nvl_rank列）
                }
                if (not is_in_dst_nvl_rank)
                    continue;

                // Get an empty slot
                int dst_slot_idx = (cached_nvl_channel_tail ++) % num_max_nvl_chunked_recv_tokens;//来自不同src_rdma_rank的token都被这一个warp写到了同一段nvl_channel_x
                auto dst_shifted = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;

                // Copy data 
                if (lane_id == 0) {
                    tma_load_1d(tma_buffer, shifted, tma_mbarrier, num_bytes_per_token, false); // shifted 是src（rdma recv staging）
                    mbarrier_arrive_and_expect_tx(tma_mbarrier, num_bytes_per_token);
                }
                __syncwarp();
                mbarrier_wait(tma_mbarrier, tma_phase);
                if (lane_id == 0)
                    tma_store_1d(tma_buffer, dst_shifted, num_bytes_per_token); // dst_shifted 是dst（nvl buffer）
                __syncwarp();

                // In case of insufficient NVL buffers, early stopping
                if ((++ num_tokens_sent) == num_max_nvl_chunked_send_tokens)//限制了最大发送数量？？提前截断了。。。最多只转发num_max_nvl_chunked_send_tokens
                    src_rdma_tail = i + 1;

                // Wait TMA to be finished
                tma_store_wait();
                __syncwarp();
            }

            // Sync head index
            if (lane_id == src_rdma_rank)
                forward_channel_head[dst_nvl_rank][src_rdma_rank] = (cached_rdma_channel_head = src_rdma_tail); // 记录每个 NVL rank（行）和每个 RDMA rank（列）在当前 channel 的“头指针”——即已经处理/转发到 NVL 的 RDMA token 的进度。

            // Move tail index
            __syncwarp();
            if (lane_id == 0)
                st_release_sys_global(nvl_channel_tail.buffer(), cached_nvl_channel_tail);
        }

        // Retired
        __syncwarp();
        if (lane_id == 0)
            forward_channel_retired[dst_nvl_rank] = true;
    } else if (warp_role == WarpRole::kForwarderCoordinator) {//作用：协调转发操作，管理RDMA到NVL的数据流控制；输出：更新转发队列状态，确保数据流畅通
        // Extra warps for forwarder coordinator should exit directly 只留一个warp
        if (target_rank > 0)
            return;

        // Forward warp coordinator
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");

        // Clean shared memory
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
        #pragma unroll
        for (int i = lane_id; i < kNumRDMARanks * NUM_MAX_NVL_PEERS; i += 32)
            forward_channel_head[i % NUM_MAX_NVL_PEERS][i / NUM_MAX_NVL_PEERS] = 0;
        if (lane_id < NUM_MAX_NVL_PEERS)
            forward_channel_retired[lane_id] = false;
        sync_forwarder_smem();

        int last_head = 0, target_rdma = lane_id < kNumRDMARanks ? lane_id : 0;
        while (true) {
            // Find minimum head
            int min_head = std::numeric_limits<int>::max();
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++ i) if (not forward_channel_retired[i])
                min_head = min(min_head, forward_channel_head[i][target_rdma]);//lane_id对应的node转发给所有nvl rank里最小的head
            if (__all_sync(0xffffffff, min_head == std::numeric_limits<int>::max()))
                break;//说明都retire了？

            // Update remote head
            if (min_head != std::numeric_limits<int>::max() and min_head >= last_head + num_max_rdma_chunked_send_tokens and lane_id < kNumRDMARanks) {
                nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank), min_head - last_head,
                                                translate_dst_rdma_rank<kLowLatencyMode>(lane_id, nvl_rank), channel_id + num_channels, lane_id == rdma_rank);
                last_head = min_head;
            }

            // Nanosleep and let other warps work
            __nanosleep(NUM_WAIT_NANOSECONDS);
        }
    } else {// kNVLReceivers ：作用：从NVL缓冲区接收数据，写入最终的输出tensor；输出：recv_x、recv_topk_idx、recv_topk_weights等最终结果
        // NVL consumers
        // Retrieve rank offset from barrier results (each lane's register stores an RDMA rank)
        int src_nvl_rank = target_rank, total_offset = 0;
        const int local_expert_begin = rank * (num_experts / num_ranks);
        const int local_expert_end = local_expert_begin + (num_experts / num_ranks);

        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
        if (lane_id < kNumRDMARanks and lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank > 0)
            total_offset = recv_gbl_rank_prefix_sum[lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank - 1]; // src_nvl_rank对应同号卡的前缀和，也就是在recv_x上的offset

        // Receive channel offsets
        int start_offset = 0, end_offset = 0, num_tokens_to_recv;
        auto start_time = clock64();
        while (lane_id < kNumRDMARanks) {
            start_offset = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id); // kRDMAAndNVLForwarder写入的，不同的nvl rank（对应不同的warp）有自己不同的nvl_channel_prefix_start
            end_offset = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);
            if (start_offset < 0 and end_offset < 0) {
                start_offset = -start_offset - 1, end_offset = -end_offset - 1;
                total_offset += start_offset;
                break;
            }

            // Timeout check
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("DeepEP dispatch NVL receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, src nvl: %d, start: %d, end: %d\n",
                       channel_id, rdma_rank, nvl_rank, lane_id, src_nvl_rank, start_offset, end_offset);
                trap();
            }
        }
        num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset); // 从当前warp对应的nvl rank的同号卡需要收的总token数；start_offset就这？？？转发到 nvl_channel_x也可以根据start_offset算出实际的offset减少显存？

        // Save for combine usage TODO
        if (lane_id < kNumRDMARanks and not kCachedMode)
            recv_gbl_channel_prefix_matrix[(lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_channels + channel_id] = total_offset;
        __syncwarp();

        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {
            // Check channel status by lane 0
            start_time = clock64();
            while (true) {
                // Ready to copy 检查 data ready
                if (cached_channel_head_idx != cached_channel_tail_idx)
                    break;
                cached_channel_tail_idx = __shfl_sync(0xffffffff, ld_acquire_sys_global(nvl_channel_tail.buffer()), 0);

                // Timeout check
                if (lane_id == 0 and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch NVL receiver timeout, channel: %d, RDMA: %d, nvl: %d, src NVL: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, src_nvl_rank, cached_channel_head_idx, cached_channel_tail_idx);
                    trap();
                }
            }

            // Copy data
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            for (int chunk_idx = 0; chunk_idx < num_recv_tokens; ++ chunk_idx, -- num_tokens_to_recv) {
                int token_idx_in_buffer = (cached_channel_head_idx ++) % num_max_nvl_chunked_recv_tokens;
                auto shifted = nvl_channel_x.buffer() + token_idx_in_buffer * num_bytes_per_token;
                auto meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));
                int64_t recv_token_idx = __shfl_sync(0xffffffff, total_offset, meta.src_rdma_rank);
                (lane_id == meta.src_rdma_rank) ? (total_offset += 1) : 0;//每个lane维护一个写入到recv_x来自src rdma rank*8+nvl rank这个全局rank那段的offset

                bool scale_aligned = (scale_bytes % 16 == 0);
                auto tma_load_bytes = hidden_bytes + (scale_aligned ? scale_bytes : 0);

                // Copy data 
                if (lane_id == 0) {
                    tma_load_1d(tma_buffer, shifted, tma_mbarrier, tma_load_bytes);//src
                    mbarrier_arrive_and_expect_tx(tma_mbarrier, tma_load_bytes);
                }
                __syncwarp();
                mbarrier_wait(tma_mbarrier, tma_phase);
                if (lane_id == 0)
                    tma_store_1d(tma_buffer, recv_x + recv_token_idx * hidden_int4, hidden_bytes, false);//dst
                __syncwarp();
                shifted += hidden_bytes;

                // Copy scales
                // TODO: make it as templated
                if (scale_aligned) {
                    tma_store_1d(tma_buffer + hidden_bytes, recv_x_scales + recv_token_idx * num_scales, scale_bytes, false);
                } else {
                    UNROLLED_WARP_COPY(1, lane_id, num_scales,
                                       recv_x_scales + recv_token_idx * num_scales,
                                       reinterpret_cast<float*>(shifted),
                                       ld_nc_global, st_na_global);
                }
                shifted += scale_bytes;

                // Copy source meta
                if (lane_id == 0 and not kCachedMode)
                    st_na_global(recv_src_meta + recv_token_idx, meta);// TODO
                shifted += sizeof(SourceMeta);

                // Copy `topk_idx` and `topk_weights`
                if (lane_id < num_topk) {
                    // Read
                    auto idx_value = static_cast<int64_t>(ld_nc_global(reinterpret_cast<int*>(shifted) + lane_id));
                    auto weight_value = ld_nc_global(reinterpret_cast<float*>(shifted + sizeof(int) * num_topk) + lane_id);
                    auto recv_idx = recv_token_idx * num_topk + lane_id;

                    // Transform and write
                    idx_value = (idx_value >= local_expert_begin and idx_value < local_expert_end) ? idx_value - local_expert_begin : -1;
                    weight_value = idx_value >= 0 ? weight_value : 0.0f;
                    st_na_global(recv_topk_idx + recv_idx, idx_value);
                    st_na_global(recv_topk_weights + recv_idx, weight_value);
                }

                // Wait TMA to be finished
                tma_store_wait();
                __syncwarp();
            }

            // Move queue
            if (lane_id == 0)
                st_relaxed_sys_global(nvl_channel_head.buffer(), cached_channel_head_idx);
        }
    }
}

void dispatch(void* recv_x, float* recv_x_scales, int64_t* recv_topk_idx, float* recv_topk_weights, void* recv_src_meta,
              const void* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
              int* send_rdma_head, int* send_nvl_head,
              int* recv_rdma_channel_prefix_matrix, int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix, const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix, const int* recv_gbl_rank_prefix_sum,
              const bool* is_token_in_rank,
              int num_tokens, int hidden_int4, int num_scales, int num_topk, int num_experts,
              int scale_token_stride, int scale_hidden_stride,
              void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
              void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
              int rank, int num_ranks, bool is_cached_dispatch,
              cudaStream_t stream, int num_channels, bool low_latency_mode) {
    constexpr int kNumDispatchRDMASenderWarps = 7;
    constexpr int kNumTMABytesPerWarp = 16384;
    constexpr int smem_size = kNumTMABytesPerWarp * NUM_MAX_NVL_PEERS;

    // Make sure never OOB
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());

#define DISPATCH_LAUNCH_CASE(num_rdma_ranks) { \
    auto dispatch_func = low_latency_mode ? \
        (is_cached_dispatch ? dispatch<true, num_rdma_ranks, true, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps> : \
                              dispatch<true, num_rdma_ranks, false, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps>) : \
        (is_cached_dispatch ? dispatch<false, num_rdma_ranks, true, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps> : \
                              dispatch<false, num_rdma_ranks, false, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps>); \
    SET_SHARED_MEMORY_FOR_TMA(dispatch_func); \
    LAUNCH_KERNEL(&cfg, dispatch_func, \
                  reinterpret_cast<int4*>(recv_x), recv_x_scales, recv_topk_idx, recv_topk_weights, reinterpret_cast<SourceMeta*>(recv_src_meta), \
                  reinterpret_cast<const int4*>(x), x_scales, topk_idx, topk_weights, \
                  send_rdma_head, send_nvl_head, \
                  recv_rdma_channel_prefix_matrix, recv_gbl_channel_prefix_matrix, \
                  rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, \
                  gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
                  is_token_in_rank, \
                  num_tokens, hidden_int4, num_scales, num_topk, num_experts, \
                  scale_token_stride, scale_hidden_stride, \
                  rdma_buffer_ptr, num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens, \
                  buffer_ptrs, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens, \
                  rank, num_ranks); } break

    EP_HOST_ASSERT((topk_idx == nullptr)  == (topk_weights == nullptr));
    EP_HOST_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

    SETUP_LAUNCH_CONFIG(num_channels * 2, (kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32, stream);
    SWITCH_RDMA_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <bool kLowLatencyMode>
__global__ void cached_notify(const int rdma_clean_offset, const int rdma_num_int_clean,
                              const int nvl_clean_offset, const int nvl_num_int_clean,
                              int* combined_rdma_head, int num_combined_tokens, int num_channels,
                              const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, int* combined_nvl_head,
                              void* rdma_buffer_ptr,
                              void** buffer_ptrs, int** barrier_signal_ptrs, int rank, int num_ranks,
                              bool is_cached_dispatch, const nvshmem_team_t rdma_team) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x);
    auto num_threads = static_cast<int>(blockDim.x);
    auto num_warps = num_threads / 32;
    auto warp_id = thread_id / 32;
    auto lane_id = get_lane_id();

    auto nvl_rank = rank % NUM_MAX_NVL_PEERS;
    auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

    // Using two SMs, which clean the RDMA/NVL buffer respectively
    if (sm_id == 0) {
        // Barrier for RDMA
        if (thread_id == 32)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);

        // Barrier for NVL
        barrier_block<NUM_MAX_NVL_PEERS, true>(barrier_signal_ptrs, nvl_rank);

        // Clean RDMA buffer
        auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
        #pragma unroll
        for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
            rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;

        // Clean NVL buffer
        auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);
        #pragma unroll
        for (int i = thread_id; i < nvl_num_int_clean; i += num_threads)
            nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;
        __syncthreads();

        // Barrier again
        if (thread_id == 32)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);
    } else if (sm_id == 1) {
        if (is_cached_dispatch)
            return;

        EP_DEVICE_ASSERT(num_warps >= num_channels);
        EP_DEVICE_ASSERT(num_rdma_ranks <= 32);

        // Iterate in reverse order
        if (lane_id < num_rdma_ranks and warp_id < num_channels) {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_channels, warp_id, token_start_idx, token_end_idx);

            // NOTES: `1 << 25` is a heuristic large number
            int last_head = 1 << 25;
            for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; -- token_idx) {
                auto current_head = __ldg(combined_rdma_head + token_idx * num_rdma_ranks + lane_id); // combined_rdma_head 即 send_nvl_head
                if (current_head < 0) {
                    combined_rdma_head[token_idx * num_rdma_ranks + lane_id] = -last_head - 1;
                } else {
                    last_head = current_head;
                }
            }
        }
    } else {
        if (is_cached_dispatch)
            return;

        EP_DEVICE_ASSERT(num_warps >= num_channels);
        EP_DEVICE_ASSERT(rdma_channel_prefix_matrix != nullptr and rdma_rank_prefix_sum != nullptr);
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Too many NVL peers");

        if (lane_id < NUM_MAX_NVL_PEERS and warp_id < num_channels) {
            for (int dst_rdma_rank = sm_id - 2; dst_rdma_rank < num_rdma_ranks; dst_rdma_rank += num_channels * 2 - 2) {
                // Iterate in reverse order
                int token_start_idx = warp_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id - 1];
                int token_end_idx = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id];
                int shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
                token_start_idx += shift, token_end_idx += shift;

                // NOTES: `1 << 25` is a heuristic large number
                int last_head = 1 << 25;
                #pragma unroll
                for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; -- token_idx)  {
                    auto current_head = __ldg(combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);
                    if (current_head < 0) {
                        combined_nvl_head[token_idx * NUM_MAX_NVL_PEERS + lane_id] = -last_head - 1;
                    } else {
                        last_head = current_head;
                    }
                }
            }
        }
    }
}

void cached_notify(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights,
                   int num_ranks, int num_channels, int num_combined_tokens, int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, int* combined_nvl_head,
                   void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs, int num_max_nvl_chunked_recv_tokens,
                   int** barrier_signal_ptrs, int rank, cudaStream_t stream,
                   int64_t num_rdma_bytes, int64_t num_nvl_bytes,
                   bool is_cached_dispatch, bool low_latency_mode) {
    const int num_threads = std::max(128, 32 * num_channels);
    const auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

    // Get clean meta
    auto rdma_clean_meta = get_rdma_clean_meta(hidden_int4, num_scales, num_topk_idx, num_topk_weights, num_rdma_ranks, num_max_rdma_chunked_recv_tokens, num_channels);
    auto nvl_clean_meta = get_nvl_clean_meta(hidden_int4, num_scales, num_topk_idx, num_topk_weights, num_rdma_ranks, NUM_MAX_NVL_PEERS, num_max_nvl_chunked_recv_tokens, num_channels, is_cached_dispatch);
    EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
    EP_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes);
    EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_nvl_bytes < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_channels * 2 > 3);

    // Launch kernel
    auto cached_notify_func = low_latency_mode ? cached_notify<true> : cached_notify<false>;
    SETUP_LAUNCH_CONFIG(num_channels * 2, num_threads, stream);
    LAUNCH_KERNEL(&cfg, cached_notify_func,
                  rdma_clean_meta.first, rdma_clean_meta.second,
                  nvl_clean_meta.first, nvl_clean_meta.second,
                  combined_rdma_head, num_combined_tokens, num_channels,
                  rdma_channel_prefix_matrix, rdma_rank_prefix_sum, combined_nvl_head,
                  rdma_buffer_ptr,
                  buffer_ptrs, barrier_signal_ptrs, rank, num_ranks,
                  is_cached_dispatch, cpu_rdma_team);
}

template <int kNumRanks, bool kMaybeWithBias, typename dtype_t, int kMaxNumRanks, typename ReceiveFn, typename ReceiveTWFn>
__device__ int combine_token(bool is_token_in_rank, int head_idx,
                             int lane_id, int hidden_int4, int num_topk,
                             int4* combined_row, float* combined_topk_weights,
                             const int4* bias_0_int4, const int4* bias_1_int4,
                             int num_max_recv_tokens, const ReceiveFn& recv_fn, const ReceiveTWFn& recv_tw_fn) {
    constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);

    // Broadcast current heads
    // Lane `i` holds the head of rank `i` and `is_token_in_rank`
    EP_STATIC_ASSERT(kMaxNumRanks <= 32, "Too many ranks");
    int num_topk_ranks = 0, topk_ranks[kMaxNumRanks], slot_indices[kMaxNumRanks];
    #pragma unroll
    for (int i = 0; i < kNumRanks; ++ i) if (__shfl_sync(0xffffffff, is_token_in_rank, i)) {//这个is_token_in_rank很重要，判断dispatch是否有发送到对应的nvl rank上
        slot_indices[num_topk_ranks] = __shfl_sync(0xffffffff, head_idx, i) % num_max_recv_tokens;
        topk_ranks[num_topk_ranks ++] = i;
    }
    EP_DEVICE_ASSERT(num_topk_ranks <= kMaxNumRanks);

    // Reduce data
    #pragma unroll
    for (int i = lane_id; i < hidden_int4; i += 32) {
        // Read bias
        // TODO: make it as a finer-grained template
        int4 bias_0_value_int4, bias_1_value_int4;
        if (kMaybeWithBias) {
            bias_0_value_int4 = bias_0_int4 != nullptr ? ld_nc_global(bias_0_int4 + i) : make_int4(0, 0, 0, 0);
            bias_1_value_int4 = bias_1_int4 != nullptr ? ld_nc_global(bias_1_int4 + i) : make_int4(0, 0, 0, 0);
        }

        // Read buffers
        // TODO: maybe too many registers here
        int4 recv_value_int4[kMaxNumRanks];
        #pragma unroll
        for (int j = 0; j < num_topk_ranks; ++ j)
            recv_value_int4[j] = recv_fn(topk_ranks[j], slot_indices[j], i);//取值
        
        // Clean
        // Reduce bias
        float values[kDtypePerInt4] = {0};
        if (kMaybeWithBias) {
            auto bias_0_values = reinterpret_cast<const dtype_t*>(&bias_0_value_int4);
            auto bias_1_values = reinterpret_cast<const dtype_t*>(&bias_1_value_int4);
            #pragma unroll
            for (int j = 0; j < kDtypePerInt4; ++ j)
                values[j] = static_cast<float>(bias_0_values[j]) + static_cast<float>(bias_1_values[j]);
        }

        // Reduce all-to-all results
        #pragma unroll
        for (int j = 0; j < num_topk_ranks; ++ j) {
            auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
            #pragma unroll
            for (int k = 0; k < kDtypePerInt4; ++ k)
                values[k] += static_cast<float>(recv_value_dtypes[k]);
        }

        // Cast back to `dtype_t` and write
        int4 out_int4;
        auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
        #pragma unroll
        for (int j = 0; j < kDtypePerInt4; ++ j)
            out_dtypes[j] = static_cast<dtype_t>(values[j]);
        st_na_global(combined_row + i, out_int4);
    }

    // Reduce `topk_weights`
    if (lane_id < num_topk) {
        float value = 0;
        #pragma unroll
        for (int i = 0; i < num_topk_ranks; ++ i)
            value += recv_tw_fn(topk_ranks[i], slot_indices[i], lane_id);
        st_na_global(combined_topk_weights + lane_id, value);
    }

    // Return the minimum top-k rank
    return topk_ranks[0];
}

template<bool kLowLatencyMode,
         int kNumRDMARanks, typename dtype_t,
         int kNumCombineForwarderWarps,
         int kNumTMABytesPerWarp,
         int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks),
         int kNumWarpsPerForwarder = (kNumCombineForwarderWarps / kNumRDMARanks > 0) ? kNumCombineForwarderWarps / kNumRDMARanks : 1,
         int kNumForwarders = kNumRDMARanks * kNumWarpsPerForwarder,
         int kNumRDMAReceivers = kNumForwarders - NUM_MAX_NVL_PEERS>
__global__ void __launch_bounds__((kNumForwarders + 1) * 32, 1)
combine(int4* combined_x, float* combined_topk_weights,
        const bool* is_combined_token_in_rank,
        const int4* x, const float* topk_weights,
        const int4* bias_0, const int4* bias_1,
        const int* combined_rdma_head, const int* combined_nvl_head,
        const SourceMeta* src_meta, const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, const int* gbl_channel_prefix_matrix,
        int num_tokens, int num_combined_tokens, int hidden, int num_topk,
        void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
        void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
        int rank, int num_ranks) {
    enum class WarpRole {
        kNVLSender,
        kNVLAndRDMAForwarder,
        kRDMAReceiver,
        kCoordinator
    };

    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const auto num_channels = static_cast<int>(gridDim.x) / 2, channel_id = sm_id / 2;
    const bool is_forwarder_sm = sm_id % 2 == 1;

    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(hidden % (sizeof(int4) / sizeof(dtype_t)) == 0);
    const auto hidden_int4 = hidden / (sizeof(int4) / sizeof(dtype_t));
    const auto hidden_bytes = hidden_int4 * sizeof(int4);
    const auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, 0, 0, num_topk);

    // NOTES: we decouple a channel into 2 SMs
    const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    auto role_meta = [=]() -> std::pair<WarpRole, int> {
        auto warp_id = thread_id / 32;
        if (not is_forwarder_sm) {
            if (warp_id < NUM_MAX_NVL_PEERS) {
                auto shuffled_warp_id = warp_id;
                shuffled_warp_id = (shuffled_warp_id + channel_id) % NUM_MAX_NVL_PEERS;
                return {WarpRole::kNVLSender, shuffled_warp_id}; // 8个sender
            } else if (warp_id < kNumForwarders) {
                return {WarpRole::kRDMAReceiver, warp_id - NUM_MAX_NVL_PEERS};
            } else {
                return {WarpRole::kCoordinator, 0};
            }
        } else {
            if (warp_id < kNumForwarders) {
                auto shuffled_warp_id = (warp_id + channel_id) % kNumForwarders;
                return {WarpRole::kNVLAndRDMAForwarder, shuffled_warp_id};
            } else {
                return {WarpRole::kCoordinator, 0};
            }
        }
    }();
    auto warp_role = role_meta.first;
    auto warp_id = role_meta.second;

    EP_DEVICE_ASSERT(num_warps == kNumForwarders + 1);
    auto num_max_nvl_chunked_recv_tokens_per_rdma = num_max_nvl_chunked_recv_tokens / kNumRDMARanks;

    if (warp_role == WarpRole::kNVLSender) {//从x拷贝到nvl_channel_x
        // NVL producers
        const auto dst_nvl_rank = warp_id;

        // NVL layouts
        // NOTES: to avoid deadlocks, we use separate NVL buffers for different RDMA sources
        auto dst_buffer_ptr = buffer_ptrs[dst_nvl_rank], local_buffer_ptr = buffer_ptrs[nvl_rank];
        auto nvl_channel_x = AsymBuffer<uint8_t>(dst_buffer_ptr, num_max_nvl_chunked_recv_tokens * num_bytes_per_token, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank).advance_also(local_buffer_ptr);
        auto nvl_channel_head = AsymBuffer<int>(local_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, dst_nvl_rank).advance_also(dst_buffer_ptr);
        auto nvl_channel_tail = AsymBuffer<int>(dst_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank).advance_also(local_buffer_ptr);

        // TMA stuffs
        extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
        auto tma_buffer = smem_tma_buffer + dst_nvl_rank * kNumTMABytesPerWarp;
        auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + hidden_bytes);
        uint32_t tma_phase = 0;
        if (lane_id == 0) {
            mbarrier_init(tma_mbarrier, 1);
            fence_view_async_shared();
            fence_barrier_init();
            EP_DEVICE_ASSERT(hidden_bytes + sizeof(uint64_t) <= kNumTMABytesPerWarp);
        }
        __syncwarp();

        // Get tasks for each RDMA lane
        int token_start_idx = 0, token_end_idx = 0;
        if (lane_id < kNumRDMARanks) {
            int prefix_idx = (lane_id * NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_channels + channel_id;//每个warp负责一个local rank
            token_start_idx = gbl_channel_prefix_matrix[prefix_idx]; // channel_id上从lane_id * NUM_MAX_NVL_PEERS + dst_nvl_rank收到的token前缀和
            token_end_idx = (prefix_idx == num_channels * num_ranks - 1) ? num_tokens : gbl_channel_prefix_matrix[prefix_idx + 1]; // num_tokens是收到的token总数
        }
        __syncwarp();

        // NOTES: here the cached value of each lane is only responsible for a single RDMA buffer
        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0; // TODO head tail始终都为0，可能是传入的nvl_num_int_clean和rdma_buffer_ptr clean了staging buffer，同时还有同号卡的barrier
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");

        // Iterate over all tokens and send by chunks
        int current_rdma_idx = channel_id % kNumRDMARanks;
        while (true) {
            // Exit if possible
            if (__all_sync(0xffffffff, token_start_idx >= token_end_idx))
                break;

            // Decide the next RDMA buffer to send
            bool is_lane_ready = false;
            auto start_time = clock64();
            while (true) {
                int num_used_slots = cached_channel_tail_idx - cached_channel_head_idx;
                is_lane_ready = lane_id < kNumRDMARanks and token_start_idx < token_end_idx and num_max_nvl_chunked_recv_tokens_per_rdma - num_used_slots >= num_max_nvl_chunked_send_tokens;//对应的node有token需要发，并且check nvl buffer ready（TODO 不同的node似乎有不同的nvl buffer）
                if (__any_sync(0xffffffff, is_lane_ready))
                    break;

                // Retry
                if (lane_id < kNumRDMARanks and token_start_idx < token_end_idx)
                    cached_channel_head_idx = ld_volatile_global(nvl_channel_head.buffer() + lane_id);//lane_id代表node，不同node维护了一个不同的head？？

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    printf("DeepEP combine NVL sender timeout, channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, RDMA lane: %d, head: %d, tail: %d, start: %d, end: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank, lane_id, ld_volatile_global(nvl_channel_head.buffer() + lane_id), cached_channel_tail_idx,
                           token_start_idx, token_end_idx);
                    trap();
                }
            }

            // Sync token start index and count
            for (int i = 0; i < kNumRDMARanks; ++ i) {
                current_rdma_idx = (current_rdma_idx + 1) % kNumRDMARanks;//选中一个node
                if (__shfl_sync(0xffffffff, (token_start_idx >= token_end_idx) or (not is_lane_ready), current_rdma_idx))
                    continue;

                // Sync token start index
                auto token_idx = static_cast<int64_t>(__shfl_sync(0xffffffff, token_start_idx, current_rdma_idx)); // token_idx后面会作为x的索引，
                int num_tokens_in_chunk = __shfl_sync(0xffffffff, min(num_max_nvl_chunked_send_tokens, token_end_idx - token_start_idx), current_rdma_idx); // 最多发num_max_nvl_chunked_send_tokens个，前面只保证了nvl buffer有num_max_nvl_chunked_send_tokens这么多的空余

                // Send by chunk
                for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk; ++ chunk_idx, ++ token_idx) {
                    // Get an empty slot
                    int dst_slot_idx = 0;
                    if (lane_id == current_rdma_idx) {
                        dst_slot_idx = (cached_channel_tail_idx ++) % num_max_nvl_chunked_recv_tokens_per_rdma;
                        dst_slot_idx = current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma + dst_slot_idx;//+前是对应node的base offset
                    }
                    dst_slot_idx = __shfl_sync(0xffffffff, dst_slot_idx, current_rdma_idx);

                    // Load data
                    auto shifted_x_buffers = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;//dst是nvl buffer
                    auto shifted_x = x + token_idx * hidden_int4;//source是input
                    if (lane_id == 0) {
                        tma_store_wait();
                        tma_load_1d(tma_buffer, shifted_x, tma_mbarrier, hidden_bytes);
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, hidden_bytes);
                    }
                    __syncwarp();
                    mbarrier_wait(tma_mbarrier, tma_phase);

                    // Load source meta
                    if (lane_id == num_topk)
                        *reinterpret_cast<SourceMeta*>(tma_buffer + hidden_bytes) = ld_nc_global(src_meta + token_idx);

                    // Load `topk_weights`
                    if (lane_id < num_topk)
                        *reinterpret_cast<float*>(tma_buffer + hidden_bytes + sizeof(SourceMeta) + lane_id * sizeof(float)) = ld_nc_global(topk_weights + token_idx * num_topk + lane_id);

                    // Issue TMA store
                    tma_store_fence();
                    __syncwarp();
                    if (lane_id == 0)
                        tma_store_1d(tma_buffer, shifted_x_buffers, num_bytes_per_token, false);
                }
                lane_id == current_rdma_idx ? (token_start_idx = static_cast<int>(token_idx)) : 0;
            }

            // Move queue tail
            tma_store_wait();
            __syncwarp();
            if (lane_id < kNumRDMARanks and is_lane_ready)
                st_release_sys_global(nvl_channel_tail.buffer() + lane_id, cached_channel_tail_idx);
        }
    } else {
        // Combiners and coordinators
        // RDMA symmetric layout
        auto rdma_channel_data = SymBuffer<int8_t>(rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels);
        auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
        auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

        // NVL layouts
        void* local_nvl_buffer = buffer_ptrs[nvl_rank];
        void* nvl_buffers[NUM_MAX_NVL_PEERS];
        #pragma unroll
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++ i)
            nvl_buffers[i] = buffer_ptrs[i];
        auto nvl_channel_x = AsymBuffer<uint8_t>(local_nvl_buffer, num_max_nvl_chunked_recv_tokens * num_bytes_per_token, NUM_MAX_NVL_PEERS, channel_id, num_channels).advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
        auto nvl_channel_head = AsymBuffer<int, NUM_MAX_NVL_PEERS>(nvl_buffers, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank).advance_also(local_nvl_buffer);
        auto nvl_channel_tail = AsymBuffer<int>(local_nvl_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels).advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);

        // Combiner warp synchronization
        __shared__ volatile int forwarder_nvl_head[kNumForwarders][NUM_MAX_NVL_PEERS];
        __shared__ volatile bool forwarder_retired[kNumForwarders];
        __shared__ volatile int rdma_receiver_rdma_head[kNumRDMAReceivers][kNumRDMARanks];
        __shared__ volatile bool rdma_receiver_retired[kNumRDMAReceivers];
        auto sync_forwarder_smem = [=]() { asm volatile("bar.sync 0, %0;" :: "r"((kNumForwarders + 1) * 32)); };
        auto sync_rdma_receiver_smem = [=]() { asm volatile("bar.sync 1, %0;" :: "r"((kNumRDMAReceivers + 1) * 32)); };

        if (warp_role == WarpRole::kNVLAndRDMAForwarder) {//从 nvl_channel_x reduce 到 send_buffer （rdma_channel_data）
            // Receive from NVL ranks and forward to RDMA ranks
            // NOTES: this part is using "large warps" for each RDMA ranks
            const auto dst_rdma_rank = warp_id / kNumWarpsPerForwarder; // kNumWarpsPerForwarder 个warp处理一个node
            const auto sub_warp_id = warp_id % kNumWarpsPerForwarder;
            auto send_buffer = dst_rdma_rank == rdma_rank ? rdma_channel_data.recv_buffer(dst_rdma_rank) : rdma_channel_data.send_buffer(dst_rdma_rank);
            auto sync_large_warp = [=]() {
                if (kNumWarpsPerForwarder == 1) {
                    __syncwarp();
                } else {
                    asm volatile("bar.sync %0, %1;" :: "r"(dst_rdma_rank + 2), "r"(kNumWarpsPerForwarder * 32));
                }
            };
            EP_STATIC_ASSERT(kNumWarpsPerForwarder == 1 or kNumRDMARanks + 2 <= 16, "Barriers are not enough");

            // Advance to the corresponding NVL buffer
            nvl_channel_x.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma * num_bytes_per_token);
            nvl_channel_head.advance(dst_rdma_rank);
            nvl_channel_tail.advance(dst_rdma_rank);

            // Clean shared memory and sync
            EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
            lane_id < NUM_MAX_NVL_PEERS ? (forwarder_nvl_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (forwarder_retired[warp_id] = false) : false;
            sync_forwarder_smem();

            // Get count and cached head
            int cached_nvl_channel_tail_idx = 0;
            int num_tokens_to_combine = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
            int num_tokens_prefix = channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
            num_tokens_to_combine -= num_tokens_prefix; // 从dst_rdma_rank收到的token总数
            num_tokens_prefix += dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
            combined_nvl_head += num_tokens_prefix * NUM_MAX_NVL_PEERS; // 维度: [num_combined_tokens, NUM_MAX_NVL_PEERS]，num_combined_tokens是dispatch的输出token数？

            // Iterate over all tokens and combine by chunks
            for (int token_start_idx = 0; token_start_idx < num_tokens_to_combine; token_start_idx += num_max_rdma_chunked_send_tokens) {
                // Check destination queue emptiness, or wait a buffer to be released
                auto token_end_idx = min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_combine); // 一次处理 num_max_rdma_chunked_send_tokens 这么多的token
                auto num_chunked_tokens = token_end_idx - token_start_idx;
                auto start_time = clock64();
                while (sub_warp_id == 0 and lane_id == 0) {
                    // Inequality: `num_max_rdma_chunked_recv_tokens - (tail - head) >= num_chunked_tokens`
                    // Here, `token_start_idx` is the actual tail
                    int num_used_slots = token_start_idx - ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
                    if (num_max_rdma_chunked_recv_tokens - num_used_slots >= num_chunked_tokens)// check rdma buffer ready
                        break;

                    // Timeout check
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP combine forwarder (RDMA check) timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA: %d, head: %ld, tail: %d, chunked: %d\n",
                               channel_id, rdma_rank, nvl_rank, dst_rdma_rank, ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank)), token_start_idx, num_chunked_tokens);
                        trap();
                    }
                }
                sync_large_warp();

                // Combine and write to the RDMA buffer
                for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx; token_idx += kNumWarpsPerForwarder) {
                    // Read expected head
                    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
                    int expected_head = -1;
                    if (lane_id < NUM_MAX_NVL_PEERS)
                        expected_head = ld_nc_global(combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id); // 表示token_idx转发到对应的nvl_channel_x，在nvl_channel_x上的index，后续也作为nvl_channel_x的 slot_idx

                    // Wait lanes to be ready
                    start_time = clock64();
                    while (cached_nvl_channel_tail_idx <= expected_head) {//大于8的线程，不满足0<=-1，直接退出，这后面也没有加sync，可能是后面函数里有对于expected_head的__shfl_sync？
                        cached_nvl_channel_tail_idx = ld_acquire_sys_global(nvl_channel_tail.buffer(lane_id));

                        // Timeout check
                        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < NUM_MAX_NVL_PEERS) {
                            printf("DeepEP combine forwarder (NVL check) timeout, channel: %d, RDMA: %d, nvl: %d, src NVL: %d, dst RDMA: %d, tail: %d, waiting: %d, total: %d, sub: %d, large: %d, expected: %d\n",
                                   channel_id, rdma_rank, nvl_rank, lane_id, dst_rdma_rank, cached_nvl_channel_tail_idx, token_idx, num_tokens_to_combine, sub_warp_id, kNumWarpsPerForwarder, expected_head);
                            trap();
                        }
                    }

                    // Combine current token
                    auto rdma_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;
                    void* shifted = send_buffer + rdma_slot_idx * num_bytes_per_token;// dst
                    auto recv_fn = [&](int src_nvl_rank, int slot_idx, int hidden_int4_idx) -> int4 { return ld_nc_global(reinterpret_cast<int4*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token) + hidden_int4_idx); };//这里的source，TODO 怎么就能保证这个几个token是从同一个source token过来的？
                    auto recv_tw_fn = [&](int src_nvl_rank, int slot_idx, int topk_idx) -> float { return ld_nc_global(reinterpret_cast<float*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token + hidden_bytes + sizeof(SourceMeta)) + topk_idx); };
                    combine_token<NUM_MAX_NVL_PEERS, false, dtype_t, NUM_MAX_NVL_PEERS>(expected_head >= 0,
                                                                                 expected_head, lane_id,
                                                                                 hidden_int4, num_topk,
                                                                                 static_cast<int4*>(shifted),
                                                                                 reinterpret_cast<float*>(static_cast<int8_t*>(shifted) + hidden_bytes + sizeof(SourceMeta)),
                                                                                 nullptr, nullptr, num_max_nvl_chunked_recv_tokens_per_rdma, recv_fn, recv_tw_fn);

                    // Update head
                    if (lane_id < NUM_MAX_NVL_PEERS)
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1) : (forwarder_nvl_head[warp_id][lane_id] = expected_head + 1);
                }
                sync_large_warp();

                // Issue RDMA send 调用GDA的warp也要干活
                if (sub_warp_id == kNumWarpsPerForwarder - 1) {
                    if (dst_rdma_rank != rdma_rank) {
                        auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
                        const size_t num_bytes_per_msg = num_chunked_tokens * num_bytes_per_token;
                        const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                        const auto src_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                        nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, num_bytes_per_msg,
                                                          translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id, lane_id, 0);
                    } else {
                        memory_fence();
                    }

                    // Write new RDMA tail
                    __syncwarp();
                    if (lane_id == 0) {
                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank), num_chunked_tokens,
                                                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id, dst_rdma_rank == rdma_rank);
                    }
                }
            }

            // Retired
            __syncwarp();
            if (lane_id == 0)
                forwarder_retired[warp_id] = true;
        } else if (warp_role == WarpRole::kRDMAReceiver) {
            // Receive from RDMA ranks and write to the output tensor
            // Clean shared memory and sync
            EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
            lane_id < kNumRDMARanks ? (rdma_receiver_rdma_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (rdma_receiver_retired[warp_id] = false) : 0;
            sync_rdma_receiver_smem();

            // The same tokens as the dispatch process
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over all tokens and combine
            int cached_channel_tail_idx = 0;
            for (int64_t token_idx = token_start_idx + warp_id; token_idx < token_end_idx; token_idx += kNumRDMAReceivers) {
                // Read expected head
                EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
                int expected_head = -1;
                if (lane_id < kNumRDMARanks) {
                    expected_head = ld_nc_global(combined_rdma_head + token_idx * kNumRDMARanks + lane_id);
                    (expected_head < 0) ? (rdma_receiver_rdma_head[warp_id][lane_id] = -expected_head - 1) : (rdma_receiver_rdma_head[warp_id][lane_id] = expected_head);
                }

                // Wait lanes to be ready
                auto start_time = clock64();
                while (cached_channel_tail_idx <= expected_head) {
                    cached_channel_tail_idx = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id)));

                    // Timeout check
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP combine RDMA receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, tail: %d, waiting: %ld, expect: %d\n",
                               channel_id, rdma_rank, nvl_rank, lane_id, cached_channel_tail_idx, token_idx, expected_head);
                        trap();
                    }
                }
                __syncwarp();

                // Combine current token
                auto recv_fn = [&](int src_rdma_rank, int slot_idx, int hidden_int4_idx) -> int4 { return ld_nc_global(reinterpret_cast<const int4*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_token) + hidden_int4_idx);};
                auto recv_tw_fn = [&](int src_rdma_rank, int slot_idx, int topk_idx) -> float { return ld_nc_global(reinterpret_cast<const float*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_token + hidden_bytes + sizeof(SourceMeta)) + topk_idx);};
                combine_token<kNumRDMARanks, true, dtype_t, kNumTopkRDMARanks>(expected_head >= 0,
                                                                         expected_head, lane_id,
                                                                         hidden_int4, num_topk,
                                                                         combined_x + token_idx * hidden_int4,
                                                                         combined_topk_weights + token_idx * num_topk,
                                                                         bias_0 == nullptr ? nullptr : bias_0 + token_idx * hidden_int4,
                                                                         bias_1 == nullptr ? nullptr : bias_1 + token_idx * hidden_int4,
                                                                         num_max_rdma_chunked_recv_tokens, recv_fn, recv_tw_fn);
            }

            // Retired
            __syncwarp();
            if (lane_id == 0)
                rdma_receiver_retired[warp_id] = true;
        } else {
            // Coordinator
            // Sync shared memory status
            is_forwarder_sm ? sync_forwarder_smem() : sync_rdma_receiver_smem();
            const auto num_warps_per_rdma_rank = kNumForwarders / kNumRDMARanks;

            int last_rdma_head = 0;
            int last_nvl_head[kNumRDMARanks] = {0};
            int dst_rdma_rank = lane_id < kNumRDMARanks ? lane_id : 0;
            int dst_nvl_rank = lane_id < NUM_MAX_NVL_PEERS ? lane_id : 0;
            EP_STATIC_ASSERT(kNumCombineForwarderWarps <= 32, "Invalid number of forwarder warps");
            while (true) {
                // Retired
                if (not is_forwarder_sm and __all_sync(0xffffffff, lane_id >= kNumRDMAReceivers or rdma_receiver_retired[lane_id]))
                    break;
                if (is_forwarder_sm and __all_sync(0xffffffff, lane_id >= kNumForwarders or forwarder_retired[lane_id]))
                    break;

                // Find minimum head for RDMA ranks
                if (not is_forwarder_sm) {
                    int min_head = std::numeric_limits<int>::max();
                    #pragma unroll
                    for (int i = 0; i < kNumRDMAReceivers; ++ i) if (not rdma_receiver_retired[i])
                        min_head = min(min_head, rdma_receiver_rdma_head[i][dst_rdma_rank]);
                    if (min_head != std::numeric_limits<int>::max() and min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens and lane_id < kNumRDMARanks) {
                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank), min_head - last_rdma_head,
                                                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id + num_channels, dst_rdma_rank == rdma_rank);
                        last_rdma_head = min_head;
                    }
                } else {
                    // Find minimum head for NVL ranks
                    #pragma unroll
                    for (int i = 0; i < kNumRDMARanks; ++ i) {
                        int min_head = std::numeric_limits<int>::max();
                        #pragma unroll
                        for (int j = 0; j < num_warps_per_rdma_rank; ++ j) if (not forwarder_retired[i * num_warps_per_rdma_rank + j])
                            min_head = min(min_head, forwarder_nvl_head[i * num_warps_per_rdma_rank + j][dst_nvl_rank]);
                        if (min_head != std::numeric_limits<int>::max() and min_head > last_nvl_head[i] and lane_id < NUM_MAX_NVL_PEERS)
                            st_relaxed_sys_global(nvl_channel_head.buffer_by(dst_nvl_rank) + i, last_nvl_head[i] = min_head);
                    }
                }

                // Nanosleep and let other warps work
                __nanosleep(NUM_WAIT_NANOSECONDS);
            }
        }
    }
}

void combine(cudaDataType_t type,
             void* combined_x, float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x, const float* topk_weights,
             const void* bias_0, const void* bias_1,
             const int* combined_rdma_head, const int* combined_nvl_head,
             const void* src_meta, const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, const int* gbl_channel_prefix_matrix,
             int num_tokens, int num_combined_tokens, int hidden, int num_topk,
             void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
             int rank, int num_ranks, cudaStream_t stream, int num_channels, bool low_latency_mode) {
    constexpr int kNumCombineForwarderWarps = 16;
    constexpr int kNumTMABytesPerWarp = 16384;
    constexpr int smem_size = kNumTMABytesPerWarp * NUM_MAX_NVL_PEERS;

#define COMBINE_LAUNCH_CASE(num_rdma_ranks) { \
    auto combine_func = low_latency_mode ? \
        combine<true, num_rdma_ranks, nv_bfloat16, kNumCombineForwarderWarps, kNumTMABytesPerWarp> : combine<false, num_rdma_ranks, nv_bfloat16, kNumCombineForwarderWarps, kNumTMABytesPerWarp>; \
    SET_SHARED_MEMORY_FOR_TMA(combine_func); \
    LAUNCH_KERNEL(&cfg, combine_func, \
                  reinterpret_cast<int4*>(combined_x), combined_topk_weights, is_combined_token_in_rank, \
                  reinterpret_cast<const int4*>(x), topk_weights, \
                  reinterpret_cast<const int4*>(bias_0), reinterpret_cast<const int4*>(bias_1), \
                  combined_rdma_head, combined_nvl_head, \
                  reinterpret_cast<const SourceMeta*>(src_meta), rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix, \
                  num_tokens, num_combined_tokens, hidden, num_topk, \
                  rdma_buffer_ptr, num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens, \
                  buffer_ptrs, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens, \
                  rank, num_ranks); } break

    int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    auto num_warps_per_forwarder = std::max(kNumCombineForwarderWarps / num_rdma_ranks, 1);
    int num_forwarder_warps = num_rdma_ranks * num_warps_per_forwarder;
    EP_HOST_ASSERT(num_forwarder_warps > NUM_MAX_NVL_PEERS and num_forwarder_warps % num_rdma_ranks == 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks > std::max(num_max_rdma_chunked_send_tokens, num_max_nvl_chunked_send_tokens));
    EP_HOST_ASSERT(type == CUDA_R_16BF);

    SETUP_LAUNCH_CONFIG(num_channels * 2, (num_forwarder_warps + 1) * 32, stream);
    SWITCH_RDMA_RANKS(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

} // namespace internode

} // namespace deep_ep
