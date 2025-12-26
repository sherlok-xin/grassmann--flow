/*
 * Fused CUDA kernels for Grassmann Flow operations.
 *
 * Key optimizations:
 * 1. Fused Plucker computation - compute antisymmetric products in registers
 * 2. Multi-window parallel processing - all windows in one kernel launch
 * 3. Warp-level reductions for normalization
 * 4. Shared memory for window accumulation
 * 5. Fused backward pass with gradient accumulation
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_REDUCED_DIM = 128;  // Max r for Plucker
constexpr int MAX_WINDOWS = 8;

// Helper: atomic add that works for half precision via float conversion
template<typename T>
__device__ __forceinline__ void gpuAtomicAdd(T* address, T val) {
    atomicAdd(address, val);
}

// Specialization for half - convert to float
template<>
__device__ __forceinline__ void gpuAtomicAdd<c10::Half>(c10::Half* address, c10::Half val) {
    // Use the native __half atomicAdd on sm_70+
    atomicAdd(reinterpret_cast<__half*>(address), __float2half(__half2float(*reinterpret_cast<__half*>(&val))));
}

// Helper: warp-level reduction (using float for shuffle operations)
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    float fval = static_cast<float>(val);
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        fval += __shfl_down_sync(0xffffffff, fval, offset);
    }
    return static_cast<T>(fval);
}

// Helper: block-level reduction using shared memory
template<typename T, int BLOCK_SIZE>
__device__ T block_reduce_sum(T val, T* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? shared[lane] : T(0);
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

/*
 * Fused Plucker Forward Kernel
 *
 * Computes Plucker coordinates for all (current, past) pairs across all windows
 * in a single kernel launch.
 *
 * Input:  z [batch, seq_len, reduced_dim] - reduced representations
 * Output: plucker_out [batch, seq_len, plucker_dim] - accumulated Plucker features
 *         counts [batch, seq_len] - number of valid windows per position
 *
 * Each block handles one (batch, position) pair.
 * Threads within a block cooperatively compute Plucker coords for all windows.
 */
template<typename scalar_t>
__global__ void fused_plucker_forward_kernel(
    const scalar_t* __restrict__ z,           // [batch, seq_len, reduced_dim]
    scalar_t* __restrict__ plucker_out,       // [batch, seq_len, plucker_dim]
    int* __restrict__ counts,                 // [batch, seq_len]
    const int* __restrict__ window_sizes,     // [num_windows]
    const int batch_size,
    const int seq_len,
    const int reduced_dim,
    const int plucker_dim,
    const int num_windows,
    const float scale,
    const float eps
) {
    // Each block handles one (batch, position)
    const int batch_idx = blockIdx.x;
    const int pos = blockIdx.y;

    if (batch_idx >= batch_size || pos >= seq_len) return;

    // Shared memory for z vectors and accumulation
    // Layout: [z_current (reduced_dim)] [z_past (reduced_dim)] [plucker_accum (plucker_dim)]
    extern __shared__ char shared_mem[];
    scalar_t* z_current = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* z_past = z_current + reduced_dim;  // Use runtime value, not template
    scalar_t* plucker_accum = z_past + reduced_dim;

    // Load current z into shared memory
    const scalar_t* z_curr_ptr = z + (batch_idx * seq_len + pos) * reduced_dim;
    for (int i = threadIdx.x; i < reduced_dim; i += blockDim.x) {
        z_current[i] = z_curr_ptr[i];
    }
    __syncthreads();

    // Initialize accumulator
    for (int i = threadIdx.x; i < plucker_dim; i += blockDim.x) {
        plucker_accum[i] = scalar_t(0);
    }
    __syncthreads();

    int valid_windows = 0;

    // Process each window
    for (int w = 0; w < num_windows; w++) {
        int delta = window_sizes[w];
        int past_pos = pos - delta;

        if (past_pos < 0) continue;
        valid_windows++;

        // Load past z into shared memory
        const scalar_t* z_past_ptr = z + (batch_idx * seq_len + past_pos) * reduced_dim;
        for (int i = threadIdx.x; i < reduced_dim; i += blockDim.x) {
            z_past[i] = z_past_ptr[i];
        }
        __syncthreads();

        // Compute Plucker coordinates: p_ij = u_i * v_j - u_j * v_i for i < j
        // Each thread handles multiple Plucker indices
        int plucker_idx = threadIdx.x;
        while (plucker_idx < plucker_dim) {
            // Convert linear index to (i, j) pair where i < j
            // Using triangular number formula: idx = i * (2*r - i - 1) / 2 + (j - i - 1)
            // Approximate i from idx
            int i = 0, j = 0;
            int remaining = plucker_idx;
            for (i = 0; i < reduced_dim - 1; i++) {
                int count = reduced_dim - i - 1;
                if (remaining < count) {
                    j = i + 1 + remaining;
                    break;
                }
                remaining -= count;
            }

            // Compute antisymmetric product
            scalar_t p = z_current[i] * z_past[j] - z_current[j] * z_past[i];
            p *= scale;

            // Accumulate
            gpuAtomicAdd(&plucker_accum[plucker_idx], p);

            plucker_idx += blockDim.x;
        }
        __syncthreads();
    }

    // Write output (average over windows)
    if (valid_windows > 0) {
        scalar_t inv_count = scalar_t(1.0f / valid_windows);
        scalar_t* out_ptr = plucker_out + (batch_idx * seq_len + pos) * plucker_dim;

        // Compute L2 norm for normalization
        scalar_t norm_sq = scalar_t(0);
        for (int i = threadIdx.x; i < plucker_dim; i += blockDim.x) {
            scalar_t val = plucker_accum[i] * inv_count;
            norm_sq += val * val;
        }

        // Reduce norm across threads
        __shared__ scalar_t shared_norm[32];
        __shared__ scalar_t broadcast_norm;
        norm_sq = block_reduce_sum<scalar_t, 256>(norm_sq, shared_norm);

        // Broadcast reduced norm to all threads
        if (threadIdx.x == 0) {
            broadcast_norm = rsqrtf(static_cast<float>(norm_sq) + eps);
        }
        __syncthreads();

        scalar_t inv_norm = broadcast_norm;

        // Write normalized output
        for (int i = threadIdx.x; i < plucker_dim; i += blockDim.x) {
            out_ptr[i] = plucker_accum[i] * inv_count * inv_norm;
        }
    }

    // Write count
    if (threadIdx.x == 0) {
        counts[batch_idx * seq_len + pos] = valid_windows;
    }
}

/*
 * Fused Grassmann Mixing Forward Kernel
 *
 * Combines:
 * 1. Linear reduction (model_dim -> reduced_dim)
 * 2. Plucker computation across all windows
 * 3. Projection back to model_dim
 * 4. Gated residual connection
 *
 * This eliminates all intermediate tensors.
 */
template<typename scalar_t>
__global__ void fused_grassmann_forward_kernel(
    const scalar_t* __restrict__ hidden_states,  // [batch, seq_len, model_dim]
    const scalar_t* __restrict__ W_reduce,       // [reduced_dim, model_dim]
    const scalar_t* __restrict__ W_proj,         // [model_dim, plucker_dim]
    const scalar_t* __restrict__ W_gate,         // [model_dim, model_dim]
    const scalar_t* __restrict__ b_gate,         // [model_dim]
    scalar_t* __restrict__ output,               // [batch, seq_len, model_dim]
    const int* __restrict__ window_sizes,        // [num_windows]
    const int batch_size,
    const int seq_len,
    const int model_dim,
    const int reduced_dim,
    const int plucker_dim,
    const int num_windows,
    const float scale,
    const float eps,
    const float output_scale
) {
    // Each block handles one (batch, position)
    const int batch_idx = blockIdx.x;
    const int pos = blockIdx.y;

    if (batch_idx >= batch_size || pos >= seq_len) return;

    extern __shared__ char shared_mem[];
    scalar_t* z_current = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* z_past = z_current + reduced_dim;
    scalar_t* plucker_accum = z_past + reduced_dim;
    scalar_t* geo_features = plucker_accum + plucker_dim;
    scalar_t* hidden_local = geo_features + model_dim;

    const scalar_t* h_ptr = hidden_states + (batch_idx * seq_len + pos) * model_dim;

    // Load hidden state
    for (int i = threadIdx.x; i < model_dim; i += blockDim.x) {
        hidden_local[i] = h_ptr[i];
    }
    __syncthreads();

    // Step 1: Compute z_current = W_reduce @ hidden (reduction)
    for (int i = threadIdx.x; i < reduced_dim; i += blockDim.x) {
        scalar_t sum = 0;
        for (int j = 0; j < model_dim; j++) {
            sum += W_reduce[i * model_dim + j] * hidden_local[j];
        }
        z_current[i] = sum;
    }
    __syncthreads();

    // Initialize Plucker accumulator
    for (int i = threadIdx.x; i < plucker_dim; i += blockDim.x) {
        plucker_accum[i] = 0;
    }
    __syncthreads();

    int valid_windows = 0;

    // Step 2: Process each window
    for (int w = 0; w < num_windows; w++) {
        int delta = window_sizes[w];
        int past_pos = pos - delta;

        if (past_pos < 0) continue;
        valid_windows++;

        // Compute z_past
        const scalar_t* h_past = hidden_states + (batch_idx * seq_len + past_pos) * model_dim;
        for (int i = threadIdx.x; i < reduced_dim; i += blockDim.x) {
            scalar_t sum = 0;
            for (int j = 0; j < model_dim; j++) {
                sum += W_reduce[i * model_dim + j] * h_past[j];
            }
            z_past[i] = sum;
        }
        __syncthreads();

        // Compute and accumulate Plucker coordinates
        for (int idx = threadIdx.x; idx < plucker_dim; idx += blockDim.x) {
            int i = 0, j = 0;
            int remaining = idx;
            for (i = 0; i < reduced_dim - 1; i++) {
                int count = reduced_dim - i - 1;
                if (remaining < count) {
                    j = i + 1 + remaining;
                    break;
                }
                remaining -= count;
            }

            scalar_t p = z_current[i] * z_past[j] - z_current[j] * z_past[i];
            plucker_accum[idx] += p * scale;
        }
        __syncthreads();
    }

    // Step 3: Average and project to model_dim
    for (int i = threadIdx.x; i < model_dim; i += blockDim.x) {
        geo_features[i] = 0;
    }
    __syncthreads();

    if (valid_windows > 0) {
        scalar_t inv_count = scalar_t(1.0f / valid_windows);

        // Project: geo = W_proj @ (plucker_accum / count)
        for (int i = threadIdx.x; i < model_dim; i += blockDim.x) {
            scalar_t sum = 0;
            for (int j = 0; j < plucker_dim; j++) {
                sum += W_proj[i * plucker_dim + j] * plucker_accum[j] * inv_count;
            }
            geo_features[i] = sum;
        }
    }
    __syncthreads();

    // Step 4: Gated residual: output = hidden + sigmoid(gate) * geo * scale
    scalar_t* out_ptr = output + (batch_idx * seq_len + pos) * model_dim;
    for (int i = threadIdx.x; i < model_dim; i += blockDim.x) {
        // Compute gate
        scalar_t gate_val = b_gate[i];
        for (int j = 0; j < model_dim; j++) {
            gate_val += W_gate[i * model_dim + j] * hidden_local[j];
        }
        scalar_t gate = 1.0f / (1.0f + expf(-gate_val));  // sigmoid

        out_ptr[i] = hidden_local[i] + gate * geo_features[i] * output_scale;
    }
}

/*
 * Backward kernel for Plucker + projection
 *
 * Computes gradients for:
 * - z (reduced representations)
 * - W_reduce
 * - W_proj
 * - W_gate, b_gate
 */
template<typename scalar_t>
__global__ void fused_grassmann_backward_kernel(
    const scalar_t* __restrict__ grad_output,    // [batch, seq_len, model_dim]
    const scalar_t* __restrict__ hidden_states,  // [batch, seq_len, model_dim]
    const scalar_t* __restrict__ z,              // [batch, seq_len, reduced_dim]
    const scalar_t* __restrict__ plucker,        // [batch, seq_len, plucker_dim]
    const scalar_t* __restrict__ geo_features,   // [batch, seq_len, model_dim]
    const scalar_t* __restrict__ W_reduce,       // [reduced_dim, model_dim]
    const scalar_t* __restrict__ W_proj,         // [model_dim, plucker_dim]
    const scalar_t* __restrict__ W_gate,         // [model_dim, model_dim]
    const scalar_t* __restrict__ b_gate,         // [model_dim]
    scalar_t* __restrict__ grad_hidden,          // [batch, seq_len, model_dim]
    scalar_t* __restrict__ grad_W_reduce,        // [reduced_dim, model_dim] (atomic)
    scalar_t* __restrict__ grad_W_proj,          // [model_dim, plucker_dim] (atomic)
    scalar_t* __restrict__ grad_W_gate,          // [model_dim, model_dim] (atomic)
    scalar_t* __restrict__ grad_b_gate,          // [model_dim] (atomic)
    const int* __restrict__ window_sizes,
    const int batch_size,
    const int seq_len,
    const int model_dim,
    const int reduced_dim,
    const int plucker_dim,
    const int num_windows,
    const float scale,
    const float output_scale
) {
    const int batch_idx = blockIdx.x;
    const int pos = blockIdx.y;

    if (batch_idx >= batch_size || pos >= seq_len) return;

    extern __shared__ char shared_mem[];
    scalar_t* grad_out_local = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* hidden_local = grad_out_local + model_dim;
    scalar_t* geo_local = hidden_local + model_dim;
    scalar_t* grad_geo = geo_local + model_dim;
    scalar_t* grad_plucker = grad_geo + model_dim;
    scalar_t* grad_z = grad_plucker + plucker_dim;

    const int offset = (batch_idx * seq_len + pos);

    // Load into shared memory
    for (int i = threadIdx.x; i < model_dim; i += blockDim.x) {
        grad_out_local[i] = grad_output[offset * model_dim + i];
        hidden_local[i] = hidden_states[offset * model_dim + i];
        geo_local[i] = geo_features[offset * model_dim + i];
    }
    __syncthreads();

    // Backward through gated residual
    // output = hidden + sigmoid(gate) * geo * scale
    // grad_hidden += grad_output (residual path)
    // grad_geo = grad_output * sigmoid(gate) * scale
    // grad_gate = grad_output * geo * scale * sigmoid(gate) * (1 - sigmoid(gate))

    for (int i = threadIdx.x; i < model_dim; i += blockDim.x) {
        // Recompute gate
        scalar_t gate_val = b_gate[i];
        for (int j = 0; j < model_dim; j++) {
            gate_val += W_gate[i * model_dim + j] * hidden_local[j];
        }
        scalar_t sig = 1.0f / (1.0f + expf(-gate_val));

        scalar_t g_out = grad_out_local[i];

        // Gradient for geo features
        grad_geo[i] = g_out * sig * output_scale;

        // Gradient for gate
        scalar_t grad_gate_i = g_out * geo_local[i] * output_scale * sig * (1.0f - sig);

        // Accumulate gradients for W_gate and b_gate
        gpuAtomicAdd(&grad_b_gate[i], grad_gate_i);
        for (int j = 0; j < model_dim; j++) {
            gpuAtomicAdd(&grad_W_gate[i * model_dim + j], grad_gate_i * hidden_local[j]);
        }

        // Gradient for hidden (residual + gate path)
        scalar_t grad_h = g_out;  // residual
        for (int k = 0; k < model_dim; k++) {
            scalar_t gk_val = b_gate[k];
            for (int j = 0; j < model_dim; j++) {
                gk_val += W_gate[k * model_dim + j] * hidden_local[j];
            }
            scalar_t sig_k = 1.0f / (1.0f + expf(-gk_val));
            scalar_t grad_gate_k = grad_out_local[k] * geo_local[k] * output_scale * sig_k * (1.0f - sig_k);
            grad_h += grad_gate_k * W_gate[k * model_dim + i];
        }
        grad_hidden[offset * model_dim + i] = grad_h;
    }
    __syncthreads();

    // Backward through projection: geo = W_proj @ plucker
    // grad_plucker = W_proj^T @ grad_geo
    // grad_W_proj += grad_geo @ plucker^T
    for (int i = threadIdx.x; i < plucker_dim; i += blockDim.x) {
        scalar_t sum = 0;
        for (int j = 0; j < model_dim; j++) {
            sum += W_proj[j * plucker_dim + i] * grad_geo[j];
        }
        grad_plucker[i] = sum;
    }
    __syncthreads();

    // Gradient for W_proj
    for (int i = threadIdx.x; i < model_dim; i += blockDim.x) {
        for (int j = 0; j < plucker_dim; j++) {
            gpuAtomicAdd(&grad_W_proj[i * plucker_dim + j],
                      grad_geo[i] * plucker[offset * plucker_dim + j]);
        }
    }
    __syncthreads();

    // Backward through Plucker computation to z
    // This requires iterating over windows and computing gradients
    const scalar_t* z_curr = z + offset * reduced_dim;

    for (int i = threadIdx.x; i < reduced_dim; i += blockDim.x) {
        grad_z[i] = 0;
    }
    __syncthreads();

    int valid_windows = 0;
    for (int w = 0; w < num_windows; w++) {
        if (pos - window_sizes[w] >= 0) valid_windows++;
    }

    if (valid_windows > 0) {
        scalar_t inv_count = 1.0f / valid_windows;

        for (int w = 0; w < num_windows; w++) {
            int delta = window_sizes[w];
            int past_pos = pos - delta;
            if (past_pos < 0) continue;

            const scalar_t* z_past = z + (batch_idx * seq_len + past_pos) * reduced_dim;

            // Gradient for z_current from Plucker
            // p_ij = z_current[i] * z_past[j] - z_current[j] * z_past[i]
            // grad_z_current[i] += grad_p_ij * z_past[j] (for all j > i)
            // grad_z_current[i] -= grad_p_ij * z_past[j] (for all j < i, where (j,i) is the pair)

            for (int i = threadIdx.x; i < reduced_dim; i += blockDim.x) {
                scalar_t grad_z_i = 0;

                for (int j = 0; j < reduced_dim; j++) {
                    if (i == j) continue;

                    int pi, pj;
                    scalar_t sign;
                    if (i < j) {
                        pi = i; pj = j;
                        sign = 1.0f;
                    } else {
                        pi = j; pj = i;
                        sign = -1.0f;
                    }

                    // Convert (pi, pj) to linear index
                    int plucker_idx = pi * (2 * reduced_dim - pi - 1) / 2 + (pj - pi - 1);

                    if (plucker_idx < plucker_dim) {
                        grad_z_i += sign * grad_plucker[plucker_idx] * z_past[j] * scale * inv_count;
                    }
                }

                gpuAtomicAdd(&grad_z[i], grad_z_i);
            }
        }
    }
    __syncthreads();

    // Backward through reduction: z = W_reduce @ hidden
    // grad_hidden += W_reduce^T @ grad_z
    // grad_W_reduce += grad_z @ hidden^T
    for (int i = threadIdx.x; i < model_dim; i += blockDim.x) {
        scalar_t sum = 0;
        for (int j = 0; j < reduced_dim; j++) {
            sum += W_reduce[j * model_dim + i] * grad_z[j];
        }
        gpuAtomicAdd(&grad_hidden[offset * model_dim + i], sum);
    }

    for (int i = threadIdx.x; i < reduced_dim; i += blockDim.x) {
        for (int j = 0; j < model_dim; j++) {
            gpuAtomicAdd(&grad_W_reduce[i * model_dim + j], grad_z[i] * hidden_local[j]);
        }
    }
}

// Wrapper functions for Python binding
torch::Tensor fused_plucker_forward(
    torch::Tensor z,
    torch::Tensor window_sizes,
    float scale,
    float eps
) {
    const int batch_size = z.size(0);
    const int seq_len = z.size(1);
    const int reduced_dim = z.size(2);
    const int plucker_dim = reduced_dim * (reduced_dim - 1) / 2;
    const int num_windows = window_sizes.size(0);

    auto plucker_out = torch::zeros({batch_size, seq_len, plucker_dim}, z.options());
    auto counts = torch::zeros({batch_size, seq_len}, z.options().dtype(torch::kInt32));

    dim3 blocks(batch_size, seq_len);
    int threads = 256;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(z.scalar_type(), "fused_plucker_forward", [&] {
        // Calculate shared memory size based on actual scalar type
        int shared_size = (2 * reduced_dim + plucker_dim) * sizeof(scalar_t);

        fused_plucker_forward_kernel<scalar_t><<<blocks, threads, shared_size>>>(
            z.data_ptr<scalar_t>(),
            plucker_out.data_ptr<scalar_t>(),
            counts.data_ptr<int>(),
            window_sizes.data_ptr<int>(),
            batch_size, seq_len, reduced_dim, plucker_dim, num_windows,
            scale, eps
        );
    });

    return plucker_out;
}

torch::Tensor fused_grassmann_forward(
    torch::Tensor hidden_states,
    torch::Tensor W_reduce,
    torch::Tensor W_proj,
    torch::Tensor W_gate,
    torch::Tensor b_gate,
    torch::Tensor window_sizes,
    float scale,
    float eps,
    float output_scale
) {
    const int batch_size = hidden_states.size(0);
    const int seq_len = hidden_states.size(1);
    const int model_dim = hidden_states.size(2);
    const int reduced_dim = W_reduce.size(0);
    const int plucker_dim = W_proj.size(1);
    const int num_windows = window_sizes.size(0);

    auto output = torch::empty_like(hidden_states);

    dim3 blocks(batch_size, seq_len);
    int threads = 256;
    int shared_size = (2 * reduced_dim + plucker_dim + 3 * model_dim) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(hidden_states.scalar_type(), "fused_grassmann_forward", [&] {
        fused_grassmann_forward_kernel<scalar_t><<<blocks, threads, shared_size>>>(
            hidden_states.data_ptr<scalar_t>(),
            W_reduce.data_ptr<scalar_t>(),
            W_proj.data_ptr<scalar_t>(),
            W_gate.data_ptr<scalar_t>(),
            b_gate.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            window_sizes.data_ptr<int>(),
            batch_size, seq_len, model_dim, reduced_dim, plucker_dim, num_windows,
            scale, eps, output_scale
        );
    });

    return output;
}

std::vector<torch::Tensor> fused_grassmann_backward(
    torch::Tensor grad_output,
    torch::Tensor hidden_states,
    torch::Tensor z,
    torch::Tensor plucker,
    torch::Tensor geo_features,
    torch::Tensor W_reduce,
    torch::Tensor W_proj,
    torch::Tensor W_gate,
    torch::Tensor b_gate,
    torch::Tensor window_sizes,
    float scale,
    float output_scale
) {
    const int batch_size = hidden_states.size(0);
    const int seq_len = hidden_states.size(1);
    const int model_dim = hidden_states.size(2);
    const int reduced_dim = z.size(2);
    const int plucker_dim = plucker.size(2);
    const int num_windows = window_sizes.size(0);

    auto grad_hidden = torch::zeros_like(hidden_states);
    auto grad_W_reduce = torch::zeros_like(W_reduce);
    auto grad_W_proj = torch::zeros_like(W_proj);
    auto grad_W_gate = torch::zeros_like(W_gate);
    auto grad_b_gate = torch::zeros_like(b_gate);

    dim3 blocks(batch_size, seq_len);
    int threads = 256;
    int shared_size = (4 * model_dim + plucker_dim + reduced_dim) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(hidden_states.scalar_type(), "fused_grassmann_backward", [&] {
        fused_grassmann_backward_kernel<scalar_t><<<blocks, threads, shared_size>>>(
            grad_output.data_ptr<scalar_t>(),
            hidden_states.data_ptr<scalar_t>(),
            z.data_ptr<scalar_t>(),
            plucker.data_ptr<scalar_t>(),
            geo_features.data_ptr<scalar_t>(),
            W_reduce.data_ptr<scalar_t>(),
            W_proj.data_ptr<scalar_t>(),
            W_gate.data_ptr<scalar_t>(),
            b_gate.data_ptr<scalar_t>(),
            grad_hidden.data_ptr<scalar_t>(),
            grad_W_reduce.data_ptr<scalar_t>(),
            grad_W_proj.data_ptr<scalar_t>(),
            grad_W_gate.data_ptr<scalar_t>(),
            grad_b_gate.data_ptr<scalar_t>(),
            window_sizes.data_ptr<int>(),
            batch_size, seq_len, model_dim, reduced_dim, plucker_dim, num_windows,
            scale, output_scale
        );
    });

    return {grad_hidden, grad_W_reduce, grad_W_proj, grad_W_gate, grad_b_gate};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_plucker_forward", &fused_plucker_forward, "Fused Plucker forward");
    m.def("fused_grassmann_forward", &fused_grassmann_forward, "Fused Grassmann forward");
    m.def("fused_grassmann_backward", &fused_grassmann_backward, "Fused Grassmann backward");
}
