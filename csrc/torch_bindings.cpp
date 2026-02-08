#include "registration.h"

#include <torch/library.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

struct LDGLayerWeights {
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
};

extern "C" void launch_ldg_decode(
    int input_token_id, int* output_token_id,
    const void* embed_weight, const LDGLayerWeights* layer_weights,
    const void* final_norm_weight, const void* lm_head_weight,
    const void* cos_table, const void* sin_table,
    void* k_cache, void* v_cache,
    void* hidden_buffer, void* g_activations, void* g_residual,
    void* g_q, void* g_k, void* g_v, void* g_attn_out,
    void* g_mlp_intermediate, void* g_normalized,
    void* block_max_vals, void* block_max_idxs,
    int num_layers, int position, int cache_len, int max_seq_len,
    float attn_scale, cudaStream_t stream);

extern "C" void launch_ldg_decode_with_logits(
    int input_token_id, int* output_token_id, float* logits_output,
    const void* embed_weight, const LDGLayerWeights* layer_weights,
    const void* final_norm_weight, const void* lm_head_weight,
    const void* cos_table, const void* sin_table,
    void* k_cache, void* v_cache,
    void* hidden_buffer, void* g_activations, void* g_residual,
    void* g_q, void* g_k, void* g_v, void* g_attn_out,
    void* g_mlp_intermediate, void* g_normalized,
    void* block_max_vals, void* block_max_idxs,
    int num_layers, int position, int cache_len, int max_seq_len,
    float attn_scale, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Op: decode one token.
//   layer_weights_packed: uint8 tensor of LDGLayerWeights structs on device.
//   output_token: int32[1] tensor written by the kernel.
// ---------------------------------------------------------------------------
void decode(
    torch::Tensor output_token,       // int32[1]  (output)
    int64_t input_token_id,
    torch::Tensor embed_weight,
    torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor hidden_buffer,
    torch::Tensor activations,
    torch::Tensor residual,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor attn_out,
    torch::Tensor mlp_intermediate,
    torch::Tensor normalized,
    torch::Tensor block_max_vals,
    torch::Tensor block_max_idxs,
    int64_t num_layers,
    int64_t position,
    int64_t max_seq_len,
    double attn_scale
) {
    launch_ldg_decode(
        static_cast<int>(input_token_id),
        static_cast<int*>(output_token.data_ptr()),
        embed_weight.data_ptr(),
        reinterpret_cast<const LDGLayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(),
        lm_head_weight.data_ptr(),
        cos_table.data_ptr(),
        sin_table.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        hidden_buffer.data_ptr(),
        activations.data_ptr(),
        residual.data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        attn_out.data_ptr(),
        mlp_intermediate.data_ptr(),
        normalized.data_ptr(),
        block_max_vals.data_ptr(),
        block_max_idxs.data_ptr(),
        static_cast<int>(num_layers),
        static_cast<int>(position),
        static_cast<int>(position + 1),  // cache_len = position + 1
        static_cast<int>(max_seq_len),
        static_cast<float>(attn_scale),
        c10::cuda::getCurrentCUDAStream().stream());
}

// ---------------------------------------------------------------------------
// Op: persistent decode (non-cooperative, faster atomic barriers)
// ---------------------------------------------------------------------------
extern "C" void launch_ldg_decode_persistent(
    int input_token_id, int* output_token_id,
    const void* embed_weight, const LDGLayerWeights* layer_weights,
    const void* final_norm_weight, const void* lm_head_weight,
    const void* cos_table, const void* sin_table,
    void* k_cache, void* v_cache,
    void* hidden_buffer, void* g_activations, void* g_residual,
    void* g_q, void* g_k, void* g_v, void* g_attn_out,
    void* g_mlp_intermediate, void* g_normalized,
    void* block_max_vals, void* block_max_idxs,
    int num_layers, int position, int cache_len, int max_seq_len,
    float attn_scale, cudaStream_t stream);

void decode_persistent(
    torch::Tensor output_token, int64_t input_token_id,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor cos_table, torch::Tensor sin_table,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor hidden_buffer, torch::Tensor activations, torch::Tensor residual,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor attn_out,
    torch::Tensor mlp_intermediate, torch::Tensor normalized,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    int64_t num_layers, int64_t position, int64_t max_seq_len, double attn_scale
) {
    launch_ldg_decode_persistent(
        static_cast<int>(input_token_id),
        static_cast<int*>(output_token.data_ptr()),
        embed_weight.data_ptr(),
        reinterpret_cast<const LDGLayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        cos_table.data_ptr(), sin_table.data_ptr(),
        k_cache.data_ptr(), v_cache.data_ptr(),
        hidden_buffer.data_ptr(), activations.data_ptr(), residual.data_ptr(),
        q.data_ptr(), k.data_ptr(), v.data_ptr(), attn_out.data_ptr(),
        mlp_intermediate.data_ptr(), normalized.data_ptr(),
        block_max_vals.data_ptr(), block_max_idxs.data_ptr(),
        static_cast<int>(num_layers), static_cast<int>(position),
        static_cast<int>(position + 1), static_cast<int>(max_seq_len),
        static_cast<float>(attn_scale),
        c10::cuda::getCurrentCUDAStream().stream());
}

// ---------------------------------------------------------------------------
// Graph-accelerated generate: capture once, replay per step.
// ---------------------------------------------------------------------------

// Access the pinned host memory from the launch function
extern int* h_pinned_position;
extern int* h_pinned_token_id;

torch::Tensor generate_graph(
    int64_t first_token_id, int64_t num_steps,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor cos_table, torch::Tensor sin_table,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor hidden_buffer, torch::Tensor activations, torch::Tensor residual,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor attn_out,
    torch::Tensor mlp_intermediate, torch::Tensor normalized,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    int64_t num_layers, int64_t start_position, int64_t max_seq_len,
    double attn_scale
) {
    auto output_token = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto result = torch::empty({num_steps}, torch::dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int* d_out = static_cast<int*>(output_token.data_ptr());
    int* h_result = static_cast<int*>(result.data_ptr());

    // Warmup (ensures all allocations are done)
    int tok = static_cast<int>(first_token_id);
    int pos = static_cast<int>(start_position);
    launch_ldg_decode_persistent(tok, d_out,
        embed_weight.data_ptr(),
        reinterpret_cast<const LDGLayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        cos_table.data_ptr(), sin_table.data_ptr(),
        k_cache.data_ptr(), v_cache.data_ptr(),
        hidden_buffer.data_ptr(), activations.data_ptr(), residual.data_ptr(),
        q.data_ptr(), k.data_ptr(), v.data_ptr(), attn_out.data_ptr(),
        mlp_intermediate.data_ptr(), normalized.data_ptr(),
        block_max_vals.data_ptr(), block_max_idxs.data_ptr(),
        static_cast<int>(num_layers), pos, pos + 1,
        static_cast<int>(max_seq_len), static_cast<float>(attn_scale), stream);
    cudaStreamSynchronize(stream);

    // Capture graph
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launch_ldg_decode_persistent(tok, d_out,
        embed_weight.data_ptr(),
        reinterpret_cast<const LDGLayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        cos_table.data_ptr(), sin_table.data_ptr(),
        k_cache.data_ptr(), v_cache.data_ptr(),
        hidden_buffer.data_ptr(), activations.data_ptr(), residual.data_ptr(),
        q.data_ptr(), k.data_ptr(), v.data_ptr(), attn_out.data_ptr(),
        mlp_intermediate.data_ptr(), normalized.data_ptr(),
        block_max_vals.data_ptr(), block_max_idxs.data_ptr(),
        static_cast<int>(num_layers), pos, pos + 1,
        static_cast<int>(max_seq_len), static_cast<float>(attn_scale), stream);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

    // Generate tokens using graph replay
    tok = static_cast<int>(first_token_id);
    pos = static_cast<int>(start_position);
    for (int step = 0; step < num_steps; step++) {
        // Update pinned memory (graph will memcpy these to device)
        *h_pinned_token_id = tok;
        *h_pinned_position = pos;

        // Replay graph
        cudaGraphLaunch(graph_exec, stream);

        // Copy output token back
        cudaMemcpyAsync(&h_result[step], d_out, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        tok = h_result[step];
        pos++;
    }

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);

    return result;
}

// ---------------------------------------------------------------------------
// Op: decode one token and also output full logits (for verification).
// ---------------------------------------------------------------------------
void decode_with_logits(
    torch::Tensor output_token,       // int32[1]
    torch::Tensor logits,             // float32[vocab_size]
    int64_t input_token_id,
    torch::Tensor embed_weight,
    torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor hidden_buffer,
    torch::Tensor activations,
    torch::Tensor residual,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor attn_out,
    torch::Tensor mlp_intermediate,
    torch::Tensor normalized,
    torch::Tensor block_max_vals,
    torch::Tensor block_max_idxs,
    int64_t num_layers,
    int64_t position,
    int64_t max_seq_len,
    double attn_scale
) {
    launch_ldg_decode_with_logits(
        static_cast<int>(input_token_id),
        static_cast<int*>(output_token.data_ptr()),
        static_cast<float*>(logits.data_ptr()),
        embed_weight.data_ptr(),
        reinterpret_cast<const LDGLayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(),
        lm_head_weight.data_ptr(),
        cos_table.data_ptr(),
        sin_table.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        hidden_buffer.data_ptr(),
        activations.data_ptr(),
        residual.data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        attn_out.data_ptr(),
        mlp_intermediate.data_ptr(),
        normalized.data_ptr(),
        block_max_vals.data_ptr(),
        block_max_idxs.data_ptr(),
        static_cast<int>(num_layers),
        static_cast<int>(position),
        static_cast<int>(position + 1),
        static_cast<int>(max_seq_len),
        static_cast<float>(attn_scale),
        c10::cuda::getCurrentCUDAStream().stream());
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def(
        "decode(Tensor output_token, int input_token_id, "
        "Tensor embed_weight, Tensor layer_weights_packed, "
        "Tensor final_norm_weight, Tensor lm_head_weight, "
        "Tensor cos_table, Tensor sin_table, "
        "Tensor k_cache, Tensor v_cache, "
        "Tensor hidden_buffer, Tensor activations, Tensor residual, "
        "Tensor q, Tensor k, Tensor v, Tensor attn_out, "
        "Tensor mlp_intermediate, Tensor normalized, "
        "Tensor block_max_vals, Tensor block_max_idxs, "
        "int num_layers, int position, int max_seq_len, "
        "float attn_scale) -> ()");
    ops.impl("decode", torch::kCUDA, &decode);

    ops.def(
        "decode_with_logits(Tensor output_token, Tensor logits, "
        "int input_token_id, "
        "Tensor embed_weight, Tensor layer_weights_packed, "
        "Tensor final_norm_weight, Tensor lm_head_weight, "
        "Tensor cos_table, Tensor sin_table, "
        "Tensor k_cache, Tensor v_cache, "
        "Tensor hidden_buffer, Tensor activations, Tensor residual, "
        "Tensor q, Tensor k, Tensor v, Tensor attn_out, "
        "Tensor mlp_intermediate, Tensor normalized, "
        "Tensor block_max_vals, Tensor block_max_idxs, "
        "int num_layers, int position, int max_seq_len, "
        "float attn_scale) -> ()");
    ops.impl("decode_with_logits", torch::kCUDA, &decode_with_logits);

    ops.def(
        "decode_persistent(Tensor output_token, int input_token_id, "
        "Tensor embed_weight, Tensor layer_weights_packed, "
        "Tensor final_norm_weight, Tensor lm_head_weight, "
        "Tensor cos_table, Tensor sin_table, "
        "Tensor k_cache, Tensor v_cache, "
        "Tensor hidden_buffer, Tensor activations, Tensor residual, "
        "Tensor q, Tensor k, Tensor v, Tensor attn_out, "
        "Tensor mlp_intermediate, Tensor normalized, "
        "Tensor block_max_vals, Tensor block_max_idxs, "
        "int num_layers, int position, int max_seq_len, "
        "float attn_scale) -> ()");
    ops.impl("decode_persistent", torch::kCUDA, &decode_persistent);

    ops.def(
        "generate_graph(int first_token_id, int num_steps, "
        "Tensor embed_weight, Tensor layer_weights_packed, "
        "Tensor final_norm_weight, Tensor lm_head_weight, "
        "Tensor cos_table, Tensor sin_table, "
        "Tensor k_cache, Tensor v_cache, "
        "Tensor hidden_buffer, Tensor activations, Tensor residual, "
        "Tensor q, Tensor k, Tensor v, Tensor attn_out, "
        "Tensor mlp_intermediate, Tensor normalized, "
        "Tensor block_max_vals, Tensor block_max_idxs, "
        "int num_layers, int start_position, int max_seq_len, "
        "float attn_scale) -> Tensor");
    ops.impl("generate_graph", torch::kCUDA, &generate_graph);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
