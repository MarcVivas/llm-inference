#include "gpu_timer.hpp"
#include "gpu_utils.hpp"
#include "model.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <print>
#include <vector>

struct BenchmarkConfig {
    std::string weights_path;
    std::string prompt;
    size_t max_new_tokens = 128;
    std::string output_json = "benches/cpp_bench.json";
};

struct BenchmarkMetrics {
    size_t prompt_tokens = 0;
    size_t generated_tokens = 0;
    float prefill_time_ms = 0.0f;
    std::vector<float> step_times_ms;
};

// Parse command line arguments
BenchmarkConfig parse_cli_args(int argc, char* argv[]) {
    BenchmarkConfig config;
    if (argc < 2) {
        throw std::runtime_error("Usage: benchmark_engine <weights.bin> [max_new_tokens] [output_json]");
    }
    config.weights_path = argv[1];
    if (argc >= 3) config.max_new_tokens = std::stoul(argv[2]);
    if (argc >= 4) config.output_json = argv[3];

    config.prompt = "<|im_start|>user\nExplain GPU memory bandwidth and latency in high performance computing.<|im_end|>\n<|im_start|>assistant\n";
    return config;
}

// Warm up GPU kernels to eliminate driver startup overhead
void warmup_gpu(Model& model) {
    std::println("[C++] Warming up GPU kernels...");
    std::vector<int> dummy_tokens = {151644, 872, 198, 151645}; // Short sequence
    // Note: Call your prefill forward pass here once implemented
    HIP_CHECK(hipDeviceSynchronize());
}

// Measure TTFT (Prefill phase)
float measure_prefill(Model& model, const std::vector<int>& tokens) {
    GpuTimer timer;
    timer.start();

    // Note: Call forward pass on the prompt sequence:
    // run_prefill_forward(model, tokens);

    return timer.stop_and_sync();
}

// Measure per-token autoregressive decoding
std::vector<float> measure_decode_loop(Model& model, size_t max_tokens) {
    GpuTimer timer;
    std::vector<float> step_latencies;
    step_latencies.reserve(max_tokens);

    for (size_t step = 0; step < max_tokens; ++step) {
        timer.start();

        // Note: Call single-token decode forward pass:
        // int next_token = run_decode_step(model);

        float ms = timer.stop_and_sync();
        step_latencies.push_back(ms);
    }

    return step_latencies;
}

// Serialize metrics to structured JSON
void export_metrics_json(const BenchmarkMetrics& m, const std::string& path) {
    float total_decode_ms = std::accumulate(m.step_times_ms.begin(), m.step_times_ms.end(), 0.0f);
    float avg_decode_ms = m.step_times_ms.empty() ? 0.0f : (total_decode_ms / m.step_times_ms.size());
    float decode_tok_per_sec = (avg_decode_ms > 0.0f) ? (1000.0f / avg_decode_ms) : 0.0f;
    float prefill_tok_per_sec = (m.prefill_time_ms > 0.0f) ? ((m.prompt_tokens / m.prefill_time_ms) * 1000.0f) : 0.0f;

    nlohmann::json root = {
        {"engine", "cpp_hip"},
        {"prompt_tokens", m.prompt_tokens},
        {"generated_tokens", m.generated_tokens},
        {"prefill_time_ms", m.prefill_time_ms},
        {"prefill_tokens_per_sec", prefill_tok_per_sec},
        {"avg_decode_time_ms", avg_decode_ms},
        {"decode_tokens_per_sec", decode_tok_per_sec},
        {"step_times_ms", m.step_times_ms}
    };

    std::ofstream file(path);
    file << root.dump(4);
    std::println("[C++] Benchmark results written to: {}", path);
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkConfig config = parse_cli_args(argc, argv);

        MemoryMappedFile file(config.weights_path);
        Model model(file);

        std::vector<int> prompt_tokens = model.tokenizer->Encode(config.prompt);
        std::println("[C++] Prompt tokenized ({} tokens)", prompt_tokens.size());

        warmup_gpu(model);

        BenchmarkMetrics metrics;
        metrics.prompt_tokens = prompt_tokens.size();
        metrics.generated_tokens = config.max_new_tokens;

        std::println("[C++] Measuring prefill...");
        metrics.prefill_time_ms = measure_prefill(model, prompt_tokens);

        std::println("[C++] Measuring decode ({} tokens)...", config.max_new_tokens);
        metrics.step_times_ms = measure_decode_loop(model, config.max_new_tokens);

        export_metrics_json(metrics, config.output_json);

    } catch (const std::exception& e) {
        std::println(stderr, "[C++ Error] {}", e.what());
        return 1;
    }

    return 0;
}