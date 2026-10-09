#include "gpu_utils.hpp"
#include "model_inference.hpp"
#include "model.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <print>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Run {
    double prefill_ms;
    double generation_ms;
    std::vector<int> generated;
};

// Include launches, GPU execution, and the token transfer to the host.
Run generate(ModelInference& engine, std::vector<int> tokens, size_t count) {
    Run run{};
    auto start = Clock::now();
    int next = engine.prefill(tokens);

    run.prefill_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    run.generated.push_back(next);
    tokens.push_back(next);

    start = Clock::now();
    for (size_t step = 1; step < count; ++step) {
        next = engine.decode(next); 
        run.generated.push_back(next);
        tokens.push_back(next);
    }
    run.generation_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    return run;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 5 || argc > 7)
            throw std::runtime_error("Usage: benchmark_engine <weights.bin> <max_new_tokens> <output_json> <input_tokens_json> [repeats=3] [warmups=1]");

        const size_t count = std::stoul(argv[2]);
        const size_t repeats = argc > 5 ? std::stoul(argv[5]) : 3;
        const size_t warmups = argc > 6 ? std::stoul(argv[6]) : 1;

        if (!count || !repeats || !warmups)
            throw std::runtime_error("Token count, repeats, and warmups must be positive");

        std::ifstream input(argv[4]);
        const auto tokens = nlohmann::json::parse(input).get<std::vector<int>>();

        MemoryMappedFile file(argv[1]);
        Model model(file);

        if (tokens.empty() || tokens.size() > model.config.max_seq_len ||
            count - 1 > model.config.max_seq_len - tokens.size())
            throw std::runtime_error("Prompt plus generation exceeds the context window");

        for (int token : tokens)
            if (token < 0 || static_cast<size_t>(token) >= model.config.vocab_size)
                throw std::runtime_error("Input token is outside this model's vocabulary");

        ModelInference engine(model);
        std::println("[C++] {} prompt tokens; warming up {} full generation runs", tokens.size(), warmups);

        for (size_t i = 0; i < warmups; ++i) generate(engine, tokens, count);

        std::vector<double> prefill_times, generation_times;

        Run last{};

        for (size_t i = 0; i < repeats; ++i) {
            last = generate(engine, tokens, count);
            prefill_times.push_back(last.prefill_ms);
            generation_times.push_back(last.generation_ms);
        }

        const double prefill_ms = median(prefill_times);
        const double generation_ms = median(generation_times);

        nlohmann::json result = {
            {"engine", "cpp_hip"}, {"uses_kv_cache", false},
            {"prompt_tokens", tokens.size()}, {"input_token_ids", tokens},
            {"generated_tokens", last.generated.size()}, {"generated_token_ids", last.generated},
            {"repeats", repeats}, {"warmups", warmups},
            {"prefill_time_ms", prefill_ms},
            {"prefill_tokens_per_sec", tokens.size() * 1000.0 / prefill_ms},
            {"generation_after_first_ms", count > 1 ? generation_ms : 0.0},
            {"generation_after_first_tokens_per_sec", count > 1 ? (count - 1) * 1000.0 / generation_ms : 0.0},
            {"prefill_samples_ms", prefill_times}, {"generation_samples_ms", generation_times}
        };

        std::ofstream output(argv[3]);
        output << result.dump(4);

        if (!output) throw std::runtime_error("Failed to write benchmark results");

        std::println("[C++] Results written to {}", argv[3]);

    } catch (const std::exception& e) {
        std::println(stderr, "[C++ Error] {}", e.what());
        return 1;
    }
}
