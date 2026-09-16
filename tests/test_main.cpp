#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "model.hpp"

// Global pointer accessible by all test files
static std::unique_ptr<MemoryMappedFile> g_mapped_file;
static std::unique_ptr<Model> g_model;

const Model& get_test_model() {
    if (!g_model) {
        throw std::runtime_error("Model was not initialized before running tests");
    }
    return *g_model;
}

int main(int argc, char** argv) {
    doctest::Context context(argc, argv);

    // Default path or passed via first CLI argument
    std::string weights_path = (argc > 1 && argv[1][0] != '-') ? argv[1] : "../../../../weights.bin";

    std::println("=== Initializing Model for Tests from '{}' ===", weights_path);
    g_mapped_file = std::make_unique<MemoryMappedFile>(weights_path);
    g_model = std::make_unique<Model>(*g_mapped_file);

    int res = context.run(); // Run all doctest TEST_CASEs

    return res;
}