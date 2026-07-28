{
    files = {
        "src/main.cpp"
    },
    depfiles = "build/.objs/llm_engine/linux/x86_64/release/src/main.cpp.o: src/main.cpp   include/gpu_utils.hpp include/model_config.hpp   include/kernels/kernels.hpp\
",
    depfiles_format = "gcc",
    values = {
        "hipcc",
        {
            "-Qunused-arguments",
            "-m64",
            "-std=c++20",
            "-I/opt/rocm/include",
            "-Iinclude",
            "-D__HIP_PLATFORM_AMD__",
            "-D__HIPCC__"
        }
    }
}