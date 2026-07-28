{
    files = {
        "src/kernels/kernels.hip.cpp"
    },
    depfiles = "build/.objs/llm_engine/linux/x86_64/release/src/kernels/kernels.hip.cpp.o:   src/kernels/kernels.hip.cpp include/kernels/kernels.hpp   include/gpu_utils.hpp\
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
            "-D__HIPCC__",
            "-x",
            "hip",
            "--offload-arch=native"
        }
    }
}