add_requires("doctest")
add_requires("nlohmann_json 3.12.0")
add_rules("plugin.compile_commands.autoupdate", {outputdir = "."})

target("tokenizers_c_lib")
    set_kind("phony")
    on_build(function (target)
        print("Checking/building tokenizers Rust library...")
        -- Run cargo directly inside the rust subdirectory (no import or os.cd needed)
        os.vrunv("cargo", {"build", "--release"}, {curdir = "3rdparty/tokenizers-cpp/rust"})
    end)

target("llm_engine")
    set_kind("binary")
    set_languages("c++23")

    add_deps("tokenizers_c_lib")

    add_files("src/*.cpp")
    add_files("3rdparty/tokenizers-cpp/src/huggingface_tokenizer.cc")

    
    add_includedirs("include")
    add_includedirs("3rdparty/tokenizers-cpp/include")

    add_linkdirs("3rdparty/tokenizers-cpp/rust/target/release")
    add_links("tokenizers_c", "rocblas")

    add_syslinks("pthread", "dl")

    
    on_load(function (target)
        import("lib.detect.find_tool")

        local hipconfig = find_tool("hipconfig")
        local nvcc = find_tool("nvcc")

        local hip_platform = os.getenv("HIP_PLATFORM")
        if not hip_platform and hipconfig then
            try { function()
                local out = os.iorun(hipconfig.program .. " --platform")
                if out then
                    hip_platform = out:trim():lower()
                end
            end }
        end

        if not hip_platform then
            if nvcc then
                hip_platform = "nvidia"
            else
                hip_platform = "amd"
            end
        end

        if hip_platform == "nvidia" or hip_platform == "nvcc" then
            target:set("toolset", "cc", "nvcc")
            target:set("toolset", "cxx", "nvcc")
            target:set("toolset", "ld", "nvcc")

            target:add("defines", "__HIP_PLATFORM_NVIDIA__", "__HIPCC__")

            local cuda_path = os.getenv("CUDA_PATH") or os.getenv("CUDA_HOME") or "/usr/local/cuda"
            if os.isdir(cuda_path) then
                target:add("includedirs", path.join(cuda_path, "include"))
            end

            if os.isdir("/tmp/hip/include") then
                target:add("includedirs", "/tmp/hip/include")
            end

            target:add("files", "src/kernels/*.hip.cpp", {
                sourcekind = "cu"
            })
        else
            target:set("toolset", "cc", "clang@hipcc")
            target:set("toolset", "cxx", "clang@hipcc")
            target:set("toolset", "ld", "clang@hipcc")

            target:add("defines", "__HIP_PLATFORM_AMD__", "__HIPCC__")

            local rocm_path = os.getenv("ROCM_PATH")
            if not rocm_path and hipconfig then
                try { function()
                    local out = os.iorun(hipconfig.program .. " --path")
                    if out then
                        rocm_path = out:trim()
                    end
                end }
            end
            rocm_path = rocm_path or "/opt/rocm"

            local inc_dir = path.join(rocm_path, "include")
            if os.isdir(inc_dir) then
                target:add("includedirs", inc_dir)
            end

            target:add("files", "src/kernels/*.hip.cpp", {
                cxxflags = {"-x hip", "--offload-arch=native"}
            })
        end

    end)
target_end()


target("test_kernels")
    set_kind("binary")
    set_languages("c++23")
    add_packages("doctest")

    add_deps("tokenizers_c_lib")

    add_files("tests/*.cpp")
    add_files("src/kernels/*.hip.cpp")
    add_files("3rdparty/tokenizers-cpp/src/huggingface_tokenizer.cc")

    add_includedirs("include")
    add_includedirs("tests")
    add_includedirs("3rdparty/tokenizers-cpp/include")

    add_linkdirs("3rdparty/tokenizers-cpp/rust/target/release")
    add_links("tokenizers_c", "rocblas")
    add_syslinks("pthread", "dl")

    add_packages("nlohmann_json")

    -- Reuse your HIP toolset setup
    set_toolset("cc", "clang@hipcc")
    set_toolset("cxx", "clang@hipcc")
    set_toolset("ld", "clang@hipcc")
    add_defines("__HIP_PLATFORM_AMD__", "__HIPCC__")

    local rocm_path = os.getenv("ROCM_PATH") or "/opt/rocm"
    add_includedirs(path.join(rocm_path, "include"))

    add_files("src/kernels/*.hip.cpp", {
        cxxflags = {"-x hip", "--offload-arch=native"}
})


target("benchmark_engine")
    set_kind("binary")
    set_languages("c++23")
    add_deps("tokenizers_c_lib")

    add_files("benches/main_bench.cpp")
    add_files("src/kernels/*.hip.cpp", { cxxflags = {"-x hip", "--offload-arch=native"} })
    add_files("3rdparty/tokenizers-cpp/src/huggingface_tokenizer.cc")

    add_includedirs("include", "benches", "3rdparty/tokenizers-cpp/include", "/opt/rocm/include")
    add_linkdirs("3rdparty/tokenizers-cpp/rust/target/release")
    add_links("tokenizers_c", "rocblas")
    add_syslinks("pthread", "dl")
    add_packages("nlohmann_json")

    set_toolset("cc", "clang@hipcc")
    set_toolset("cxx", "clang@hipcc")
    set_toolset("ld", "clang@hipcc")
    add_defines("__HIP_PLATFORM_AMD__", "__HIPCC__")