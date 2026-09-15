add_requires("nlohmann_json")
add_rules("plugin.compile_commands.autoupdate", {outputdir = "."})

target("llm_engine")
    set_kind("binary")
    set_languages("c++23")

    add_files("src/*.cpp")
    add_packages("nlohmann_json")

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

        target:add("includedirs", "include")
    end)
target_end()
