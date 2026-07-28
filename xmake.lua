-- xmake.lua (Replaces CMakeLists.txt and Makefile)
target("llm_engine")
    set_kind("binary")
    set_languages("c++20")

    -- Tell xmake to use AMD's hipcc compiler
    set_toolchains("hip")

    -- Add all C++ and HIP files automatically
    add_files("src/*.cpp")
    add_files("src/kernels/*.hip")
    add_includedirs("include")

    -- Need a JSON library? xmake downloads & links it automatically
    add_requires("nlohmann_json")
    add_packages("nlohmann_json")
