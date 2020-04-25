load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")

package(default_visibility = ["//visibility:public"])

# Group the sources of the library so that CMake rule have access to it
all_content = """filegroup(name = "all", srcs = glob(["**"]), visibility = ["//visibility:public"])"""


cmake_external(
    name = "pytorch_cpp",
    cache_entries = {
        "CMAKE_PREFIX_PATH": "."
    },
    lib_source = all_content
)
