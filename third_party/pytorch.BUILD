package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "torch",
    hdrs = glob(["include/torch/csrc/api/include/torch/**/*.h"]),
    srcs = glob(["include/torch/csrc/api/include/torch/**/*.cpp"]),
    strip_include_prefix = "include/torch/csrc/api/include",
    deps = [
        ":torch_csrc",
        ":torch_lib",
    ],
)

cc_library(
    name = "torch_csrc",
    hdrs = glob(["include/torch/**/*.h"]),
    srcs = glob(["include/torch/**/*.cpp"]),
    strip_include_prefix = "include",
    deps = [
        "c10"
    ],    
    linkopts = ["-pthread"],
)

cc_library(
    name = "c10",
    hdrs = glob(["include/c10/**/*.h"]),
    srcs = glob(["include/c10/**/*.cpp"]),
    strip_include_prefix = "include",
    deps = [
        ":aten"
    ],
)

cc_library(
    name = "aten",
    hdrs = glob(["include/ATen/**/*.h"]),
    srcs = glob(["include/ATen/**/*.cpp"]),
    strip_include_prefix = "include",
)

cc_library(
    name = "torch_lib",
    srcs = glob([
        "lib/*.so*",
        "lib/*.a"
    ]),
)
