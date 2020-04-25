load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "main",
    srcs = [
        "main.cpp"
    ],
    deps = [
        "@pytorch_src//:c10_headers",
        "@local_config_cuda//:cuda_headers",
        "@local_config_cuda//:cuda_runtime",
    ],
    linkopts = [
        "-L/usr/local/cuda/lib64/stubs",
    ],
)