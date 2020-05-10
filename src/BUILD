package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

cc_binary(
    name = "main",
    srcs = [
        "main.cpp"
    ],

    deps = [
        "@local_config_cuda//cuda"
    ] + if_cuda([
        "@pytorch//:torch"], 
        ["@pytorch_cpu//:torch",
    ]),
    
)
