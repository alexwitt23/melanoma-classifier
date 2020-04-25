workspace(name = "melanoma_classifier")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

http_archive(
    name = "pytorch_src",
    urls = [
        "https://github.com/alexwitt2399/melanoma-classifier/releases/download/v0.0.1-alpha/pytorch.zip",
    ],
    sha256 = "60a5b63b83f53beb7563bf156c678a6fa38e70e45ec6bb62d1e725b28a19a2a1",
    strip_prefix = "pytorch",
)

load("//third_party:cuda_configure.bzl", "cuda_configure")
cuda_configure(name = "local_config_cuda")