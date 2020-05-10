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

load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")
cuda_configure(name = "local_config_cuda")

http_archive(
    name = "pytorch",
    sha256 = "2ba21afa9c05d794bf0936b54b9c2d54e0532677fe0c27ec01b4331625521c19",
    strip_prefix = "libtorch",
    urls = [
        "https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.5.0%2Bcu101.zip",
    ],
    build_file = "//third_party:pytorch.BUILD",
)

http_archive(
    name = "pytorch_cpu",
    sha256 = "2ba21afa9c05d794bf0936b54b9c2d54e0532677fe0c27ec01b4331625521c19",
    strip_prefix = "libtorch",
    urls = [
        "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip",
    ],
    build_file = "//third_party:pytorch.BUILD",
)
