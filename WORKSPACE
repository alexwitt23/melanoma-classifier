workspace(name = "melanoma_classifier")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

all_content = """filegroup(name = "all", srcs = glob(["**"]), visibility = ["//visibility:public"])"""


# Rule repository
http_archive(
   name = "rules_foreign_cc",
   strip_prefix = "rules_foreign_cc-master",
   url = "https://github.com/bazelbuild/rules_foreign_cc/archive/master.zip",
)

load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")

http_archive(
    name = "pytorch_cpp_lib",
    urls = [
        "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip",
    ],
    strip_prefix = "libtorch",
    sha256 = "30501ad7277f76ab32133529c59d8eb3a87e7aaaa4c85e8b9e1ecdc186cab19b",
    build_file_content = all_content,
)
