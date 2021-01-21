package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "assets",
    srcs = [
        "assets/franka_panda_headers.xml",
        "assets/franka_panda.xml",
        "assets/desk.xml",
    ] + glob([
        "assets/meshes/**/*.stl",
        "assets/textures/*.png",
        "assets/meshes/*.stl",
    ]),
)

py_library(
    name = "robodesk",
    srcs = ["robodesk.py"],
    data = [
        ":assets",
    ],
    deps = [
        "//third_party/py/PIL:pil",
        "//third_party/py/dm_control:control",
        "//third_party/py/dm_control/utils:inverse_kinematics",
        "//third_party/py/dm_control/utils:transformations",
        "//third_party/py/gym",
    ],
)

py_library(
    name = "envs",
    srcs = ["__init__.py"],
    deps = [
        "//third_party/py/gym",
    ],
)

py_test(
    name = "robodesk_test",
    srcs = ["robodesk_test.py"],
    python_version = "PY3",
    deps = [
        ":robodesk",
        "//testing/pybase",
    ],
)
