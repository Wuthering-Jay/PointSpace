import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars

opt_vars = get_config_vars("OPT")
if opt_vars and opt_vars[0] is not None:
    opt = opt_vars[0]
    os.environ["OPT"] = " ".join(
        flag for flag in opt.split() if flag != "-Wstrict-prototypes"
    )

src = "src"
sources = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(src)
    for file in files
    if file.endswith(".cpp") or file.endswith(".cu")
]

setup(
    name="pointops",
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["pointops"],
    package_dir={"pointops": "functions"},
    ext_modules=[
        CUDAExtension(
            name="pointops._C",
            sources=sources,
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
