# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""setup.py"""
import datetime
import os
import sys

from setuptools import find_packages
from setuptools import setup

name = "timecast"
version = "0.4.0"

if "--nightly" in sys.argv:
    name += "-nightly"

    date_suffix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    version += ".dev" + date_suffix
    sys.argv.remove("--nightly")
    print(name, version)

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"), encoding="utf-8",
) as f:
    long_description = f.read()

setup(
    name=name,
    version=version,
    author="Google AI Princeton",
    author_email="dsuo@google.com",
    description="Performant, composable online learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google/timecast",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    keywords=["timeseries", "time", "series", "analysis", "online", "learning"],
    python_requires=">=3.6",
    install_requires=["numpy", "jax", "jaxlib", "pandas", "pathos", "tqdm"],
    extras_require={
        "dev": [
            "flake8",
            "flake8-print",
            "flake8-bugbear",
            "mypy",
            "scikit-learn",
            "pytest",
            "pytest-cov",
            "pytest-instafail",
            "pytest-xdist",
            "pylint",
            "black",
            "reorder-python-imports",
            "autoflake",
            "pre-commit",
            "pydocstring-coverage",
            "bumpversion",
            "ipython",
            "jupyter",
            "gprof2dot",
            "bandit",
            "check-python-versions",
            "line-profiler",
        ],
        "docs": ["sphinx", "sphinxcontrib-napoleon", "sphinx_rtd_theme"],
    },
)
