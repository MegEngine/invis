#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved

import re
import setuptools


def get_install_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        reqs = [x.strip() for x in f.read().splitlines()]
    reqs = [x for x in reqs if not x.startswith("#")]
    return reqs


def get_version():
    with open("invis/__init__.py", "r") as f:
        version = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
            f.read(), re.MULTILINE
        ).group(1)
    return version


def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
    return long_description


setuptools.setup(
    name="invis",
    version=get_version(),
    author="Feng Wang",
    # package_dir=get_package_dir(),
    python_requires=">=3.6",
    install_requires=get_install_requirements(),
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3", "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    packages=setuptools.find_packages(),
)
