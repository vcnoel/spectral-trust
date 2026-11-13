# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="spectral_trust",
    version="0.4.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"spectral_trust.cli": ["data/*.json"]},
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tqdm",
        "accelerate"
    ],
    # The core install imports neither TransformerLens nor nnsight; the
    # adapters in spectral_trust.interop read their tensors duck-typed.
    extras_require={
        "tl": ["transformer_lens>=2.0"],
        "nnsight": ["nnsight>=0.4"],
        # The SVG views need nothing beyond numpy; the heatmap's matplotlib
        # fallback uses Colormap.with_extremes (matplotlib 3.4).
        "viz": ["matplotlib>=3.4"],
        "docs": [
            "mkdocs>=1.6",
            "mkdocs-material>=9.5",
            "mkdocstrings[python]>=0.26",
        ],
        "dev": [
            "pytest",
            "build",
            "twine",
            # readme_renderer 46 renders Markdown with comrak, which has no
            # recent Windows wheels; 44 uses cmarkgfm.
            "readme_renderer[md]; platform_system != 'Windows'",
            "readme_renderer[md]<45; platform_system == 'Windows'",
            "nbclient",
            "nbformat",
            "ipykernel",
        ],
    },
    entry_points={
        "console_scripts": [
            "gsp-cli=spectral_trust.cli:main",
        ],
    },
    author="Valentin Noël",
    author_email="val.noel@proton.me",
    description="Graph-spectral diagnostics for transformer attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vcnoel/spectral-trust",
    project_urls={
        "Documentation": "https://vcnoel.github.io/spectral-trust/",
        "Source": "https://github.com/vcnoel/spectral-trust",
        "Changelog": "https://github.com/vcnoel/spectral-trust/blob/main/CHANGELOG.md",
    },
    license="Apache-2.0",
    license_files=["LICENSE"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)
