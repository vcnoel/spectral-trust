# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="spectral_trust",
    version="0.4.0rc1",
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
    license="AGPL-3.0-or-later",
    license_files=["LICENSE"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)
