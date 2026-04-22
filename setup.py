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
    version="0.2.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
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
    entry_points={
        "console_scripts": [
            "gsp-cli=spectral_trust.cli:main",
        ],
    },
    author="Valentin Noël",
    author_email="val.noel@proton.me",
    description="Spectral diagnostics for trust in LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)
