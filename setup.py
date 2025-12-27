from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="spectral_trust",
    version="0.1.2",
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
