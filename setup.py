from setuptools import setup, find_packages

setup(
    name="spectral_trust",
    version="0.1.0",
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
    author="Google DeepMind (simulated)",
    description="Spectral diagnostics for trust in LLMs",
    python_requires=">=3.8",
)
