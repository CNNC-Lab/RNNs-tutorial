from setuptools import setup, find_packages

setup(
    name="rnn_dynamical_systems",
    version="0.1.0",
    description="Tutorial: RNNs as Computational Dynamical Systems",
    author="Renato Duarte",
    author_email="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "torch>=2.0.0",
        "torchdiffeq>=0.2.3",
        "norse>=1.0.0",
    ],
)
