from setuptools import setup, find_packages
from pathlib import Path
import subprocess
import sys

# Function to read the requirements.txt file
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Safely read the README.md file
def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""

# Function to install PyTorch with CUDA support
def install_pytorch():
    try:
        print("Installing PyTorch with CUDA 12.4 support...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu124",
                "torch~=2.5.1",
            ]
        )
        print("PyTorch with CUDA 12.4 installed successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to install PyTorch with CUDA 12.4 support. Exiting.")
        sys.exit(1)


# Ensure PyTorch is installed before proceeding
install_pytorch()

setup(
    name="nn-dataset",
    version="1.0.1",
    description="Neural Network Dataset",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="ABrain One and contributors",
    author_email="AI@ABrain.one",
    url="https://github.com/ABrain-One/nn-dataset",
    packages=find_packages(include=["ab.*"]),
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    include_package_data=True,
)

