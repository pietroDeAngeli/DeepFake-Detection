from setuptools import setup, find_packages

setup(
    name="deepfake-tools",
    version="0.1.0", 
    description="Deepfake detection tools",
    author="Pietro De Angeli",
    author_email="pietro.deangeli@studenti.unitn.it",
    url="https://github.com/pdeangeli/deepfake-tools",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[ 
        "av==15.0.0", 
        "matplotlib", 
        "seaborn",
        "tqdm",
        "opencv-python",
        "mtcnn",
        "tensorflow",
        "opencv-python-headless>=4.5.5",
        "torch"
    ],
    python_requires=">=3.10",
)