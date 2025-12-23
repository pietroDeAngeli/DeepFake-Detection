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
        "matplotlib==3.10.3", 
        "seaborn==0.13.2",
        "tqdm==4.67.1",
        "opencv-python==4.12.0.88",
        "albumentations",
        #"PyQt6"
        #"mtcnn",
        #"tensorflow==2.19.0",
        #"opencv-python-headless>=4.5.5",
        #"torch==2.7.1",
        #"torchvision"
    ],
    python_requires=">=3.10",
)
