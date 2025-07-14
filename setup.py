from setuptools import setup, find_packages

setup(
    name="deepfake-tools",                      # Nome del pacchetto generale
    version="0.1.0",                             # Versione iniziale
    description="Deepfake detection tools",      # Descrizione breve
    author="Pietro De Angeli",
    author_email="pietro.deangeli@studenti.unitn.it",
    url="https://github.com/pdeangeli/deepfake-tools",  # Opzionale
    package_dir={"": "src"},                   # Directory di root per i package
    packages=find_packages(where="src"),         # Include tutti i package sotto src/
    install_requires=[                            # Dipendenze esterne
        "av==15.0.0",                            # Uso di == per la versione
        "matplotlib",                            # Puoi specificare range di versioni se serve
        "seaborn",
        "tqdm",                                  # Corretto da "tdqm"
        "opencv-python",
        "mtcnn",
        "tensorflow",
    ],
    python_requires=">=3.10",                    # Versione minima di Python
)