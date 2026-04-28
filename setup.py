from setuptools import setup, find_packages

setup(
    name="pac-classifier",
    version="0.1.0",
    description="Purifying Archetype Classifier (PAC) - A Supervised Error-Driven Clustering Algorithm",
    author="Mario Raúl Carbonell Martínez",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "matplotlib>=3.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
