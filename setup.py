from setuptools import setup, find_packages

setup(
    name="marine-microplastics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "streamlit>=1.10.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "geopandas>=0.10.0",
        "plotly>=5.3.0",
        "imblearn>=0.10.0",
    ],
    author="Marine Research Team",
    author_email="contact@example.com",
    description="A project for analyzing marine microplastics using machine learning",
    keywords="marine, microplastics, machine learning, environmental science",
    url="https://github.com/username/marine-microplastics",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Environmental Science",
    ],
    python_requires=">=3.8",
)
