from setuptools import find_packages, setup

setup(
    name="dialogue-generation-models",
    version="0.0.1",
    description="Pingpong Dialogue Generation Models",
    install_requires=[
        "torch==1.6.0",
        "sentencepiece==0.1.91",
        "tensorflow==2.5.2",
        "transformers==3.1.0",
    ],
    url="https://github.com/pingpong-ai/dialogue-generation-models.git",
    author="ScatterLab",
    author_email="developers@scatterlab.co.kr",
    packages=find_packages(exclude=["tests"]),
    license="MIT License",
)
