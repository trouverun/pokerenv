import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pokerenv",
    version="0.1.4",
    author="Trouverun",
    author_email="aleksi.maki-penttila@tuni.fi",
    description="A no limit hold'em environment for training RL agents.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trouverun/pokerenv",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)