import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pokergym",
    version="0.0.1",
    author="Trouverun",
    author_email="aleksi.maki-penttila@tuni.fi",
    description="A no limit hold'em environment for training RL agents.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trouverun/pokergym",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)