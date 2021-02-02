import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pokergym",
    version="0.0.1",
    author="Trouverun",
    author_email="author@example.com",
    description="A no-limit hold'em environment for training RL agents.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trouverun/pokergym",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU GPLv3 ",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)