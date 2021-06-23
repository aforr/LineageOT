import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lineageot",
    version="0.1.0",
    author="Aden Forrow",
    author_email="aden.forrow@maths.ox.ac.uk",
    description="LineageOT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aforr/LineageOT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
