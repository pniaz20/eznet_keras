from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

VERSION = '1.1.3' 
DESCRIPTION = "Easily build Keras models: utils for training/testing, built-in ANN, CNN, RNN models, modular Dense and Convolutional blocks, etc."

# Setting up
setup(
        # the name must match the folder name
        name="eznet_keras", 
        version=VERSION,
        author="Pouya P. Niaz",
        author_email="<pniaz20@ku.edu.tr>",
        url='https://github.com/pniaz20/eznet_keras',
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=find_packages(),
        # packages=find_packages('eznet_keras'),
        # package_dir={'': 'eznet_keras'},
        python_requires=">=3.7, <4",
        license='MIT',
        install_requires=[
            'numpy','matplotlib','tensorflow(>=2.4)'
        ],
        keywords=['tensorflow','keras','deep learning','neural network','keras2cpp'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: End Users/Desktop",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
            "Topic :: Utilities",
            "Topic :: Education",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Embedded Systems"
        ]
)