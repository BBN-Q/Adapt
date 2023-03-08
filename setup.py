from setuptools import setup

setup(
    name='bbnadapt',
    version='2019.1',
    author='Graham Rowlands',
    package_dir={'':'src'},
    packages=[
        'adapt',
    ],
    scripts=[],
    url='https://github.com/BBN-Q/adapt',
    download_url='https://github.com/BBN-Q/adapt',
    license="Apache 2.0 License",
    description='Adaptive meshing for fun and profit.',
    long_description='Adaptive meshing for fun and profit.',
    install_requires=[
        "numpy >= 1.12.1",
        "scipy >= 0.17.1"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    keywords="adaptive refinement "
)