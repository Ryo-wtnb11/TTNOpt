from setuptools import find_packages, setup

setup(
    name="ttnopt",
    version="0.1.0",
    description="A Python package for tree tensor network algorithms",
    author="Ryo Watanabe, Hidetaka Manabe",
    author_email="manabe@acs.i.kyoto-u.ac.jp",
    url="https://github.com/Watayo/TTNOpt",
    packages=find_packages(),
    zip_safe=False,
    python_requires=">=3.6",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        "console_scripts": [
            "gss=ttnopt:ground_state_search",  # Link ttnopt_gss command to your main function
            "ft=ttnopt:factorize_tensor",  # Link ttnopt_gss command to your main function
            "samplettn=ttnopt:sample",  # Link ttnopt_sample command to your sample function
        ],
    },
    install_requires=[
        "numpy>=1.23.5",
        "jaxlib<0.4.34",
        "tensornetwork",
        "networkx",
        "pydot",
        "pyyaml",
        "dotmap",
        "matplotlib",
        "scipy",
        "pandas",
    ],
)
