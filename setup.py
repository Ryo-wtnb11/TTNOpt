from setuptools import setup, find_packages

setup(
    name="ttnopt",
    version="0.1.0",
    description="A Python package for tree tensor network algorithms",
    author="Ryo Watanabe, Hidetaka Manabe",
    author_email="manabe@acs.i.kyoto-u.ac.jp",
    url="https://github.com/Watayo/TTNOpt",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[],
    python_requires=">=3.6",
    license='Apache License 2.0',
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: Apache Software License',
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        'console_scripts': [
            'gss=ttnopt:ground_state_search',  # Link ttnopt_gss command to your main function
            'samplettn=ttnopt:sample',  # Link ttnopt_sample command to your sample function
        ],
    },
)