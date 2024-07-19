from setuptools import setup, find_packages

setup(
    name='chatmemorydb',
    version='1.3',
    author='Carlo Moro',
    author_email='cnmoro@gmail.com',
    description="Memory",
    packages=find_packages(),
    package_data={
        "memory": ["resources/*"]
    },
    include_package_data=True,
    install_requires=[
        "minivectordb>=2.2.1",
        "logzero",
        "numpy<2",
        "nltk",
        "scikit-learn",
        "tiktoken",
        "fasttext"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)