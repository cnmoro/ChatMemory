from setuptools import setup, find_packages

setup(
    name='chatmemorydb',
    version='0.3',
    author='Carlo Moro',
    author_email='cnmoro@gmail.com',
    description="Memory",
    packages=find_packages(),
    package_data={
    },
    include_package_data=True,
    install_requires=[
        "mongita",
        "logzero",
        "sumy",
        "langdetect",
        "openai==1.12.0",
        "minivectordb",
        "text-util-en-pt",
        "numpy"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)