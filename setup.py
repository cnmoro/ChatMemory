from setuptools import setup, find_packages

setup(
    name='chatmemorydb-lmdb-onnx',
    version='1.2',
    author='Carlo Moro',
    author_email='cnmoro@gmail.com',
    description="Memory",
    packages=find_packages(),
    package_data={},
    include_package_data=True,
    install_requires=[
        "numpy",
        "lmdb",
        "msgpack",
        "pymongo",
        "onnxruntime",
        "onnxruntime_extensions",
        "faiss-cpu"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)