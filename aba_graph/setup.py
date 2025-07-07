from setuptools import setup, find_packages

setup(
    name='aba_graph_visualisation',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'py_arg',
        'networkx',
        'matplotlib'
    ],
)
