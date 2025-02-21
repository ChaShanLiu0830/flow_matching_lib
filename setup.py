from setuptools import setup, find_packages

setup(
    name='flow_matching_lib',
    version='0.3',
    description='Modular Conditional Flow Matching for Deep Generative Modeling',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'beartype>=0.10.0'  # Added beartype requirement
    ],
) 