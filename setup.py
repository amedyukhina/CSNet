from setuptools import setup

import csnet

setup(
    name='csnet',
    version=csnet.__version__,
    url='https://github.com/amedyukhina/CSNet',
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['csnet',
              ],
    license='MIT',
    include_package_data=True,

    test_suite='csnet.tests',

    install_requires=[
        'scipy',
        'numpy',
        'pytest',
        'wandb',
        'torch>=1.10',
        'matplotlib',
        'scikit-image',
    ],
)
