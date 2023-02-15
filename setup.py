from setuptools import setup

import csnet

setup(
    name='csnet',
    version=csnet.__version__,
    url='https://github.com/amedyukhina/CSNet',
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['csnet',
              'csnet.models',
              'csnet.losses',
              'csnet.utils',
              'csnet.transforms'
              ],
    license='MIT',
    include_package_data=True,

    test_suite='csnet.tests',

    install_requires=[
        'ipykernel',
        'scipy',
        'numpy',
        'pytest',
        'wandb',
        'tqdm',
        'torch>=1.10',
        'matplotlib',
        'scikit-image',
        'monai',
        'tensorboard',
        'cs_sim @ git+https://github.com/amedyukhina/CS-Sim.git',
    ],
)
