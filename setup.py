from setuptools import setup, find_packages

install_requires = [
        'tensorboard',
        'scikit-image',
        'scikit-learn',
        'opencv-python',
        'tqdm',
        'h5py',
        'transforms3d'
        ]

setup(name='Fusion2D3DMUNEGC',
      description="Official implementation of the paper 2D-3D Geometric Fusion using Multi-neighbourhood Graph Convolution for RGB-D Indoor Scene Classification. https://www.sciencedirect.com/science/article/pii/S1566253521001032",
      version='1.0.0',
      author='AlbertMosellaMontoro',
      install_requires=install_requires,
      packages=find_packages(),
      )
