from setuptools import setup


setup(
    name='skylibs',
    description=('Tools to read, write, perform projections and handle LDR/HDR environment maps (IBL).'),
    author='Yannick Hold',
    author_email='yannickhold@gmail.com',
    license="LGPLv3",
    url='https://github.com/soravux/skylibs',
    version='0.6.6',
    packages=['ezexr', 'envmap', 'hdrio', 'hdrtools', 'hdrtools/tonemapping', 'skydb', 'tools3d'],
    include_package_data=True,
    install_requires=['imageio>=1.6', 'tqdm', 'numpy', 'scipy'],
)
