## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    name="block_recognition",
    description="Recognize blocks from a downward facing depth camera",
    packages=['block_recognition'],
    package_dir={'': 'src'})

setup(**setup_args)