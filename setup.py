from setuptools import setup, find_packages

setup(name='multiagent_particle_envs',
      version='0.0.1',
      description='Multi-Agent Goal-Driven Communication Environment',
      url='https://github.com/Abddo9/multiagent-particle-envs',
      author='Abdalwhab Bakheet Mohamed Abdalwhab',
      author_email='07ino@gmail.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
