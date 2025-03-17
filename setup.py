from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='f1_project',
      version="0.0.1",
      description="f1_model",
      license="ékip_license",
      author="ékip",
    #   url="https://github.com/Constance310/F1_project_1907",
      install_requires=requirements,
      #packages=find_packages(include=["f1_project", "f1_project.*"]))
      packages=find_packages())
