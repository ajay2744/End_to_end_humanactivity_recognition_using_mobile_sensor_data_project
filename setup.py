from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    
    
    with open('requirements.txt') as file_obj:
        requirements=file_obj.readlines()
    requirements=[req.replace("\n","") for req in requirements]
    if "-e ." in requirements:
        requirements.remove("-e .")
    return requirements



setup(name="Human activity recoginition using mobile sensor data",version="0.0.1",
      author="AJAY KUMAR SADHU",author_email="ajay.2744@gmail.com",packages=find_packages(),
      install_packages=get_requirements('requirements.txt'))