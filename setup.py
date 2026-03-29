from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str)-> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # FIX 1: Use 'requirements' (the list), not 'requirements.txt'
        # FIX 2: Re-assign it back to 'requirements' (spelled correctly)
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements

setup(
    name="mlproject",
    version="0.1.0",
    author="hassan",
    author_email="hassanmustafa5550@gmail.com", # FIX 3: Removed extra 'r'
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)