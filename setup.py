from setuptools import find_packages, setup

def get_requirements(file_path):
    """
    This function will return the list of requirements.
    """
    with open(file_path) as file:
        requirements = file.readlines()
        list_of_req = [req.replace('\n', '') for req in requirements if req!='-e .']

    return list_of_req

setup(
    name='General_ML_Project_Structure',
    version='0.0.1',
    author='Deepesh Sharma',
    author_email='dsh.2065@gmail.com',
    packages=find_packages(),
    requires=get_requirements('requirements.txt')
)