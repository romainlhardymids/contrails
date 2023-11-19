from setuptools import setup, find_packages  
setup(
    name="contrails", 
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True
)