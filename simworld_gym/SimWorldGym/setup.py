from setuptools import setup

setup(
    name='simworld_gym',
    version='0.1',
    install_requires=[
        "simworld @ git+https://github.com/SimWorld-AI/SimWorld.git@main",
        "gym==0.26.2",
        "numpy==1.26.4",
        "pygame==2.6.0", 
        "openai", 
        "osmnx",
        "python-dotenv", 
        "matplotlib",
    ],
    python_requires='>=3.10'
)