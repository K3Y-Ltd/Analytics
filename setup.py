from setuptools import setup, find_packages

setup(
    name="utils",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Add your project dependencies here, for example:
        # "requests>=2.25.1",
    ],
    entry_points={
        'console_scripts': [
            # Define command-line scripts here, for example:
            # 'your_script_name=your_package.module:function',
        ],
    },
    python_requires='>=3.8',
)
