from setuptools import setup, find_packages

setup(
    name='LangSegment',
    version='0.3.5',
    description='LangSegment language segmentation library',
    packages=find_packages(include=['LangSegment', 'LangSegment.*']),
    include_package_data=True,
    zip_safe=False,
)
