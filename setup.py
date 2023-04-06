from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

__version__ = '0.0.1'


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


def fetch_readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


ext_modules = [
    CppExtension(name='elixir.c_utils',
                 sources=['src/simulator.cpp'],
                 extra_compile_args=['-O3', '-DVERSION_GE_1_1', '-DVERSION_GE_1_3', '-DVERSION_GE_1_5'])
]

setup(
    name='elixir',
    version=__version__,
    author='Haichen Huang',
    author_email='c2h214748@gmail.com',
    url='https://github.com/hpcaitech/Elixir',
    packages=find_packages(exclude=(
        'example',
        'profile',
        'src',
        'test',
        '*.egg-info',
    )),
    description='An Optimized Implementation of Elixir (Gemini2.0)',
    long_description=fetch_readme(),
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=fetch_requirements('requirements.txt'),
    python_requires='>=3.8',
)
