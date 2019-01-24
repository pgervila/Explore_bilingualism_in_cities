from setuptools import setup, find_packages

setup(name='bilangcity',
      python_requires='>=3.6',
      version='0.1',
      description='Library to quantify and analyze language choice in bilingual cities using Twitter data',
      url='https://github.com/pgervila/Explore_bilingualism_in_cities',
      author='Paolo Gervasoni Vila',
      author_email='pgervila@gmail.com',
      packages=find_packages(),
      install_requires=['tweepy', 'numpy', 'pandas', 'matplotlib',
                        'langdetect', 'statsmodels', 'pyprind'],
      include_package_data=True,
      package_data={'bilangcity': 'data/*.h5'}
      )
__author__ = 'Paolo Gervasoni Vila'