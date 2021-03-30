from distutils.core import setup
setup(
  name = 'aztools', 
  packages = ['aztools'],
  version = '0.1.0',
  license='MIT',
  description = ('Tools for spectral/timing analysis in X-ray astronomy'),
  author = 'Abdu Zoghbi',
  author_email = 'astrozoghbi@gmail.com',
  url = 'https://zoghbi-a.github.io/aztools/',
  download_url = 'https://github.com/zoghbi-a/aztools/archive/refs/tags/0.1.1.tar.gz',
  keywords = ['Astronomy', 'Time-Series', 'X-ray', 'Spectroscopy'],
  install_requires=[ 
          'scipy',
          'astropy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers', 
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3', 
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
