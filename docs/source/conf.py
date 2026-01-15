# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information



import os
import sys

sys.path.insert(0, os.path.abspath('../..'))


project = 'Mensa Roboter'
copyright = '2026, Manuel Kaiser, Fabian Kröger'
author = 'Manuel Kaiser, Fabian Kröger'
release = '26.01.2026'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    ]

autodoc_mock_imports = [
    "tensorrt",
    "pycuda",
    "pycuda.driver",
    "pycuda.autoinit",
]


templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


