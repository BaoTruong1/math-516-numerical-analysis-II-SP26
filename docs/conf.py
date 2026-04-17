import os
import sys
# Point Sphinx to the src directory so it can import the modules
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'Numerical Analysis Library'
copyright = '2026, M. Tezzele and Students'
author = 'M. Tezzele and Students'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Automatically pull docstrings
    'sphinx.ext.napoleon',     # Parse Google-style docstrings (Args, Returns)
    'sphinx.ext.viewcode',     # Add links to highlighted source code
    'sphinx.ext.mathjax',      # Render math formulas
]

# Napoleon settings to handle your specific docstring formatting
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme' # The classic, highly readable ReadTheDocs theme
