# ----------------------------------------------------------------------------
# Copyright (c) 2024, Justinas Kavoliūnas.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import Citations, Plugin, Str
from q2_types.feature_data import FeatureData, Taxonomy
from q2_types.feature_table import Frequency
from q2_jk import __version__
from q2_jk._methods import compare_taxonomies
from q2_jk._visualizer import visualize_taxonomy_comparison

citations = Citations.load("citations.bib", package="q2_jk")

plugin = Plugin(
    name="jk",
    version=__version__,
    website="https://github.com/justinaskav/q2-jk",
    package="q2_jk",
    description="Just my methods, may extract later.",
    short_description="Misc. tools.",
    # The plugin-level citation of 'Caporaso-Bolyen-2024' is provided as
    # an example. You can replace this with citations to other references
    # in citations.bib.
    citations=[citations['Caporaso-Bolyen-2024']]
)

# Register the taxonomy comparison visualizer
plugin.visualizers.register_function(
    function=visualize_taxonomy_comparison,
    inputs={
        'tax1': FeatureData[Taxonomy],
        'tax2': FeatureData[Taxonomy]
    },
    parameters={
        'tax1_name': Str,
        'tax2_name': Str
    },
    input_descriptions={
        'tax1': 'First taxonomy table to compare.',
        'tax2': 'Second taxonomy table to compare.'
    },
    parameter_descriptions={
        'tax1_name': 'Name for the first taxonomy method (default: "Taxonomy 1")',
        'tax2_name': 'Name for the second taxonomy method (default: "Taxonomy 2")'
    },
    name='Compare taxonomy classifications',
    description='Compares two taxonomy classifications and visualizes the differences.',
    citations=[]
)
