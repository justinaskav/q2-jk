# ----------------------------------------------------------------------------
# Copyright (c) 2024, Justinas Kavoliūnas.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import unittest
import pandas as pd
import numpy as np
from q2_jk._methods import parse_taxonomy_string, get_taxonomic_levels, compare_taxonomies


class TestTaxonomyMethods(unittest.TestCase):
    def setUp(self):
        # Create two simple test taxonomy dataframes
        self.tax1_df = pd.DataFrame({
            'Taxon': [
                'k__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; o__Enterobacterales; f__Enterobacteriaceae; g__Escherichia; s__coli',
                'k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales; f__Lactobacillaceae; g__Lactobacillus; s__'
            ]
        }, index=['ASV1', 'ASV2'])
        
        self.tax2_df = pd.DataFrame({
            'Taxon': [
                'k__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; o__Enterobacterales; f__Enterobacteriaceae; g__Escherichia; s__coli',
                'k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales; f__Streptococcaceae; g__Streptococcus; s__'
            ]
        }, index=['ASV1', 'ASV2'])
    
    def test_parse_taxonomy_string(self):
        tax_string = "k__Bacteria; p__Proteobacteria; c__Gamma(0.95); o__Entero; f__; g__; s__"
        expected = ['k__Bacteria', 'p__Proteobacteria', 'c__Gamma', 'o__Entero', 'f__', 'g__', 's__']
        result = parse_taxonomy_string(tax_string)
        self.assertEqual(result, expected)
    
    def test_get_taxonomic_levels(self):
        levels_df = get_taxonomic_levels(self.tax1_df)
        self.assertEqual(levels_df.shape, (2, 7))  # 2 rows, 7 taxonomic levels
        self.assertEqual(list(levels_df.columns), ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'])
        self.assertEqual(levels_df.loc['ASV1', 'Family'], 'f__Enterobacteriaceae')
    
    def test_compare_taxonomies(self):
        summary_stats, conf_matrices, common_diffs = compare_taxonomies(
            self.tax1_df, self.tax2_df, "Test1", "Test2"
        )
        
        # Check summary stats
        self.assertEqual(summary_stats.shape, (7, 4))  # 7 levels, 4 columns
        self.assertEqual(summary_stats.loc['Family', 'Same'], 1)  # One matching family
        self.assertEqual(summary_stats.loc['Family', 'Different'], 1)  # One different family
        
        # Check confusion matrices
        self.assertIn('Family', conf_matrices)
        conf_matrix = conf_matrices['Family']
        self.assertEqual(conf_matrix.loc['f__Enterobacteriaceae', 'f__Enterobacteriaceae'], 1)
        self.assertEqual(conf_matrix.loc['f__Lactobacillaceae', 'f__Streptococcaceae'], 1)


if __name__ == '__main__':
    unittest.main()
