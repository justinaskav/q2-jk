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
from q2_jk._methods import parse_taxonomy_string, get_taxonomic_levels, compare_taxonomies, normalize_taxonomy_assignment


class TestTaxonomyMethods(unittest.TestCase):
    def setUp(self):
        # Create two simple test taxonomy dataframes with various edge cases
        self.tax1_df = pd.DataFrame({
            'Taxon': [
                'k__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; o__Enterobacterales; f__Enterobacteriaceae; g__Escherichia; s__coli',
                'k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales; f__Lactobacillaceae; g__Lactobacillus; s__',
                'k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales; f__Family XI; g__; s__',
                'k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales; f__Incertae_Sedis; g__Unknown; s__'
            ]
        }, index=['ASV1', 'ASV2', 'ASV3', 'ASV4'])
        
        self.tax2_df = pd.DataFrame({
            'Taxon': [
                'k__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; o__Enterobacterales; f__Enterobacteriaceae; g__Escherichia; s__coli',
                'k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales; f__Streptococcaceae; g__Streptococcus; s__',
                'k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales; f__Family_XI; g__; s__',
                'k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales; f__; g__Unknown; s__'
            ]
        }, index=['ASV1', 'ASV2', 'ASV3', 'ASV4'])
    
    def test_normalize_taxonomy_assignment(self):
        # Test empty and NA values
        self.assertEqual(normalize_taxonomy_assignment(''), '')
        self.assertEqual(normalize_taxonomy_assignment(None), '')
        self.assertEqual(normalize_taxonomy_assignment(np.nan), '')
        
        # Test rank prefixes
        self.assertEqual(normalize_taxonomy_assignment('g__'), '')
        self.assertEqual(normalize_taxonomy_assignment('s__'), '')
        self.assertEqual(normalize_taxonomy_assignment('f__'), '')
        
        # Test rank prefixes with content
        self.assertEqual(normalize_taxonomy_assignment('g__Escherichia'), 'escherichia')
        self.assertEqual(normalize_taxonomy_assignment('f__Enterobacteriaceae'), 'enterobacteriaceae')
        
        # Test Incertae Sedis variations
        self.assertEqual(normalize_taxonomy_assignment('Incertae Sedis'), '')
        self.assertEqual(normalize_taxonomy_assignment('Incertae_Sedis'), '')
        self.assertEqual(normalize_taxonomy_assignment('incertae sedis'), '')
        self.assertEqual(normalize_taxonomy_assignment('INCERTAE_SEDIS'), '')
        
        # Test underscore and space normalization
        self.assertEqual(normalize_taxonomy_assignment('Family XI'), 'family xi')
        self.assertEqual(normalize_taxonomy_assignment('Family_XI'), 'family xi')
        self.assertEqual(normalize_taxonomy_assignment('Family__XI'), 'family xi')
        self.assertEqual(normalize_taxonomy_assignment('Family   XI'), 'family xi')
    
    def test_parse_taxonomy_string(self):
        tax_string = "k__Bacteria; p__Proteobacteria; c__Gamma(0.95); o__Entero; f__; g__; s__"
        expected = ['k__Bacteria', 'p__Proteobacteria', 'c__Gamma', 'o__Entero', 'f__', 'g__', 's__']
        result = parse_taxonomy_string(tax_string)
        self.assertEqual(result, expected)
    
    def test_get_taxonomic_levels(self):
        levels_df = get_taxonomic_levels(self.tax1_df)
        self.assertEqual(levels_df.shape, (4, 7))  # 4 rows, 7 taxonomic levels
        self.assertEqual(list(levels_df.columns), ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'])
        self.assertEqual(levels_df.loc['ASV1', 'Family'], 'f__Enterobacteriaceae')
    
    def test_compare_taxonomies_enhanced(self):
        summary_stats, conf_matrices, common_diffs = compare_taxonomies(
            self.tax1_df, self.tax2_df, "Test1", "Test2"
        )
        
        # Check summary stats
        self.assertEqual(summary_stats.shape, (7, 4))  # 7 levels, 4 columns
        
        # Family level should show enhanced comparison:
        # ASV1: both have Enterobacteriaceae (same)
        # ASV2: Lactobacillaceae vs Streptococcaceae (different)
        # ASV3: "Family XI" vs "Family_XI" should be SAME due to normalization
        # ASV4: "Incertae_Sedis" vs "" should be SAME due to normalization
        self.assertEqual(summary_stats.loc['Family', 'Same'], 3)  # ASV1, ASV3, ASV4
        self.assertEqual(summary_stats.loc['Family', 'Different'], 1)  # ASV2
        
        # Check that differences include example feature IDs
        if 'Family' in common_diffs:
            family_diffs = common_diffs['Family']
            self.assertIn('Example Feature IDs', family_diffs.columns)
            # Should have ASV2 as an example for the Lactobacillaceae vs Streptococcaceae difference
            self.assertTrue(any('ASV2' in str(examples) for examples in family_diffs['Example Feature IDs']))


if __name__ == '__main__':
    unittest.main()
