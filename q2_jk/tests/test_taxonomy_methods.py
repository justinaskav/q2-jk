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
from q2_jk._methods import (parse_taxonomy_string, get_taxonomic_levels, compare_taxonomies, 
                          normalize_taxonomy_assignment, create_gram_staining_database,
                          classify_gram_status, analyze_gram_staining)


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

    def test_create_gram_staining_database(self):
        db = create_gram_staining_database()
        
        # Check that database contains expected levels
        self.assertIn('phylum', db)
        self.assertIn('family', db)
        self.assertIn('genus', db)
        
        # Check some known classifications
        self.assertEqual(db['phylum']['firmicutes'], 'positive')
        self.assertEqual(db['phylum']['proteobacteria'], 'negative')
        self.assertEqual(db['genus']['escherichia'], 'negative')
        self.assertEqual(db['genus']['staphylococcus'], 'positive')
        self.assertEqual(db['family']['enterobacteriaceae'], 'negative')

    def test_classify_gram_status(self):
        # Test gram-positive genus
        taxonomy_levels = {
            'Kingdom': 'k__Bacteria',
            'Phylum': 'p__Firmicutes', 
            'Family': 'f__Staphylococcaceae',
            'Genus': 'g__Staphylococcus'
        }
        status, level, confidence = classify_gram_status(taxonomy_levels)
        self.assertEqual(status, 'Gram Positive')
        self.assertEqual(level, 'Genus')
        self.assertEqual(confidence, 0.9)  # High confidence for genus-level
        
        # Test gram-negative genus
        taxonomy_levels = {
            'Kingdom': 'k__Bacteria',
            'Phylum': 'p__Proteobacteria',
            'Family': 'f__Enterobacteriaceae', 
            'Genus': 'g__Escherichia'
        }
        status, level, confidence = classify_gram_status(taxonomy_levels)
        self.assertEqual(status, 'Gram Negative')
        self.assertEqual(level, 'Genus')
        self.assertEqual(confidence, 0.9)  # High confidence for genus-level
        
        # Test family-level classification (no genus)
        taxonomy_levels = {
            'Kingdom': 'k__Bacteria',
            'Phylum': 'p__Firmicutes',
            'Family': 'f__Lactobacillaceae',
            'Genus': 'g__'
        }
        status, level, confidence = classify_gram_status(taxonomy_levels)
        self.assertEqual(status, 'Gram Positive')
        self.assertEqual(level, 'Family')
        self.assertEqual(confidence, 0.7)  # Good confidence for family-level
        
        # Test phylum-level classification
        taxonomy_levels = {
            'Kingdom': 'k__Bacteria',
            'Phylum': 'p__Bacteroidetes',
            'Family': 'f__',
            'Genus': 'g__'
        }
        status, level, confidence = classify_gram_status(taxonomy_levels)
        self.assertEqual(status, 'Gram Negative')
        self.assertEqual(level, 'Phylum')
        self.assertEqual(confidence, 0.5)  # Moderate confidence for phylum-level
        
        # Test non-bacterial
        taxonomy_levels = {
            'Kingdom': 'k__Archaea',
            'Phylum': 'p__Euryarchaeota'
        }
        status, level, confidence = classify_gram_status(taxonomy_levels)
        self.assertEqual(status, 'Non-Bacterial')
        self.assertEqual(level, 'Kingdom')
        self.assertEqual(confidence, 0.2)  # Low confidence for kingdom-level
        
        # Test variable classification (Mycoplasma)
        taxonomy_levels = {
            'Kingdom': 'k__Bacteria',
            'Phylum': 'p__Tenericutes',
            'Genus': 'g__Mycoplasma'
        }
        status, level, confidence = classify_gram_status(taxonomy_levels)
        self.assertEqual(status, 'Variable/Unknown')
        self.assertEqual(level, 'Genus')
        self.assertEqual(confidence, 0.72)  # 0.9 * 0.8 for variable classification
        
        # Test unclassified
        taxonomy_levels = {
            'Kingdom': '',
            'Phylum': '',
            'Genus': ''
        }
        status, level, confidence = classify_gram_status(taxonomy_levels)
        self.assertEqual(status, 'Unclassified')
        self.assertEqual(level, 'None')
        self.assertEqual(confidence, 0.0)

    def test_analyze_gram_staining(self):
        # Create test taxonomy DataFrame
        test_tax_df = pd.DataFrame({
            'Taxon': [
                'k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales; f__Staphylococcaceae; g__Staphylococcus; s__aureus',
                'k__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; o__Enterobacterales; f__Enterobacteriaceae; g__Escherichia; s__coli',
                'k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales; f__Lactobacillaceae; g__Lactobacillus; s__',
                'k__Bacteria; p__Tenericutes; c__Mollicutes; o__Mycoplasmatales; f__Mycoplasmataceae; g__Mycoplasma; s__',
                'k__Archaea; p__Euryarchaeota; c__Methanobacteria; o__; f__; g__; s__',
                'k__; p__; c__; o__; f__; g__; s__'
            ]
        }, index=['ASV1', 'ASV2', 'ASV3', 'ASV4', 'ASV5', 'ASV6'])
        
        (summary_stats, detailed_classification, level_breakdown, unique_taxa_summary, 
         confidence_summary, abundance_weighted_stats, database_coverage, pipeline_stats) = analyze_gram_staining(test_tax_df)
        
        # Check summary statistics
        self.assertEqual(len(summary_stats), 5)  # Should have 5 gram status categories
        self.assertEqual(summary_stats.loc['Gram Positive', 'Count'], 2)  # ASV1, ASV3
        self.assertEqual(summary_stats.loc['Gram Negative', 'Count'], 1)  # ASV2
        self.assertEqual(summary_stats.loc['Variable/Unknown', 'Count'], 1)  # ASV4
        self.assertEqual(summary_stats.loc['Non-Bacterial', 'Count'], 1)  # ASV5
        self.assertEqual(summary_stats.loc['Unclassified', 'Count'], 1)  # ASV6
        
        # Check detailed classification
        self.assertEqual(len(detailed_classification), 6)
        self.assertEqual(detailed_classification.loc[0, 'GramStatus'], 'Gram Positive')  # ASV1
        self.assertEqual(detailed_classification.loc[1, 'GramStatus'], 'Gram Negative')  # ASV2
        self.assertEqual(detailed_classification.loc[2, 'GramStatus'], 'Gram Positive')  # ASV3
        self.assertEqual(detailed_classification.loc[3, 'GramStatus'], 'Variable/Unknown')  # ASV4
        self.assertEqual(detailed_classification.loc[4, 'GramStatus'], 'Non-Bacterial')  # ASV5
        self.assertEqual(detailed_classification.loc[5, 'GramStatus'], 'Unclassified')  # ASV6
        
        # Check classification levels used
        self.assertEqual(detailed_classification.loc[0, 'ClassificationLevel'], 'Genus')  # Staphylococcus
        self.assertEqual(detailed_classification.loc[1, 'ClassificationLevel'], 'Genus')  # Escherichia
        self.assertEqual(detailed_classification.loc[2, 'ClassificationLevel'], 'Genus')  # Lactobacillus
        self.assertEqual(detailed_classification.loc[3, 'ClassificationLevel'], 'Genus')  # Mycoplasma
        
        # Check confidence scores are present
        self.assertIn('ConfidenceScore', detailed_classification.columns)
        self.assertTrue(all(detailed_classification['ConfidenceScore'] >= 0))
        self.assertTrue(all(detailed_classification['ConfidenceScore'] <= 1))
        
        # Check level breakdown exists (now has two categories)
        self.assertIn('classification_source', level_breakdown)
        self.assertIn('taxonomic_resolution', level_breakdown)
        
        # Check percentages are calculated correctly
        total = len(detailed_classification)
        expected_gram_pos_pct = (2 / total) * 100
        self.assertEqual(summary_stats.loc['Gram Positive', 'Percentage'], expected_gram_pos_pct)
        
        # Check unique taxa summary exists
        self.assertIsInstance(unique_taxa_summary, dict)
        self.assertTrue(len(unique_taxa_summary) > 0)
        
        # Check confidence summary exists
        self.assertIsInstance(confidence_summary, pd.DataFrame)
        self.assertIn('MeanConfidence', confidence_summary.columns)
        
        # Check no abundance data when feature table not provided
        self.assertIsNone(abundance_weighted_stats)
        
        # Check new components exist
        self.assertIsInstance(database_coverage, dict)
        self.assertIsInstance(pipeline_stats, pd.DataFrame)
        self.assertIn('Step', pipeline_stats.columns)

    def test_gram_staining_edge_cases(self):
        # Test with normalized taxonomy assignments
        taxonomy_levels = {
            'Kingdom': 'k__Bacteria',
            'Phylum': 'p__Firmicutes',
            'Family': 'f__Family_XI',  # Should normalize to 'family xi'
            'Genus': 'g__'
        }
        status, level, confidence = classify_gram_status(taxonomy_levels)
        # Should fall back to phylum level since family not in database
        self.assertEqual(status, 'Gram Positive')
        self.assertEqual(level, 'Phylum')
        self.assertEqual(confidence, 0.5)
        
        # Test with Incertae Sedis
        taxonomy_levels = {
            'Kingdom': 'k__Bacteria',
            'Phylum': 'p__Proteobacteria',
            'Family': 'f__Incertae_Sedis',
            'Genus': 'g__'
        }
        status, level, confidence = classify_gram_status(taxonomy_levels)
        self.assertEqual(status, 'Gram Negative')
        self.assertEqual(level, 'Phylum')
        self.assertEqual(confidence, 0.5)


if __name__ == '__main__':
    unittest.main()
