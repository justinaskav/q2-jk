# ----------------------------------------------------------------------------
# Copyright (c) 2024, Justinas Kavoliūnas.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_taxonomy_string(tax_string):
    """Parse a taxonomy string into a list of taxonomic levels."""
    # Split by semicolon and remove any confidence values in parentheses
    levels = []
    for level in tax_string.split(';'):
        # Remove any confidence values or other text in parentheses
        if '(' in level:
            level = level.split('(')[0].strip()
        levels.append(level.strip())
    return levels


def get_taxonomic_levels(taxonomy_df, col_name='Taxon'):
    """Extract taxonomic levels from a taxonomy DataFrame."""
    # Get the column with taxonomy strings
    tax_col = taxonomy_df[col_name]
    # Parse each taxonomy string
    parsed_levels = [parse_taxonomy_string(tax) for tax in tax_col]
    # Determine the maximum number of levels
    max_levels = max(len(levels) for levels in parsed_levels)
    
    # Create columns for each level
    level_names = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    level_names = level_names[:max_levels]  # Adjust to actual number of levels
    
    # Create a DataFrame with level columns
    levels_df = pd.DataFrame(index=taxonomy_df.index)
    for i, level_name in enumerate(level_names):
        # Extract the appropriate level or use an empty string if not available
        levels_df[level_name] = [levels[i] if i < len(levels) else '' for levels in parsed_levels]
    
    return levels_df


def compare_taxonomies(tax1_df: pd.DataFrame, tax2_df: pd.DataFrame, tax1_name="Taxonomy 1", tax2_name="Taxonomy 2"):
    """Compare two taxonomy DataFrames.
    
    Parameters
    ----------
    tax1_df : pd.DataFrame
        First taxonomy DataFrame
    tax2_df : pd.DataFrame
        Second taxonomy DataFrame
    tax1_name : str, optional
        Name for the first taxonomy
    tax2_name : str, optional
        Name for the second taxonomy
    
    Returns
    -------
    pd.DataFrame
        Summary statistics DataFrame
    """
    # Get common feature IDs
    common_features = set(tax1_df.index).intersection(set(tax2_df.index))
    only_tax1 = set(tax1_df.index) - common_features
    only_tax2 = set(tax2_df.index) - common_features
    
    # Filter to common features
    if common_features:
        tax1_df = tax1_df.loc[list(common_features)]
        tax2_df = tax2_df.loc[list(common_features)]
    else:
        raise ValueError("No common features found between taxonomies")
    
    # Extract taxonomic levels for each taxonomy
    tax1_levels = get_taxonomic_levels(tax1_df)
    tax2_levels = get_taxonomic_levels(tax2_df)
    
    # Overall summary statistics
    level_names = tax1_levels.columns
    summary_stats = pd.DataFrame(index=level_names, 
                               columns=['Total', 'Same', 'Different', 'PctSame'])
    
    # Generate confusion matrices for visualization
    confusion_matrices = {}
    common_differences = {}
    
    # Detailed differences for each level
    for level in level_names:
        # Compare assignments at this level
        same_mask = tax1_levels[level] == tax2_levels[level]
        diff_mask = ~same_mask
        
        # Count matches and differences
        n_same = sum(same_mask)
        n_diff = sum(diff_mask)
        n_total = len(common_features)
        pct_same = (n_same / n_total) * 100 if n_total > 0 else 0
        
        # Add to summary stats
        summary_stats.loc[level] = [n_total, n_same, n_diff, pct_same]
        
        # Create a detailed report for differences at this level
        if n_diff > 0:
            diff_df = pd.DataFrame({
                'FeatureID': tax1_df.loc[diff_mask].index,
                'Tax1_Assignment': tax1_levels.loc[diff_mask, level],
                'Tax2_Assignment': tax2_levels.loc[diff_mask, level],
                'Full_Tax1': tax1_df.loc[diff_mask, 'Taxon'],
                'Full_Tax2': tax2_df.loc[diff_mask, 'Taxon']
            })
            
            # Create confusion matrix for this level
            unique_vals1 = set(tax1_levels[level])
            unique_vals2 = set(tax2_levels[level])
            all_vals = sorted(list(unique_vals1.union(unique_vals2)))
            
            # Create confusion matrix
            conf_matrix = pd.DataFrame(0, index=all_vals, columns=all_vals)
            for i, val1 in enumerate(tax1_levels[level]):
                val2 = tax2_levels[level].iloc[i]
                conf_matrix.loc[val1, val2] += 1
                
            confusion_matrices[level] = conf_matrix
            
            # Find most common mismatches (differences between taxonomies)
            mismatches = []
            for i, row in conf_matrix.iterrows():
                for j, val in row.items():
                    if i != j and val > 0:  # Skip the diagonal (matching assignments)
                        mismatches.append((i, j, val, (val / n_total) * 100))
            
            # Sort by count (descending)
            mismatches.sort(key=lambda x: x[2], reverse=True)
            
            # Create a DataFrame for most common differences
            if mismatches:
                top_n = min(20, len(mismatches))  # Show top 20 or fewer
                common_diff_df = pd.DataFrame(
                    mismatches[:top_n], 
                    columns=[f'{tax1_name} Assignment', f'{tax2_name} Assignment', 
                            'Count', 'Percent of Features']
                )
                common_diff_df['Percent of Features'] = common_diff_df['Percent of Features'].round(2)
                common_differences[level] = common_diff_df
    
    return summary_stats, confusion_matrices, common_differences
