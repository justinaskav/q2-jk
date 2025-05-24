# ----------------------------------------------------------------------------
# Copyright (c) 2024, Justinas Kavoliūnas.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qiime2
import q2templates
from ._methods import compare_taxonomies, get_taxonomic_levels
import re


# Define standard taxonomic level order
TAXONOMIC_ORDER = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']


def calculate_classification_rate(taxonomy_df, col_name='Taxon', truncate_prefixes=True):
    """Calculate the percentage of classified taxa at each level.
    
    Parameters
    ----------
    taxonomy_df : pd.DataFrame
        DataFrame containing taxonomy data
    col_name : str, optional
        Column containing taxonomy strings
    truncate_prefixes : bool, optional
        Whether to truncate rank prefixes like 'g__', 's__', etc.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with classification rates at each level
    """
    tax_levels = get_taxonomic_levels(taxonomy_df, col_name)
    total_features = len(tax_levels)
    
    classification_stats = []
    level_names = tax_levels.columns
    
    for level in level_names:
        # Count non-empty values (classified)
        classified = tax_levels[level].str.strip().astype(bool).sum()
        unclassified = total_features - classified
        pct_classified = (classified / total_features) * 100 if total_features > 0 else 0
        
        # Count unique taxa at this level (excluding empty strings)
        unique_taxa = tax_levels[level].replace('', np.nan).dropna().nunique()
        
        # Check for patterns like "g__" or "s__" that indicate empty classifications
        if truncate_prefixes:
            prefix_pattern = r'^[gkpcofsd]__$'
            empty_prefix = tax_levels[level].str.match(prefix_pattern).sum()
            classified -= empty_prefix
            unclassified += empty_prefix
            pct_classified = (classified / total_features) * 100 if total_features > 0 else 0
        
        classification_stats.append([
            level, classified, unclassified, pct_classified, unique_taxa
        ])
    
    df = pd.DataFrame(
        classification_stats,
        columns=['Level', 'Classified', 'Unclassified', 'PctClassified', 'UniqueTaxa']
    )
    
    # Order by standard taxonomic levels
    df['LevelOrder'] = df['Level'].map(lambda x: TAXONOMIC_ORDER.index(x) if x in TAXONOMIC_ORDER else 999)
    df = df.sort_values('LevelOrder').drop(columns=['LevelOrder'])
    
    return df


def create_enhanced_heatmap(conf_matrix, level, tax1_name, tax2_name, output_path):
    """Create an enhanced heatmap visualization for a confusion matrix.
    
    Parameters
    ----------
    conf_matrix : pd.DataFrame
        Confusion matrix data
    level : str
        Taxonomic level name
    tax1_name : str
        Name for the first taxonomy
    tax2_name : str
        Name for the second taxonomy
    output_path : str
        Path to save the figure
        
    Returns
    -------
    bool
        True if the heatmap was created successfully
    """
    # If matrix is too large, return False
    if len(conf_matrix) > 100:
        return False
    
    # Calculate dynamic figure size based on number of unique labels
    n_labels = len(conf_matrix)
    fig_size = min(12, max(8, n_labels * 0.5))  # Scale size but cap it
    
    # Create figure and axes
    plt.figure(figsize=(fig_size, fig_size), dpi=100)
    
    # For large matrices, adjust the font size accordingly
    if n_labels > 30:
        annot_size = 6
    elif n_labels > 20:
        annot_size = 8
    else:
        annot_size = 10
        
    # Create the heatmap with appropriate styling
    ax = sns.heatmap(
        conf_matrix, 
        annot=True, 
        cmap='Blues', 
        fmt='g',
        annot_kws={"size": annot_size},
        cbar_kws={'label': 'Count of Features'}
    )
    
    # Rotate x-axis labels for better readability and adjust alignment
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Improved labeling with detailed information
    plt.title(f'Confusion Matrix at {level} Level\nValues show count of features', fontsize=14)
    plt.xlabel(f'{tax2_name} Assignment', fontsize=12)
    plt.ylabel(f'{tax1_name} Assignment', fontsize=12)
    
    # Add a note about the diagonal
    plt.figtext(0.5, 0.01, 
                "Diagonal values represent matching assignments; off-diagonal values show disagreements.", 
                ha="center", fontsize=10, style='italic')
    
    # Use tight layout to prevent overflow
    plt.tight_layout()
    
    # Save figure with high resolution
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return True


def visualize_taxonomy_comparison(output_dir: str,
                                 tax1: pd.DataFrame,
                                 tax2: pd.DataFrame,
                                 tax1_name: str = "Taxonomy 1",
                                 tax2_name: str = "Taxonomy 2") -> None:
    """
    Generate a visualization comparing two taxonomy classifications.
    
    Parameters
    ----------
    output_dir : str
        Directory where the visualization will be written
    tax1 : pd.DataFrame
        First taxonomy DataFrame
    tax2 : pd.DataFrame
        Second taxonomy DataFrame
    tax1_name : str, optional
        Name for the first taxonomy method
    tax2_name : str, optional
        Name for the second taxonomy method
    """
    # Get comparison results
    summary_stats, confusion_matrices, common_differences = compare_taxonomies(
        tax1, tax2, tax1_name, tax2_name
    )
    
    # Apply taxonomic level ordering to summary stats
    summary_stats['LevelOrder'] = summary_stats.index.map(lambda x: TAXONOMIC_ORDER.index(x) if x in TAXONOMIC_ORDER else 999)
    summary_stats = summary_stats.sort_values('LevelOrder').drop(columns=['LevelOrder'])
    
    # Count features in each taxonomy
    common_features = set(tax1.index).intersection(set(tax2.index))
    only_tax1 = set(tax1.index) - common_features
    only_tax2 = set(tax2.index) - common_features
    
    feature_counts = pd.DataFrame([
        [len(tax1.index), len(only_tax1), f"{(len(only_tax1) / len(tax1.index) * 100):.1f}%"],
        [len(tax2.index), len(only_tax2), f"{(len(only_tax2) / len(tax2.index) * 100):.1f}%"],
        [len(common_features), len(common_features), "100.0%"]
    ], columns=["Total Features", "Unique Features", "Percentage Unique"],
       index=[tax1_name, tax2_name, "Common"])
    
    # Calculate classification rates for each taxonomy
    tax1_class_rates = calculate_classification_rate(tax1)
    tax2_class_rates = calculate_classification_rate(tax2)
    
    # Add taxonomy name to the classification rates
    tax1_class_rates['Taxonomy'] = tax1_name
    tax2_class_rates['Taxonomy'] = tax2_name
    
    # Combine classification stats
    classification_stats = pd.concat([tax1_class_rates, tax2_class_rates])
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, "summary_stats.tsv")
    summary_stats.to_csv(summary_file, sep='\t')
    
    # Save classification stats
    class_stats_file = os.path.join(output_dir, "classification_stats.tsv")
    classification_stats.to_csv(class_stats_file, sep='\t', index=False)
    
    # Create visualizations for confusion matrices
    level_names = sorted(list(confusion_matrices.keys()), 
                         key=lambda x: TAXONOMIC_ORDER.index(x) if x in TAXONOMIC_ORDER else 999)
    
    # Create enhanced heatmaps with matplotlib/seaborn
    viz_files = []
    for level in level_names:
        conf_matrix = confusion_matrices[level]
        viz_file = f"confusion_{level}.png"
        output_path = os.path.join(output_dir, viz_file)
        
        if create_enhanced_heatmap(conf_matrix, level, tax1_name, tax2_name, output_path):
            viz_files.append(viz_file)
    
    # Save common differences
    for level in level_names:
        if level in common_differences:
            diff_df = common_differences[level]
            diff_file = os.path.join(output_dir, f"common_differences_{level}.tsv")
            diff_df.to_csv(diff_file, sep='\t', index=False)
    
    # Create index.html using q2templates
    index_fp = os.path.join(TEMPLATES, 'index.html')
    context = {
        'title': 'Taxonomy Comparison',
        'tax1_name': tax1_name,
        'tax2_name': tax2_name,
        'feature_counts': feature_counts,
        'summary_stats': summary_stats,
        'classification_stats': classification_stats,
        'levels': level_names,
        'viz_files': viz_files
    }
    
    # Save all DataFrames as HTML for inclusion in the report
    feature_counts_html = feature_counts.to_html(classes=["table", "table-striped", "table-hover"])
    summary_stats_html = summary_stats.to_html(classes=["table", "table-striped", "table-hover"])
    
    # Create a pivot table for classification rates for easier comparison
    # Ensure proper ordering of taxonomic levels
    class_pivot = classification_stats.pivot_table(
        index='Level', 
        columns='Taxonomy', 
        values=['PctClassified', 'UniqueTaxa']
    )
    
    # Add order for sorting
    class_pivot_reset = class_pivot.reset_index()
    class_pivot_reset['LevelOrder'] = class_pivot_reset['Level'].map(
        lambda x: TAXONOMIC_ORDER.index(x) if x in TAXONOMIC_ORDER else 999
    )
    class_pivot_reset = class_pivot_reset.sort_values('LevelOrder').drop(columns=['LevelOrder'])
    class_pivot = class_pivot_reset.set_index('Level')
    
    # Format column names
    class_pivot.columns = [f"{col[1]} {col[0]}" for col in class_pivot.columns]
    class_stats_html = class_pivot.to_html(classes=["table", "table-striped", "table-hover"])
    
    # Create HTML for common differences with proper ordering
    common_diffs_html = {}
    for level in level_names:
        if level in common_differences:
            common_diffs_html[level] = common_differences[level].to_html(
                classes=["table", "table-striped", "table-hover"], index=False)
    
    context['feature_counts_html'] = feature_counts_html
    context['summary_stats_html'] = summary_stats_html
    context['classification_stats_html'] = class_stats_html
    context['common_diffs_html'] = common_diffs_html
    
    # Create HTML report
    q2templates.render(index_fp, output_dir, context=context)


# Define the templates directory
TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
