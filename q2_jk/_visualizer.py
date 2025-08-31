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
from ._methods import (compare_taxonomies, get_taxonomic_levels, normalize_taxonomy_assignment,
                       analyze_gram_staining, analyze_comprehensive_taxonomic_levels)
from ._sample_analysis import (calculate_abundance_weighted_summary, calculate_sample_wise_feature_counts,
                              calculate_sample_wise_abundance, calculate_feature_abundance_discrepancy)
from ._ratios import calculate_all_ratios
from ._metadata import analyze_metadata_groups
from ._bias_detection import assess_all_extraction_bias
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
        # Use the new normalization function to determine classification status
        normalized_assignments = tax_levels[level].apply(normalize_taxonomy_assignment)
        classified = (normalized_assignments != "").sum()
        unclassified = total_features - classified
        pct_classified = (classified / total_features) * 100 if total_features > 0 else 0
        
        # Count unique taxa at this level (excluding normalized empty assignments)
        unique_taxa = normalized_assignments[normalized_assignments != ""].nunique()
        
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
    class_pivot_reset = class_pivot_reset.sort_values('LevelOrder')
    class_pivot_reset = class_pivot_reset.drop(columns=['LevelOrder'])
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


def visualize_gram_staining(output_dir: str, taxonomy: pd.DataFrame, feature_table: pd.DataFrame = None, metadata: pd.DataFrame = None) -> None:
    """
    Generate a comprehensive visualization analyzing gram staining composition with advanced diagnostics.
    
    Parameters
    ----------
    output_dir : str
        Directory where the visualization will be written
    taxonomy : pd.DataFrame
        Taxonomy DataFrame with feature data
    feature_table : pd.DataFrame, optional
        Feature abundance table for abundance-weighted analysis and bias detection
    metadata : pd.DataFrame, optional
        Sample metadata for group-based comparisons and statistical analysis
    """
    _visualize_gram_staining_impl(output_dir, taxonomy, feature_table, metadata)




def calculate_abundance_weighted_summary(detailed_classification: pd.DataFrame, feature_table: pd.DataFrame):
    """
    Calculate abundance-weighted summary statistics for gram staining categories.
    
    Parameters
    ----------
    detailed_classification : pd.DataFrame
        Per-feature classification results
    feature_table : pd.DataFrame
        Feature abundance table
        
    Returns
    -------
    pd.DataFrame
        Abundance-weighted summary statistics
    """
    # Ensure feature IDs match
    classification_features = set(detailed_classification['FeatureID'])
    table_features = set(feature_table.index)
    common_features = list(classification_features.intersection(table_features))
    
    
    if not common_features:
        return None
    
    # Calculate total abundance for each gram status
    abundance_by_gram = {}
    
    for gram_status in detailed_classification['GramStatus'].unique():
        # Get features with this gram status
        gram_features = detailed_classification[detailed_classification['GramStatus'] == gram_status]['FeatureID'].tolist()
        # Filter to common features and calculate total abundance across all samples
        gram_features_common = [f for f in gram_features if f in common_features]
        if gram_features_common:
            total_abundance = feature_table.loc[gram_features_common].sum().sum()
        else:
            total_abundance = 0
        abundance_by_gram[gram_status] = total_abundance
    
    # Convert to DataFrame
    abundance_counts = pd.Series(abundance_by_gram)
    total_abundance = abundance_counts.sum()
    
    if total_abundance == 0:
        return None
    
    abundance_summary = pd.DataFrame({
        'Count': abundance_counts,  # Total abundance counts
        'Percentage': (abundance_counts / total_abundance * 100).round(2)
    })
    abundance_summary = abundance_summary.sort_values('Count', ascending=False)
    
    return abundance_summary


def calculate_sample_wise_feature_counts(detailed_classification: pd.DataFrame, feature_table: pd.DataFrame):
    """
    Calculate gram staining composition for each sample based on feature presence/absence.
    
    Parameters
    ----------
    detailed_classification : pd.DataFrame
        Per-feature classification results
    feature_table : pd.DataFrame
        Feature abundance table (features x samples)
        
    Returns
    -------
    pd.DataFrame
        Sample-wise feature count statistics
    """
    # Ensure feature IDs match
    classification_features = set(detailed_classification['FeatureID'])
    table_features = set(feature_table.index)
    common_features = list(classification_features.intersection(table_features))
    
    if not common_features:
        return None
    
    # Filter both dataframes to common features
    filtered_classification = detailed_classification[detailed_classification['FeatureID'].isin(common_features)].copy()
    filtered_table = feature_table.loc[common_features]
    
    # Create mapping from feature ID to gram status
    feature_to_gram = dict(zip(filtered_classification['FeatureID'], filtered_classification['GramStatus']))
    
    # Calculate feature presence/absence per sample (>0 abundance = present)
    presence_matrix = (filtered_table > 0).astype(int)
    
    # Calculate gram composition for each sample
    sample_results = []
    for sample_id in presence_matrix.columns:
        sample_presence = presence_matrix[sample_id]
        present_features = sample_presence[sample_presence > 0].index.tolist()
        
        # Count gram categories for present features
        gram_counts = {'Gram Positive': 0, 'Gram Negative': 0, 'Variable/Unknown': 0, 'Non-Bacterial': 0, 'Unclassified': 0}
        for feature_id in present_features:
            gram_status = feature_to_gram[feature_id]
            if gram_status in gram_counts:
                gram_counts[gram_status] += 1
        
        total_features = len(present_features)
        
        # Calculate percentages
        sample_result = {
            'SampleID': sample_id,
            'TotalFeatures': total_features,
            'GramPositive_Count': gram_counts['Gram Positive'],
            'GramNegative_Count': gram_counts['Gram Negative'],
            'VariableUnknown_Count': gram_counts['Variable/Unknown'],
            'NonBacterial_Count': gram_counts['Non-Bacterial'],
            'Unclassified_Count': gram_counts['Unclassified'],
            'GramPositive_Pct': round((gram_counts['Gram Positive'] / total_features * 100) if total_features > 0 else 0, 2),
            'GramNegative_Pct': round((gram_counts['Gram Negative'] / total_features * 100) if total_features > 0 else 0, 2),
            'VariableUnknown_Pct': round((gram_counts['Variable/Unknown'] / total_features * 100) if total_features > 0 else 0, 2),
            'NonBacterial_Pct': round((gram_counts['Non-Bacterial'] / total_features * 100) if total_features > 0 else 0, 2),
            'Unclassified_Pct': round((gram_counts['Unclassified'] / total_features * 100) if total_features > 0 else 0, 2)
        }
        sample_results.append(sample_result)
    
    return pd.DataFrame(sample_results)


def calculate_sample_wise_abundance(detailed_classification: pd.DataFrame, feature_table: pd.DataFrame):
    """
    Calculate abundance-weighted gram staining composition for each sample.
    
    Parameters
    ----------
    detailed_classification : pd.DataFrame
        Per-feature classification results
    feature_table : pd.DataFrame
        Feature abundance table (features x samples)
        
    Returns
    -------
    pd.DataFrame
        Sample-wise abundance statistics
    """
    # Ensure feature IDs match
    classification_features = set(detailed_classification['FeatureID'])
    table_features = set(feature_table.index)
    common_features = list(classification_features.intersection(table_features))
    
    if not common_features:
        return None
    
    # Filter both dataframes to common features
    filtered_classification = detailed_classification[detailed_classification['FeatureID'].isin(common_features)].copy()
    filtered_table = feature_table.loc[common_features]
    
    # Create mapping from feature ID to gram status
    feature_to_gram = dict(zip(filtered_classification['FeatureID'], filtered_classification['GramStatus']))
    
    # Calculate abundance-weighted composition for each sample
    sample_results = []
    for sample_id in filtered_table.columns:
        sample_abundances = filtered_table[sample_id]
        
        # Calculate total abundance for each gram category
        gram_abundances = {'Gram Positive': 0, 'Gram Negative': 0, 'Variable/Unknown': 0, 'Non-Bacterial': 0, 'Unclassified': 0}
        for feature_id, abundance in sample_abundances.items():
            if abundance > 0:  # Only count features with non-zero abundance
                gram_status = feature_to_gram[feature_id]
                if gram_status in gram_abundances:
                    gram_abundances[gram_status] += abundance
        
        total_abundance = sum(gram_abundances.values())
        
        # Calculate percentages
        sample_result = {
            'SampleID': sample_id,
            'TotalAbundance': int(total_abundance),
            'GramPositive_Abundance': int(gram_abundances['Gram Positive']),
            'GramNegative_Abundance': int(gram_abundances['Gram Negative']),
            'VariableUnknown_Abundance': int(gram_abundances['Variable/Unknown']),
            'NonBacterial_Abundance': int(gram_abundances['Non-Bacterial']),
            'Unclassified_Abundance': int(gram_abundances['Unclassified']),
            'GramPositive_AbundancePct': round((gram_abundances['Gram Positive'] / total_abundance * 100) if total_abundance > 0 else 0, 2),
            'GramNegative_AbundancePct': round((gram_abundances['Gram Negative'] / total_abundance * 100) if total_abundance > 0 else 0, 2),
            'VariableUnknown_AbundancePct': round((gram_abundances['Variable/Unknown'] / total_abundance * 100) if total_abundance > 0 else 0, 2),
            'NonBacterial_AbundancePct': round((gram_abundances['Non-Bacterial'] / total_abundance * 100) if total_abundance > 0 else 0, 2),
            'Unclassified_AbundancePct': round((gram_abundances['Unclassified'] / total_abundance * 100) if total_abundance > 0 else 0, 2)
        }
        sample_results.append(sample_result)
    
    return pd.DataFrame(sample_results)


def calculate_feature_abundance_discrepancy(sample_wise_stats: pd.DataFrame, sample_wise_abundance: pd.DataFrame):
    """
    Calculate discrepancy between feature diversity and abundance patterns.
    Large discrepancies may indicate DNA extraction bias.
    
    Parameters
    ----------
    sample_wise_stats : pd.DataFrame
        Sample-wise feature count statistics
    sample_wise_abundance : pd.DataFrame
        Sample-wise abundance statistics
        
    Returns
    -------
    pd.DataFrame
        Discrepancy analysis for each sample
    """
    if sample_wise_stats is None or sample_wise_abundance is None:
        return None
    
    # Merge the dataframes on SampleID
    merged = sample_wise_stats.merge(sample_wise_abundance, on='SampleID', suffixes=('_Features', '_Abundance'))
    
    # Calculate discrepancies (Abundance % - Feature %)
    discrepancy_results = []
    for _, row in merged.iterrows():
        sample_id = row['SampleID']
        
        # Calculate discrepancies for each gram category
        gp_discrepancy = row['GramPositive_AbundancePct'] - row['GramPositive_Pct']
        gn_discrepancy = row['GramNegative_AbundancePct'] - row['GramNegative_Pct']
        
        # Calculate extraction bias score (negative = gram-positive underrepresented)
        extraction_bias_score = gp_discrepancy  # Negative values suggest gram-positive extraction issues
        
        # Classify pattern
        if abs(gp_discrepancy) < 5 and abs(gn_discrepancy) < 5:
            pattern = "Balanced"
        elif gp_discrepancy < -10:
            pattern = "Gram+ Underrepresented"
        elif gp_discrepancy > 10:
            pattern = "Gram+ Overrepresented"
        elif gn_discrepancy < -10:
            pattern = "Gram- Underrepresented"
        elif gn_discrepancy > 10:
            pattern = "Gram- Overrepresented"
        else:
            pattern = "Minor Discrepancy"
        
        # Determine extraction bias likelihood
        if extraction_bias_score < -15:
            bias_likelihood = "High"
        elif extraction_bias_score < -10:
            bias_likelihood = "Moderate"
        elif extraction_bias_score < -5:
            bias_likelihood = "Low"
        else:
            bias_likelihood = "Minimal"
        
        discrepancy_result = {
            'SampleID': sample_id,
            'GramPos_Feature_Pct': round(row['GramPositive_Pct'], 2),
            'GramPos_Abundance_Pct': round(row['GramPositive_AbundancePct'], 2),
            'GramPos_Discrepancy': round(gp_discrepancy, 2),
            'GramNeg_Feature_Pct': round(row['GramNegative_Pct'], 2),
            'GramNeg_Abundance_Pct': round(row['GramNegative_AbundancePct'], 2),
            'GramNeg_Discrepancy': round(gn_discrepancy, 2),
            'Extraction_Bias_Score': round(extraction_bias_score, 2),
            'Pattern': pattern,
            'Bias_Likelihood': bias_likelihood
        }
        discrepancy_results.append(discrepancy_result)
    
    return pd.DataFrame(discrepancy_results)


def generate_classification_examples():
    """
    Generate concrete examples of how different taxonomic situations are classified.
    
    Returns
    -------
    dict
        Dictionary with example scenarios and their classifications
    """
    from ._methods import classify_gram_status
    
    examples = []
    
    # Example 1: Complete taxonomy with genus-level classification
    taxonomy1 = {
        'Kingdom': 'k__Bacteria',
        'Phylum': 'p__Bacillota',  # Modern GTDB name for Firmicutes
        'Class': 'c__Bacilli',
        'Order': 'o__Bacillales',
        'Family': 'f__Staphylococcaceae',
        'Genus': 'g__Staphylococcus',
        'Species': 's__aureus'
    }
    status1, level1, conf1 = classify_gram_status(taxonomy1)
    examples.append({
        'scenario': 'Complete Taxonomy',
        'taxonomy_string': 'k__Bacteria; p__Bacillota; c__Bacilli; o__Bacillales; f__Staphylococcaceae; g__Staphylococcus; s__aureus',
        'classification': status1,
        'level_used': level1,
        'confidence': f"{conf1:.1%}",
        'explanation': 'Feature has complete taxonomy down to species. Classified using genus-level database match (Staphylococcus = Gram Positive).',
        'color_class': 'success' if status1 == 'Gram Positive' else 'danger'
    })
    
    # Example 2: Taxonomy stopping at family level
    taxonomy2 = {
        'Kingdom': 'k__Bacteria',
        'Phylum': 'p__Proteobacteria',
        'Class': 'c__Gammaproteobacteria',
        'Order': 'o__Enterobacterales',
        'Family': 'f__Enterobacteriaceae',
        'Genus': 'g__',
        'Species': 's__'
    }
    status2, level2, conf2 = classify_gram_status(taxonomy2)
    examples.append({
        'scenario': 'Family-Level Only',
        'taxonomy_string': 'k__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; o__Enterobacterales; f__Enterobacteriaceae; g__; s__',
        'classification': status2,
        'level_used': level2,
        'confidence': f"{conf2:.1%}",
        'explanation': 'Genus and species are empty, but family-level classification available (Enterobacteriaceae = Gram Negative).',
        'color_class': 'success' if status2 == 'Gram Positive' else 'danger'
    })
    
    # Example 3: Only phylum-level information
    taxonomy3 = {
        'Kingdom': 'k__Bacteria',
        'Phylum': 'p__Bacteroidota',  # Modern GTDB name for Bacteroidetes
        'Class': 'c__',
        'Order': 'o__',
        'Family': 'f__',
        'Genus': 'g__',
        'Species': 's__'
    }
    status3, level3, conf3 = classify_gram_status(taxonomy3)
    examples.append({
        'scenario': 'Phylum-Level Only',
        'taxonomy_string': 'k__Bacteria; p__Bacteroidota; c__; o__; f__; g__; s__',
        'classification': status3,
        'level_used': level3,
        'confidence': f"{conf3:.1%}",
        'explanation': 'Only phylum information available. Uses broad phylum-level rule (Bacteroidota = Gram Negative) with lower confidence.',
        'color_class': 'success' if status3 == 'Gram Positive' else 'danger'
    })
    
    # Example 4: Non-bacterial
    taxonomy4 = {
        'Kingdom': 'k__Archaea',
        'Phylum': 'p__Euryarchaeota',
        'Class': 'c__Methanobacteria',
        'Order': 'o__',
        'Family': 'f__',
        'Genus': 'g__',
        'Species': 's__'
    }
    status4, level4, conf4 = classify_gram_status(taxonomy4)
    examples.append({
        'scenario': 'Non-Bacterial',
        'taxonomy_string': 'k__Archaea; p__Euryarchaeota; c__Methanobacteria; o__; f__; g__; s__',
        'classification': status4,
        'level_used': level4,
        'confidence': f"{conf4:.1%}",
        'explanation': 'Not bacterial (Archaea). Gram staining is not applicable to archaeal organisms.',
        'color_class': 'secondary'
    })
    
    # Example 5: Variable/Unknown (Mycoplasma)
    taxonomy5 = {
        'Kingdom': 'k__Bacteria',
        'Phylum': 'p__Tenericutes',
        'Class': 'c__Mollicutes',
        'Order': 'o__Mycoplasmatales',
        'Family': 'f__Mycoplasmataceae',
        'Genus': 'g__Mycoplasma',
        'Species': 's__'
    }
    status5, level5, conf5 = classify_gram_status(taxonomy5)
    examples.append({
        'scenario': 'Variable/Unknown',
        'taxonomy_string': 'k__Bacteria; p__Tenericutes; c__Mollicutes; o__Mycoplasmatales; f__Mycoplasmataceae; g__Mycoplasma; s__',
        'classification': status5,
        'level_used': level5,
        'confidence': f"{conf5:.1%}",
        'explanation': 'Mycoplasma lacks cell wall, making gram staining results variable or meaningless.',
        'color_class': 'warning'
    })
    
    # Example 6: Completely unclassified
    taxonomy6 = {
        'Kingdom': 'k__',
        'Phylum': 'p__',
        'Class': 'c__',
        'Order': 'o__',
        'Family': 'f__',
        'Genus': 'g__',
        'Species': 's__'
    }
    status6, level6, conf6 = classify_gram_status(taxonomy6)
    examples.append({
        'scenario': 'Unclassified',
        'taxonomy_string': 'k__; p__; c__; o__; f__; g__; s__',
        'classification': status6,
        'level_used': level6,
        'confidence': f"{conf6:.1%}",
        'explanation': 'No taxonomic information available at any level. Cannot make gram staining prediction.',
        'color_class': 'secondary'
    })
    
    # Example 7: Family with normalization (Incertae Sedis)
    taxonomy7 = {
        'Kingdom': 'k__Bacteria',
        'Phylum': 'p__Bacillota',
        'Class': 'c__Bacilli',
        'Order': 'o__Lactobacillales',
        'Family': 'f__Incertae_Sedis',
        'Genus': 'g__',
        'Species': 's__'
    }
    status7, level7, conf7 = classify_gram_status(taxonomy7)
    examples.append({
        'scenario': 'Incertae Sedis (Normalized)',
        'taxonomy_string': 'k__Bacteria; p__Bacillota; c__Bacilli; o__Lactobacillales; f__Incertae_Sedis; g__; s__',
        'classification': status7,
        'level_used': level7,
        'confidence': f"{conf7:.1%}",
        'explanation': '"Incertae Sedis" (uncertain placement) is normalized to empty, so falls back to phylum-level classification (Bacillota = Gram Positive).',
        'color_class': 'success' if status7 == 'Gram Positive' else 'danger'
    })
    
    # Example 8: Taxonomic resolution demonstration
    taxonomy8 = {
        'Kingdom': 'k__Bacteria',
        'Phylum': 'p__Bacillota',
        'Class': 'c__',
        'Order': 'o__',
        'Family': 'f__',
        'Genus': 'g__',
        'Species': 's__'
    }
    status8, level8, conf8 = classify_gram_status(taxonomy8)
    examples.append({
        'scenario': 'Resolution Drop-off Example',
        'taxonomy_string': 'k__Bacteria; p__Bacillota; c__; o__; f__; g__; s__',
        'classification': status8,
        'level_used': level8,
        'confidence': f"{conf8:.1%}",
        'explanation': 'This feature would be "Gram Positive" at Phylum level, but "Unclassified" at Genus level due to missing genus information (just "g__").',
        'color_class': 'success' if status8 == 'Gram Positive' else 'danger'
    })
    
    # Example 9: Domain prefix demonstration
    taxonomy9 = {
        'Kingdom': 'd__Bacteria',
        'Phylum': 'p__Proteobacteria',
        'Class': 'c__Gammaproteobacteria',
        'Order': 'o__',
        'Family': 'f__',
        'Genus': 'g__',
        'Species': 's__'
    }
    status9, level9, conf9 = classify_gram_status(taxonomy9)
    examples.append({
        'scenario': 'Domain Prefix (d__)',
        'taxonomy_string': 'd__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; o__; f__; g__; s__',
        'classification': status9,
        'level_used': level9,
        'confidence': f"{conf9:.1%}",
        'explanation': 'Uses domain prefix "d__Bacteria" (equivalent to "k__Bacteria"). Classified at phylum level since order/family/genus are empty.',
        'color_class': 'success' if status9 == 'Gram Positive' else 'danger'
    })
    
    return examples


def _visualize_gram_staining_impl(output_dir: str, taxonomy: pd.DataFrame, 
                                feature_table: pd.DataFrame = None, 
                                metadata: pd.DataFrame = None) -> None:
    """
    Internal implementation for gram staining visualization with comprehensive analysis.
    
    Parameters
    ----------
    output_dir : str
        Directory where the visualization will be written
    taxonomy : pd.DataFrame
        Taxonomy DataFrame with feature data
    feature_table : pd.DataFrame, optional
        Feature abundance table for abundance-weighted analysis
    metadata : pd.DataFrame, optional
        Sample metadata for group-based comparisons
    """
    # Perform gram staining analysis
    (summary_stats, detailed_classification, level_breakdown, unique_taxa_summary, 
     confidence_summary, abundance_weighted_stats, database_coverage, pipeline_stats) = analyze_gram_staining(taxonomy, feature_table)
    
    # Perform comprehensive taxonomic level analysis
    comprehensive_level_analysis = analyze_comprehensive_taxonomic_levels(taxonomy)
    
    # Generate classification examples for user understanding
    classification_examples = generate_classification_examples()
    
    # Initialize enhanced analysis variables
    abundance_summary_stats = None
    abundance_comprehensive_analysis = None
    sample_wise_stats = None
    sample_wise_abundance = None
    discrepancy_analysis = None
    fb_ratios = None
    extraction_ratios = None
    bias_assessment = None
    metadata_analysis = None
    
    # Enhanced analysis if feature_table provided
    if feature_table is not None:
        # QIIME2 feature tables come transposed (samples x features), we need (features x samples)
        feature_table_transposed = feature_table.T
        
        # Basic abundance analysis
        from ._sample_analysis import calculate_abundance_weighted_summary, calculate_sample_wise_feature_counts, calculate_sample_wise_abundance, calculate_feature_abundance_discrepancy
        abundance_summary_stats = calculate_abundance_weighted_summary(detailed_classification, feature_table_transposed)
        abundance_comprehensive_analysis = analyze_comprehensive_taxonomic_levels(taxonomy, feature_table_transposed)
        sample_wise_stats = calculate_sample_wise_feature_counts(detailed_classification, feature_table_transposed)
        sample_wise_abundance = calculate_sample_wise_abundance(detailed_classification, feature_table_transposed)
        discrepancy_analysis = calculate_feature_abundance_discrepancy(sample_wise_stats, sample_wise_abundance)
        
        # Phylum-level ratio calculations
        from ._ratios import calculate_sample_wise_fb_ratios, calculate_extraction_efficiency_ratios
        fb_ratios = calculate_sample_wise_fb_ratios(detailed_classification, feature_table_transposed)
        extraction_ratios = calculate_extraction_efficiency_ratios(detailed_classification, feature_table_transposed)
        
        # DNA extraction bias assessment
        from ._bias_detection import assess_all_extraction_bias
        bias_assessment = assess_all_extraction_bias(
            sample_discrepancy=discrepancy_analysis,
            fb_ratios=fb_ratios,
            extraction_ratios=extraction_ratios
        )
    
    # Metadata analysis if provided
    if metadata is not None:
        from ._metadata import analyze_metadata_groups
        metadata_analysis = analyze_metadata_groups(metadata, fb_ratios, sample_wise_stats)
    
    # Create summary visualization
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall gram distribution pie chart
    gram_counts = summary_stats['Count']
    colors = {
        'Gram Positive': '#2E7D32',      # Green
        'Gram Negative': '#C62828',      # Red
        'Variable/Unknown': '#FF8F00',   # Orange
        'Non-Bacterial': '#6A1B9A',     # Purple
        'Unclassified': '#757575',       # Gray
        'Bacterial': '#4CAF50'           # Light Green (for Kingdom level domain classification)
    }
    
    plot_colors = [colors.get(status, '#757575') for status in gram_counts.index]
    
    wedges, texts, autotexts = ax1.pie(
        gram_counts.values, 
        labels=gram_counts.index,
        autopct='%1.1f%%',
        colors=plot_colors,
        startangle=90
    )
    ax1.set_title('Overall Gram Staining Distribution', fontsize=14, fontweight='bold')
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Comprehensive taxonomic level analysis (bar chart)
    # Use comprehensive analysis for consistent data representation
    level_data = []
    for level, distribution in comprehensive_level_analysis.items():
        for gram_status, row in distribution.iterrows():
            level_data.append({
                'Level': level,
                'GramStatus': gram_status,
                'Percentage': row['Percentage']
            })
    
    if level_data:
        level_df = pd.DataFrame(level_data)
        # Pivot for stacked bar chart
        pivot_df = level_df.pivot(index='Level', columns='GramStatus', values='Percentage').fillna(0)
        
        # Reorder levels and columns
        level_order = [level for level in TAXONOMIC_ORDER if level in pivot_df.index]
        pivot_df = pivot_df.reindex(level_order)
        
        # Reorder columns to match color scheme (handle both Kingdom and other levels)
        all_possible_columns = ['Bacterial', 'Gram Positive', 'Gram Negative', 'Variable/Unknown', 'Non-Bacterial', 'Unclassified']
        col_order = [col for col in all_possible_columns if col in pivot_df.columns]
        pivot_df = pivot_df[col_order]
        
        # Create stacked bar chart
        pivot_df.plot(
            kind='bar', 
            stacked=True, 
            ax=ax2, 
            color=[colors.get(col, '#757575') for col in pivot_df.columns],
            width=0.7
        )
        
        ax2.set_title('Comprehensive Taxonomic Level Analysis', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Taxonomic Level', fontsize=12)
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.legend(title='Gram Status', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_ylim(0, 100)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gram_staining_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create abundance-weighted visualizations if abundance data available
    if abundance_summary_stats is not None:
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Abundance-weighted pie chart
        abundance_counts = abundance_summary_stats['Count']
        plot_colors_abundance = [colors.get(status, '#757575') for status in abundance_counts.index]
        
        wedges2, texts2, autotexts2 = ax3.pie(
            abundance_counts.values,
            labels=abundance_counts.index,
            autopct='%1.1f%%',
            colors=plot_colors_abundance,
            startangle=90
        )
        ax3.set_title('Abundance-Weighted Gram Distribution', fontsize=14, fontweight='bold')
        
        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Abundance-weighted bar chart by taxonomic level
        if abundance_comprehensive_analysis:
            abundance_level_data = []
            for level, distribution in abundance_comprehensive_analysis.items():
                for gram_status, row in distribution.iterrows():
                    abundance_level_data.append({
                        'Level': level,
                        'GramStatus': gram_status,
                        'Percentage': row['Percentage']
                    })
            
            if abundance_level_data:
                abundance_level_df = pd.DataFrame(abundance_level_data)
                abundance_pivot_df = abundance_level_df.pivot(index='Level', columns='GramStatus', values='Percentage').fillna(0)
                
                # Reorder levels and columns
                level_order = [level for level in TAXONOMIC_ORDER if level in abundance_pivot_df.index]
                abundance_pivot_df = abundance_pivot_df.reindex(level_order)
                
                all_possible_columns = ['Bacterial', 'Gram Positive', 'Gram Negative', 'Variable/Unknown', 'Non-Bacterial', 'Unclassified']
                col_order = [col for col in all_possible_columns if col in abundance_pivot_df.columns]
                abundance_pivot_df = abundance_pivot_df[col_order]
                
                abundance_pivot_df.plot(
                    kind='bar',
                    stacked=True,
                    ax=ax4,
                    color=[colors.get(col, '#757575') for col in abundance_pivot_df.columns],
                    width=0.7
                )
                
                ax4.set_title('Abundance-Weighted Taxonomic Level Analysis', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Taxonomic Level', fontsize=12)
                ax4.set_ylabel('Percentage (%)', fontsize=12)
                ax4.legend(title='Gram Status', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax4.set_ylim(0, 100)
                
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'abundance_weighted_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save detailed data as TSV files
    summary_file = os.path.join(output_dir, "gram_summary.tsv")
    summary_stats.to_csv(summary_file, sep='\t')
    
    detailed_file = os.path.join(output_dir, "detailed_classification.tsv")
    detailed_classification.to_csv(detailed_file, sep='\t', index=False)
    
    # Save comprehensive taxonomic level analysis
    comprehensive_combined = []
    for level, distribution in comprehensive_level_analysis.items():
        if isinstance(distribution, pd.DataFrame) and not distribution.empty:
            distribution_copy = distribution.copy()
            distribution_copy['TaxonomicLevel'] = level
            distribution_copy = distribution_copy.reset_index()
            distribution_copy = distribution_copy.rename(columns={'index': 'GramStatus'})
            comprehensive_combined.append(distribution_copy)
    
    if comprehensive_combined:
        comprehensive_df = pd.concat(comprehensive_combined, ignore_index=True)
        comprehensive_file = os.path.join(output_dir, "comprehensive_taxonomic_analysis.tsv")
        comprehensive_df.to_csv(comprehensive_file, sep='\t', index=False)

    # Save level breakdown (both types)
    # Classification source breakdown
    classification_source_combined = []
    for level, breakdown in level_breakdown['classification_source'].items():
        if isinstance(breakdown, pd.DataFrame) and not breakdown.empty:
            breakdown_copy = breakdown.copy()
            breakdown_copy['TaxonomicLevel'] = level
            breakdown_copy['BreakdownType'] = 'Classification Source'
            breakdown_copy = breakdown_copy.reset_index()
            breakdown_copy = breakdown_copy.rename(columns={'index': 'GramStatus'})
            classification_source_combined.append(breakdown_copy)
    
    if classification_source_combined:
        classification_source_df = pd.concat(classification_source_combined, ignore_index=True)
        classification_source_file = os.path.join(output_dir, "classification_source_breakdown.tsv")
        classification_source_df.to_csv(classification_source_file, sep='\t', index=False)
    
    # Taxonomic resolution breakdown  
    taxonomic_resolution_combined = []
    for level, breakdown in level_breakdown['taxonomic_resolution'].items():
        if isinstance(breakdown, pd.DataFrame) and not breakdown.empty:
            breakdown_copy = breakdown.copy()
            breakdown_copy['TaxonomicLevel'] = level
            breakdown_copy['BreakdownType'] = 'Taxonomic Resolution'
            breakdown_copy = breakdown_copy.reset_index()
            breakdown_copy = breakdown_copy.rename(columns={'index': 'GramStatus'})
            taxonomic_resolution_combined.append(breakdown_copy)
    
    if taxonomic_resolution_combined:
        taxonomic_resolution_df = pd.concat(taxonomic_resolution_combined, ignore_index=True)
        taxonomic_resolution_file = os.path.join(output_dir, "taxonomic_resolution_breakdown.tsv")
        taxonomic_resolution_df.to_csv(taxonomic_resolution_file, sep='\t', index=False)
    
    # Save confidence statistics
    confidence_file = os.path.join(output_dir, "confidence_statistics.tsv")
    confidence_summary.to_csv(confidence_file, sep='\t', index=False)
    
    # Save unique taxa summaries
    for level, taxa_summary in unique_taxa_summary.items():
        if isinstance(taxa_summary, pd.DataFrame) and not taxa_summary.empty:
            taxa_file = os.path.join(output_dir, f"unique_taxa_{level.lower()}.tsv")
            taxa_summary.to_csv(taxa_file, sep='\t', index=False)
    
    # Save database coverage statistics
    if database_coverage:
        coverage_rows = []
        for level, stats in database_coverage.items():
            coverage_rows.append({
                'TaxonomicLevel': level,
                'UniqueTaxaInData': stats['unique_taxa_in_data'],
                'TaxaInDatabase': stats['taxa_in_database'],
                'CoveragePercentage': round(stats['coverage_percentage'], 2),
                'DatabaseSize': stats['database_size']
            })
        
        if coverage_rows:
            coverage_df = pd.DataFrame(coverage_rows)
            coverage_file = os.path.join(output_dir, "database_coverage_report.tsv")
            coverage_df.to_csv(coverage_file, sep='\t', index=False)
    
    # Save classification pipeline statistics
    pipeline_file = os.path.join(output_dir, "classification_pipeline_stats.tsv")
    pipeline_stats.to_csv(pipeline_file, sep='\t', index=False)
    
    # Save sample-wise feature count statistics if available
    if sample_wise_stats is not None:
        sample_wise_file = os.path.join(output_dir, "sample_wise_feature_counts.tsv")
        sample_wise_stats.to_csv(sample_wise_file, sep='\t', index=False)
    
    # Save sample-wise abundance statistics if available
    if sample_wise_abundance is not None:
        sample_abundance_file = os.path.join(output_dir, "sample_wise_abundance.tsv")
        sample_wise_abundance.to_csv(sample_abundance_file, sep='\t', index=False)
    
    # Save discrepancy analysis if available
    if discrepancy_analysis is not None:
        discrepancy_file = os.path.join(output_dir, "feature_abundance_discrepancy.tsv")
        discrepancy_analysis.to_csv(discrepancy_file, sep='\t', index=False)
    
    # Save abundance-weighted statistics if available
    if abundance_weighted_stats is not None:
        abundance_file = os.path.join(output_dir, "abundance_weighted_stats.tsv")
        abundance_weighted_stats.to_csv(abundance_file, sep='\t', index=False)
        
        # Create per-sample summary
        sample_summary = abundance_weighted_stats.groupby('SampleID').agg({
            'AbsoluteAbundance': 'sum',
            'TotalReads': 'first'
        }).reset_index()
        sample_summary_file = os.path.join(output_dir, "sample_summary.tsv")
        sample_summary.to_csv(sample_summary_file, sep='\t', index=False)
    
    # Save F/B ratio analysis if available
    if fb_ratios is not None and not fb_ratios.empty:
        fb_ratios_file = os.path.join(output_dir, "fb_ratios.tsv")
        fb_ratios.to_csv(fb_ratios_file, sep='\t', index=False)
    
    # Save extraction efficiency ratios if available
    if extraction_ratios is not None and not extraction_ratios.empty:
        extraction_ratios_file = os.path.join(output_dir, "extraction_ratios.tsv")
        extraction_ratios.to_csv(extraction_ratios_file, sep='\t', index=False)
    
    # Save bias assessment results if available
    if bias_assessment is not None:
        if not bias_assessment['fb_ratio_bias'].empty:
            fb_bias_file = os.path.join(output_dir, "fb_ratio_bias_assessment.tsv")
            bias_assessment['fb_ratio_bias'].to_csv(fb_bias_file, sep='\t', index=False)
        
        if not bias_assessment['missing_taxa_assessment'].empty:
            missing_taxa_file = os.path.join(output_dir, "missing_taxa_assessment.tsv")
            bias_assessment['missing_taxa_assessment'].to_csv(missing_taxa_file, sep='\t', index=False)
        
        if not bias_assessment['sample_warnings'].empty:
            sample_warnings_file = os.path.join(output_dir, "sample_warnings.tsv")
            bias_assessment['sample_warnings'].to_csv(sample_warnings_file, sep='\t', index=False)
    
    # Save metadata analysis if available  
    if metadata_analysis is not None and metadata_analysis.get('group_comparisons') is not None:
        group_comparisons_file = os.path.join(output_dir, "group_comparisons.tsv")
        metadata_analysis['group_comparisons'].to_csv(group_comparisons_file, sep='\t', index=False)
    
    # Generate classification statistics for backwards compatibility
    classification_stats = detailed_classification.groupby(['ClassificationLevel', 'GramStatus']).size().unstack(fill_value=0)
    classification_stats_file = os.path.join(output_dir, "classification_level_stats.tsv")
    classification_stats.to_csv(classification_stats_file, sep='\t')
    
    # Create HTML report using q2templates
    gram_templates = os.path.join(TEMPLATES, 'gram_staining')
    index_fp = os.path.join(gram_templates, 'index.html')
    
    # Debug: Check if template file exists  
    if not os.path.exists(index_fp):
        raise FileNotFoundError(f"Template file not found: {index_fp}. Available templates: {os.listdir(gram_templates) if os.path.exists(gram_templates) else 'Template directory not found'}")
    
    # Create summary tables as HTML
    summary_html = summary_stats.to_html(classes=["table", "table-striped", "table-hover"])
    
    # Create detailed classification sample (first 100 rows)
    sample_detailed = detailed_classification.head(100) if len(detailed_classification) > 100 else detailed_classification
    detailed_html = sample_detailed.to_html(classes=["table", "table-striped", "table-hover"], index=False)
    
    # Create level breakdown HTML (both types)
    classification_source_html = {}
    taxonomic_resolution_html = {}
    comprehensive_analysis_html = {}
    
    for level, breakdown in level_breakdown['classification_source'].items():
        if isinstance(breakdown, pd.DataFrame) and not breakdown.empty:
            classification_source_html[level] = breakdown.to_html(classes=["table", "table-striped", "table-hover"])
    
    for level, breakdown in level_breakdown['taxonomic_resolution'].items():
        if isinstance(breakdown, pd.DataFrame) and not breakdown.empty:
            taxonomic_resolution_html[level] = breakdown.to_html(classes=["table", "table-striped", "table-hover"])
    
    # Create comprehensive analysis HTML
    for level, distribution in comprehensive_level_analysis.items():
        if isinstance(distribution, pd.DataFrame) and not distribution.empty:
            # Remove the TaxonomicLevel column for display
            display_df = distribution.drop(columns=['TaxonomicLevel'], errors='ignore')
            comprehensive_analysis_html[level] = display_df.to_html(classes=["table", "table-striped", "table-hover"])
    
    # Classification level statistics
    class_stats_html = classification_stats.to_html(classes=["table", "table-striped", "table-hover"])
    
    # Confidence statistics
    confidence_html = confidence_summary.to_html(classes=["table", "table-striped", "table-hover"], index=False)
    
    # Unique taxa summaries
    unique_taxa_html = {}
    for level, taxa_summary in unique_taxa_summary.items():
        if isinstance(taxa_summary, pd.DataFrame) and not taxa_summary.empty:
            unique_taxa_html[level] = taxa_summary.to_html(classes=["table", "table-striped", "table-hover"], index=False)
    
    # Database coverage statistics
    database_coverage_html = None
    if database_coverage:
        coverage_rows = []
        for level, stats in database_coverage.items():
            coverage_rows.append({
                'Level': level,
                'Unique Taxa in Data': stats['unique_taxa_in_data'],
                'Taxa in Database': stats['taxa_in_database'],
                'Coverage %': f"{stats['coverage_percentage']:.1f}%",
                'Database Size': stats['database_size']
            })
        if coverage_rows:
            coverage_df = pd.DataFrame(coverage_rows)
            database_coverage_html = coverage_df.to_html(classes=["table", "table-striped", "table-hover"], index=False)
    
    # Classification pipeline statistics
    pipeline_html = pipeline_stats.to_html(classes=["table", "table-striped", "table-hover"], index=False)
    
    # Abundance-weighted statistics
    abundance_html = None
    sample_summary_html = None
    if abundance_weighted_stats is not None:
        # Create pivot table for better display
        abundance_pivot = abundance_weighted_stats.pivot_table(
            index='SampleID', 
            columns='GramStatus', 
            values='RelativeAbundance',
            fill_value=0
        ).round(2)
        abundance_html = abundance_pivot.to_html(classes=["table", "table-striped", "table-hover"])
        
        # Sample summary
        sample_summary = abundance_weighted_stats.groupby('SampleID').agg({
            'TotalReads': 'first'
        }).reset_index()
        sample_summary_html = sample_summary.to_html(classes=["table", "table-striped", "table-hover"], index=False)
    
    # F/B ratio analysis HTML
    fb_ratios_html = None
    if fb_ratios is not None and not fb_ratios.empty:
        fb_ratios_html = fb_ratios.to_html(classes=["table", "table-striped", "table-hover"], index=False)
    
    # Extraction efficiency ratios HTML
    extraction_ratios_html = None
    if extraction_ratios is not None and not extraction_ratios.empty:
        extraction_ratios_html = extraction_ratios.to_html(classes=["table", "table-striped", "table-hover"], index=False)
    
    # Bias assessment HTML
    bias_assessment_html = {}
    systematic_bias_html = None
    if bias_assessment is not None:
        if not bias_assessment['fb_ratio_bias'].empty:
            bias_assessment_html['fb_ratio_bias'] = bias_assessment['fb_ratio_bias'].to_html(classes=["table", "table-striped", "table-hover"], index=False)
        if not bias_assessment['missing_taxa_assessment'].empty:
            bias_assessment_html['missing_taxa'] = bias_assessment['missing_taxa_assessment'].to_html(classes=["table", "table-striped", "table-hover"], index=False)
        if not bias_assessment['sample_warnings'].empty:
            bias_assessment_html['sample_warnings'] = bias_assessment['sample_warnings'].to_html(classes=["table", "table-striped", "table-hover"], index=False)
        
        # Systematic bias assessment summary
        if bias_assessment.get('systematic_bias_assessment'):
            systematic_bias_summary = bias_assessment['systematic_bias_assessment']
            if systematic_bias_summary:
                # Create a summary DataFrame for display
                summary_data = []
                if 'dataset_bias_summary' in systematic_bias_summary:
                    for analysis_type, data in systematic_bias_summary['dataset_bias_summary'].items():
                        summary_data.append({
                            'Analysis Type': analysis_type.replace('_', ' ').title(),
                            'Details': str(data)
                        })
                if 'extraction_quality_flags' in systematic_bias_summary:
                    for flag in systematic_bias_summary['extraction_quality_flags']:
                        summary_data.append({
                            'Analysis Type': 'Quality Flag',
                            'Details': flag
                        })
                if summary_data:
                    systematic_df = pd.DataFrame(summary_data)
                    systematic_bias_html = systematic_df.to_html(classes=["table", "table-striped", "table-hover"], index=False)
    
    # Metadata analysis HTML
    metadata_summary_html = None
    group_comparisons_html = None
    if metadata_analysis is not None:
        # Metadata summary
        if metadata_analysis.get('metadata_summary'):
            metadata_summary = metadata_analysis['metadata_summary']
            summary_data = []
            summary_data.append({'Property': 'Total Metadata Samples', 'Value': metadata_summary.get('total_metadata_samples', 0)})
            summary_data.append({'Property': 'Common Samples', 'Value': metadata_summary.get('common_samples', 0)})
            summary_data.append({'Property': 'Overlap Percentage', 'Value': f"{metadata_summary.get('overlap_percentage', 0)}%"})
            summary_data.append({'Property': 'Suitable Group Columns', 'Value': ', '.join(metadata_summary.get('suitable_group_columns', []))})
            metadata_summary_df = pd.DataFrame(summary_data)
            metadata_summary_html = metadata_summary_df.to_html(classes=["table", "table-striped", "table-hover"], index=False)
        
        # Group comparisons
        if metadata_analysis.get('group_comparisons') is not None:
            group_comparisons_html = metadata_analysis['group_comparisons'].to_html(classes=["table", "table-striped", "table-hover"], index=False)
    
    context = {
        'title': 'Comprehensive Gram Staining Analysis with Advanced Diagnostics',
        'total_features': len(detailed_classification),
        'summary_stats': summary_stats,
        'summary_html': summary_html,
        'detailed_html': detailed_html,
        'classification_source_html': classification_source_html,
        'taxonomic_resolution_html': taxonomic_resolution_html,
        'comprehensive_analysis_html': comprehensive_analysis_html,
        'class_stats_html': class_stats_html,
        'confidence_html': confidence_html,
        'unique_taxa_html': unique_taxa_html,
        'database_coverage_html': database_coverage_html,
        'pipeline_html': pipeline_html,
        'abundance_html': abundance_html,
        'sample_summary_html': sample_summary_html,
        'classification_source_levels': list(level_breakdown['classification_source'].keys()),
        'taxonomic_resolution_levels': list(level_breakdown['taxonomic_resolution'].keys()),
        'comprehensive_analysis_levels': list(comprehensive_level_analysis.keys()),
        'classification_levels': list(classification_stats.index),
        'unique_taxa_levels': list(unique_taxa_summary.keys()),
        'has_detailed_sample': len(detailed_classification) > 100,
        'has_abundance_data': abundance_weighted_stats is not None,
        'has_database_coverage': database_coverage_html is not None,
        'confidence_summary': confidence_summary,
        'database_coverage': database_coverage,
        'pipeline_stats': pipeline_stats,
        'classification_examples': classification_examples,
        'is_abundance_weighted': feature_table is not None,
        'abundance_summary_stats': abundance_summary_stats,
        'abundance_comprehensive_analysis': abundance_comprehensive_analysis,
        'sample_wise_stats': sample_wise_stats,
        'sample_wise_abundance': sample_wise_abundance,
        'discrepancy_analysis': discrepancy_analysis,
        # Enhanced analysis features
        'fb_ratios': fb_ratios,
        'fb_ratios_html': fb_ratios_html,
        'extraction_ratios': extraction_ratios,
        'extraction_ratios_html': extraction_ratios_html,
        'bias_assessment': bias_assessment,
        'bias_assessment_html': bias_assessment_html,
        'systematic_bias_html': systematic_bias_html,
        'metadata_analysis': metadata_analysis,
        'metadata_summary_html': metadata_summary_html,
        'group_comparisons_html': group_comparisons_html,
        # Feature flags for template
        'has_ratio_analysis': fb_ratios is not None and not fb_ratios.empty,
        'has_bias_assessment': bias_assessment is not None,
        'has_metadata_analysis': metadata_analysis is not None,
        'has_systematic_bias': systematic_bias_html is not None,
        'has_group_comparisons': group_comparisons_html is not None
    }
    
    # Create HTML report
    q2templates.render(index_fp, output_dir, context=context)


# Define the templates directory
TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
