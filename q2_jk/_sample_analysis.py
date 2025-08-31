"""
Sample-wise analysis functions for gram staining composition.

This module contains functions for calculating sample-level statistics,
including feature counts, abundance calculations, and discrepancy analysis.
"""

import pandas as pd
from typing import Optional


def calculate_abundance_weighted_summary(detailed_classification: pd.DataFrame, feature_table: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Calculate abundance-weighted summary statistics for gram staining categories.
    
    Parameters
    ----------
    detailed_classification : pd.DataFrame
        Per-feature classification results
    feature_table : pd.DataFrame
        Feature abundance table (features x samples)
        
    Returns
    -------
    pd.DataFrame or None
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


def calculate_sample_wise_feature_counts(detailed_classification: pd.DataFrame, feature_table: pd.DataFrame) -> Optional[pd.DataFrame]:
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
    pd.DataFrame or None
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


def calculate_sample_wise_abundance(detailed_classification: pd.DataFrame, feature_table: pd.DataFrame) -> Optional[pd.DataFrame]:
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
    pd.DataFrame or None
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


def calculate_feature_abundance_discrepancy(sample_wise_stats: pd.DataFrame, sample_wise_abundance: pd.DataFrame) -> Optional[pd.DataFrame]:
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
    pd.DataFrame or None
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