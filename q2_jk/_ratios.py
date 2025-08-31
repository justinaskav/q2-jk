"""
Phylum-level ratio calculations for microbiome analysis.

This module contains functions for calculating key microbiome ratios
including F/B ratios, Gram+/Gram- ratios, and extraction efficiency markers.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from ._methods import get_taxonomic_levels, normalize_taxonomy_assignment


def _extract_phylum_from_taxonomy(taxonomy_string: str) -> str:
    """
    Extract phylum from a full taxonomy string.
    
    Parameters
    ----------
    taxonomy_string : str
        Full taxonomy string (e.g., "k__Bacteria; p__Firmicutes; c__Bacilli; ...")
        
    Returns
    -------
    str
        Normalized phylum name or 'Unclassified'
    """
    if not taxonomy_string or pd.isna(taxonomy_string):
        return 'Unclassified'
    
    # Split by semicolon and look for phylum level
    parts = [part.strip() for part in str(taxonomy_string).split(';')]
    
    for part in parts:
        if part.startswith('p__'):
            phylum = part[3:].strip()  # Remove 'p__' prefix
            if phylum and phylum.lower() not in ['', 'unknown', 'unassigned', 'unclassified']:
                # Normalize common names
                phylum_normalized = normalize_taxonomy_assignment(phylum)
                
                # Handle both modern (GTDB) and legacy names
                if phylum_normalized in ['firmicutes', 'bacillota']:
                    return 'Bacillota'
                elif phylum_normalized in ['bacteroidetes', 'bacteroidota']:
                    return 'Bacteroidetes'  
                elif phylum_normalized in ['actinobacteria', 'actinobacteriota']:
                    return 'Actinomycetota'  # Use modern name
                elif phylum_normalized in ['proteobacteria', 'pseudomonadota']:
                    return 'Proteobacteria'
                else:
                    return phylum_normalized.capitalize()
    
    return 'Unclassified'


def extract_phylum_abundances(taxonomy_df: pd.DataFrame, feature_table: Optional[pd.DataFrame] = None, col_name: str = 'Taxon') -> pd.DataFrame:
    """
    Extract phylum-level abundances from taxonomy data.
    
    Parameters
    ----------
    taxonomy_df : pd.DataFrame
        DataFrame containing taxonomy data
    feature_table : pd.DataFrame, optional
        Feature abundance table (features x samples)
    col_name : str, optional
        Column containing taxonomy strings
        
    Returns
    -------
    pd.DataFrame
        Phylum-level abundance data
    """
    # Get taxonomic levels
    tax_levels = get_taxonomic_levels(taxonomy_df, col_name)
    
    # Extract and normalize phylum assignments
    phylum_data = []
    for idx in tax_levels.index:
        phylum_raw = tax_levels.loc[idx, 'Phylum']
        phylum_normalized = normalize_taxonomy_assignment(phylum_raw)
        
        # Map to standardized phylum names
        phylum_name = _map_phylum_name(phylum_normalized)
        
        phylum_info = {
            'FeatureID': idx,
            'Phylum_Raw': phylum_raw,
            'Phylum_Normalized': phylum_normalized,
            'Phylum_Standard': phylum_name
        }
        
        # Add abundance data if available
        if feature_table is not None and idx in feature_table.index:
            total_abundance = feature_table.loc[idx].sum()
            phylum_info['TotalAbundance'] = total_abundance
        else:
            phylum_info['TotalAbundance'] = 0
        
        phylum_data.append(phylum_info)
    
    return pd.DataFrame(phylum_data)


def _map_phylum_name(phylum_normalized: str) -> str:
    """
    Map normalized phylum names to standardized names.
    Handles both GTDB and legacy naming conventions.
    
    Parameters
    ----------
    phylum_normalized : str
        Normalized phylum assignment
        
    Returns
    -------
    str
        Standardized phylum name
    """
    if not phylum_normalized:
        return 'Unclassified'
    
    phylum_lower = phylum_normalized.lower()
    
    # Map common phyla to standard names
    phylum_mapping = {
        'bacillota': 'Bacillota',  # GTDB name for Firmicutes
        'firmicutes': 'Bacillota',  # Legacy name -> GTDB
        'bacteroidota': 'Bacteroidetes',  # GTDB name
        'bacteroidetes': 'Bacteroidetes',  # Legacy name
        'actinobacteriota': 'Actinomycetota',  # GTDB name
        'actinobacteria': 'Actinomycetota',  # Legacy name -> modern
        'proteobacteria': 'Proteobacteria',
        'verrucomicrobiota': 'Verrucomicrobia',  # GTDB name
        'verrucomicrobia': 'Verrucomicrobia',  # Legacy name
        'desulfobacterota': 'Desulfobacterota',  # GTDB name
        'fusobacteriota': 'Fusobacteria',  # GTDB name
        'fusobacteria': 'Fusobacteria',  # Legacy name
        'planctomycetota': 'Planctomycetes',  # GTDB name
        'planctomycetes': 'Planctomycetes',  # Legacy name
        'cyanobacteria': 'Cyanobacteria',
        'spirochaetota': 'Spirochaetes',  # GTDB name
        'spirochaetes': 'Spirochaetes',  # Legacy name
        'synergistota': 'Synergistetes',  # GTDB name
        'synergistetes': 'Synergistetes',  # Legacy name
    }
    
    # Check for direct matches
    for key, standard_name in phylum_mapping.items():
        if key in phylum_lower:
            return standard_name
    
    # If no match found, return capitalized version
    return phylum_normalized.capitalize()


def calculate_fb_ratios(phylum_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Bacillota/Bacteroidetes (F/B) ratios for each sample.
    
    Parameters
    ----------
    phylum_data : pd.DataFrame
        Phylum-level abundance data
        
    Returns
    -------
    pd.DataFrame
        F/B ratios per sample
    """
    if 'TotalAbundance' not in phylum_data.columns:
        raise ValueError("TotalAbundance column required for ratio calculations")
    
    # Group by phylum and sum abundances
    phylum_totals = phylum_data.groupby('Phylum_Standard')['TotalAbundance'].sum()
    
    # Get Bacillota and Bacteroidetes abundances
    bacillota_abundance = phylum_totals.get('Bacillota', 0)
    bacteroidetes_abundance = phylum_totals.get('Bacteroidetes', 0)
    
    # Calculate F/B ratio
    if bacteroidetes_abundance > 0:
        fb_ratio = bacillota_abundance / bacteroidetes_abundance
    else:
        fb_ratio = float('inf') if bacillota_abundance > 0 else 0
    
    # Create results dataframe
    ratio_data = {
        'Bacillota_Abundance': bacillota_abundance,
        'Bacteroidetes_Abundance': bacteroidetes_abundance,
        'FB_Ratio': fb_ratio,
        'FB_Ratio_Log10': np.log10(fb_ratio) if fb_ratio > 0 and fb_ratio != float('inf') else None
    }
    
    return pd.DataFrame([ratio_data])


def calculate_sample_wise_fb_ratios(detailed_classification: pd.DataFrame, feature_table: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate F/B ratios for each sample individually.
    
    Parameters
    ----------
    detailed_classification : pd.DataFrame
        Detailed classification data with FeatureID and FullTaxonomy
    feature_table : pd.DataFrame
        Feature abundance table (features x samples)
        
    Returns
    -------
    pd.DataFrame
        Sample-wise F/B ratios
    """
    # Extract phylum information from detailed classification
    phylum_mapping = {}
    for _, row in detailed_classification.iterrows():
        feature_id = row['FeatureID']
        full_taxonomy = row['FullTaxonomy']
        
        # Parse phylum from taxonomy string
        phylum = _extract_phylum_from_taxonomy(full_taxonomy)
        phylum_mapping[feature_id] = phylum
    
    # Get common features
    common_features = list(set(phylum_mapping.keys()).intersection(set(feature_table.index)))
    
    if not common_features:
        return pd.DataFrame()
    
    # Calculate ratios for each sample
    sample_results = []
    for sample_id in feature_table.columns:
        sample_abundances = feature_table.loc[common_features, sample_id]
        
        # Sum abundances by phylum
        phylum_abundances = {}
        for feature_id, abundance in sample_abundances.items():
            phylum = phylum_mapping.get(feature_id, 'Unclassified')
            if phylum not in phylum_abundances:
                phylum_abundances[phylum] = 0
            phylum_abundances[phylum] += abundance
        
        # Calculate F/B ratio
        bacillota = phylum_abundances.get('Bacillota', 0)
        bacteroidetes = phylum_abundances.get('Bacteroidetes', 0)
        
        if bacteroidetes > 0:
            fb_ratio = bacillota / bacteroidetes
        else:
            fb_ratio = float('inf') if bacillota > 0 else 0
        
        # Determine clinical interpretation
        clinical_interpretation = _interpret_fb_ratio(fb_ratio)
        
        sample_result = {
            'SampleID': sample_id,
            'Bacillota_Abundance': int(bacillota),
            'Bacteroidetes_Abundance': int(bacteroidetes),
            'FB_Ratio': round(fb_ratio, 4) if fb_ratio != float('inf') else fb_ratio,
            'FB_Ratio_Log10': round(np.log10(fb_ratio), 3) if fb_ratio > 0 and fb_ratio != float('inf') else None,
            'Clinical_Interpretation': clinical_interpretation
        }
        sample_results.append(sample_result)
    
    return pd.DataFrame(sample_results)


def calculate_extraction_efficiency_ratios(detailed_classification: pd.DataFrame, feature_table: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ratios that indicate DNA extraction efficiency.
    Bacillota/Actinomycetota ratio is key - Actinomycetota are hardest to lyse.
    
    Parameters
    ----------
    detailed_classification : pd.DataFrame
        Detailed classification data with FeatureID and FullTaxonomy
    feature_table : pd.DataFrame
        Feature abundance table (features x samples)
        
    Returns
    -------
    pd.DataFrame
        Extraction efficiency ratios per sample
    """
    # Extract phylum information from detailed classification
    phylum_mapping = {}
    for _, row in detailed_classification.iterrows():
        feature_id = row['FeatureID']
        full_taxonomy = row['FullTaxonomy']
        
        # Parse phylum from taxonomy string
        phylum = _extract_phylum_from_taxonomy(full_taxonomy)
        phylum_mapping[feature_id] = phylum
    
    # Get common features
    common_features = list(set(phylum_mapping.keys()).intersection(set(feature_table.index)))
    
    if not common_features:
        return pd.DataFrame()
    
    # Calculate extraction ratios for each sample
    sample_results = []
    for sample_id in feature_table.columns:
        sample_abundances = feature_table.loc[common_features, sample_id]
        
        # Sum abundances by phylum
        phylum_abundances = {}
        for feature_id, abundance in sample_abundances.items():
            phylum = phylum_mapping.get(feature_id, 'Unclassified')
            if phylum not in phylum_abundances:
                phylum_abundances[phylum] = 0
            phylum_abundances[phylum] += abundance
        
        # Key extraction efficiency markers
        bacillota = phylum_abundances.get('Bacillota', 0)
        actinomycetota = phylum_abundances.get('Actinomycetota', 0)
        
        # Bacillota/Actinomycetota ratio (key extraction efficiency marker)
        if actinomycetota > 0:
            ba_ratio = bacillota / actinomycetota
        else:
            ba_ratio = float('inf') if bacillota > 0 else 0
        
        # Overall Gram+/Gram- ratio at phylum level
        gram_positive_phyla = ['Bacillota', 'Actinomycetota']
        gram_negative_phyla = ['Bacteroidetes', 'Proteobacteria', 'Verrucomicrobia', 'Fusobacteria']
        
        gram_positive_total = sum(phylum_abundances.get(p, 0) for p in gram_positive_phyla)
        gram_negative_total = sum(phylum_abundances.get(p, 0) for p in gram_negative_phyla)
        
        if gram_negative_total > 0:
            gp_gn_ratio = gram_positive_total / gram_negative_total
        else:
            gp_gn_ratio = float('inf') if gram_positive_total > 0 else 0
        
        # Assess extraction quality
        extraction_quality = _assess_extraction_quality(ba_ratio, actinomycetota, gp_gn_ratio)
        
        sample_result = {
            'SampleID': sample_id,
            'Bacillota_Abundance': int(bacillota),
            'Actinomycetota_Abundance': int(actinomycetota),
            'BA_Ratio': round(ba_ratio, 4) if ba_ratio != float('inf') else ba_ratio,
            'GramPos_Total': int(gram_positive_total),
            'GramNeg_Total': int(gram_negative_total),
            'GramPos_GramNeg_Ratio': round(gp_gn_ratio, 4) if gp_gn_ratio != float('inf') else gp_gn_ratio,
            'Extraction_Quality': extraction_quality
        }
        sample_results.append(sample_result)
    
    return pd.DataFrame(sample_results)


def _interpret_fb_ratio(fb_ratio: float) -> str:
    """
    Provide clinical interpretation of F/B ratio.
    
    Parameters
    ----------
    fb_ratio : float
        Bacillota/Bacteroidetes ratio
        
    Returns
    -------
    str
        Clinical interpretation
    """
    if fb_ratio == 0:
        return "No Bacillota detected"
    elif fb_ratio == float('inf'):
        return "No Bacteroidetes detected"
    elif fb_ratio < 0.2:
        return "Extremely low - possible Gram+ extraction issues"
    elif fb_ratio < 0.5:
        return "Low - may indicate dysbiosis or extraction bias"
    elif fb_ratio <= 1.5:
        return "Normal range for healthy adults"
    elif fb_ratio <= 5:
        return "Elevated - possible Gram+ overrepresentation"
    elif fb_ratio <= 10:
        return "High - investigate extraction protocol"
    else:
        return "Extremely high - likely technical artifact"


def _assess_extraction_quality(ba_ratio: float, actinomycetota_abundance: float, gp_gn_ratio: float) -> str:
    """
    Assess DNA extraction quality based on multiple markers.
    
    Parameters
    ----------
    ba_ratio : float
        Bacillota/Actinomycetota ratio
    actinomycetota_abundance : float
        Total Actinomycetota abundance
    gp_gn_ratio : float
        Gram-positive/Gram-negative ratio
        
    Returns
    -------
    str
        Extraction quality assessment
    """
    # Check for missing Actinomycetota (strongest indicator of extraction failure)
    if actinomycetota_abundance == 0:
        return "Poor - No Actinomycetota detected (extraction failure likely)"
    
    # Check for extreme ratios
    if gp_gn_ratio < 0.1:
        return "Poor - Severe Gram+ underrepresentation"
    elif gp_gn_ratio > 20:
        return "Poor - Extreme Gram+ overrepresentation"
    elif ba_ratio > 100:
        return "Questionable - Very high BA ratio"
    elif 0.1 <= gp_gn_ratio <= 10:
        return "Good - Balanced extraction efficiency"
    else:
        return "Moderate - Some extraction bias possible"


def calculate_all_ratios(taxonomy_df: pd.DataFrame, feature_table: Optional[pd.DataFrame] = None, col_name: str = 'Taxon') -> Dict[str, Any]:
    """
    Calculate all phylum-level ratios and return comprehensive results.
    
    Parameters
    ----------
    taxonomy_df : pd.DataFrame
        DataFrame containing taxonomy data
    feature_table : pd.DataFrame, optional
        Feature abundance table (features x samples)
    col_name : str, optional
        Column containing taxonomy strings
        
    Returns
    -------
    dict
        Dictionary containing all ratio analysis results
    """
    # Extract phylum-level data
    phylum_data = extract_phylum_abundances(taxonomy_df, feature_table, col_name)
    
    results = {
        'phylum_data': phylum_data,
        'sample_fb_ratios': None,
        'extraction_ratios': None,
        'overall_fb_ratio': None
    }
    
    if feature_table is not None:
        # Calculate sample-wise F/B ratios
        results['sample_fb_ratios'] = calculate_sample_wise_fb_ratios(phylum_data, feature_table)
        
        # Calculate extraction efficiency ratios
        results['extraction_ratios'] = calculate_extraction_efficiency_ratios(phylum_data, feature_table)
        
        # Calculate overall F/B ratio
        results['overall_fb_ratio'] = calculate_fb_ratios(phylum_data)
    
    return results