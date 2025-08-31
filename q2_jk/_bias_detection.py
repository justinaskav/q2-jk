"""
DNA extraction bias detection and assessment functions.

This module provides automated detection of systematic biases in DNA extraction
that may affect microbiome analysis results, particularly Gram-positive underrepresentation.
"""

import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
import numpy as np


def assess_fb_ratio_bias(fb_ratios: pd.DataFrame, 
                        low_threshold: float = 0.2, 
                        high_threshold: float = 10.0) -> pd.DataFrame:
    """
    Assess F/B ratio bias patterns across samples.
    
    Parameters
    ----------
    fb_ratios : pd.DataFrame
        Sample-wise F/B ratios
    low_threshold : float, optional
        Threshold below which F/B ratios suggest Gram+ loss
    high_threshold : float, optional
        Threshold above which F/B ratios suggest Gram+ overrepresentation
        
    Returns
    -------
    pd.DataFrame
        F/B ratio bias assessment for each sample
    """
    if 'FB_Ratio' not in fb_ratios.columns:
        return pd.DataFrame()
    
    bias_results = []
    
    for _, row in fb_ratios.iterrows():
        sample_id = row['SampleID']
        fb_ratio = row['FB_Ratio']
        
        # Determine bias level
        if fb_ratio == float('inf'):
            bias_level = "Severe"
            bias_type = "No Bacteroidetes detected"
        elif fb_ratio == 0:
            bias_level = "Severe"
            bias_type = "No Bacillota detected"
        elif fb_ratio < low_threshold:
            bias_level = "High" if fb_ratio < 0.1 else "Moderate"
            bias_type = "Bacillota underrepresentation"
        elif fb_ratio > high_threshold:
            bias_level = "High" if fb_ratio > 20 else "Moderate"
            bias_type = "Bacillota overrepresentation"
        else:
            bias_level = "Minimal"
            bias_type = "Normal range"
        
        # Clinical interpretation
        clinical_flag = _get_clinical_flag(fb_ratio, bias_level)
        
        bias_result = {
            'SampleID': sample_id,
            'FB_Ratio': fb_ratio,
            'Bias_Level': bias_level,
            'Bias_Type': bias_type,
            'Clinical_Flag': clinical_flag,
            'Extraction_Concern': bias_level in ['High', 'Severe']
        }
        bias_results.append(bias_result)
    
    return pd.DataFrame(bias_results)


def detect_missing_key_taxa(extraction_ratios: pd.DataFrame) -> pd.DataFrame:
    """
    Detect missing key bacterial taxa that indicate extraction problems.
    
    Parameters
    ----------
    extraction_ratios : pd.DataFrame
        Extraction efficiency ratios including Actinomycetota data
        
    Returns
    -------
    pd.DataFrame
        Missing taxa assessment for each sample
    """
    if 'Actinomycetota_Abundance' not in extraction_ratios.columns:
        return pd.DataFrame()
    
    missing_taxa_results = []
    
    for _, row in extraction_ratios.iterrows():
        sample_id = row['SampleID']
        actinomycetota = row['Actinomycetota_Abundance']
        bacillota = row['Bacillota_Abundance']
        
        # Check for missing Actinomycetota (strongest indicator)
        missing_actino = actinomycetota == 0
        low_actino = actinomycetota > 0 and actinomycetota < 10
        
        # Check for missing Bacillota
        missing_bacillota = bacillota == 0
        
        # Overall assessment
        if missing_actino and missing_bacillota:
            severity = "Severe"
            issue = "Both Actinomycetota and Bacillota absent"
        elif missing_actino:
            severity = "High"
            issue = "Actinomycetota absent (tough cell wall indicator)"
        elif missing_bacillota:
            severity = "Moderate"  
            issue = "Bacillota absent"
        elif low_actino:
            severity = "Low"
            issue = "Very low Actinomycetota abundance"
        else:
            severity = "Minimal"
            issue = "Key taxa present"
        
        missing_result = {
            'SampleID': sample_id,
            'Missing_Actinomycetota': missing_actino,
            'Missing_Bacillota': missing_bacillota,
            'Low_Actinomycetota': low_actino,
            'Severity': severity,
            'Issue_Description': issue,
            'Extraction_Failure_Risk': severity in ['High', 'Severe']
        }
        missing_taxa_results.append(missing_result)
    
    return pd.DataFrame(missing_taxa_results)


def assess_systematic_bias(discrepancy_data: pd.DataFrame, 
                          ratio_bias_data: pd.DataFrame,
                          missing_taxa_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Assess systematic bias patterns across the entire dataset.
    
    Parameters
    ----------
    discrepancy_data : pd.DataFrame
        Feature-abundance discrepancy analysis
    ratio_bias_data : pd.DataFrame
        F/B ratio bias assessment
    missing_taxa_data : pd.DataFrame
        Missing key taxa assessment
        
    Returns
    -------
    dict
        Systematic bias assessment
    """
    assessment = {
        'dataset_bias_summary': {},
        'extraction_quality_flags': [],
        'recommendations': []
    }
    
    # Analyze discrepancy patterns
    if not discrepancy_data.empty:
        high_bias_samples = len(discrepancy_data[discrepancy_data['Bias_Likelihood'] == 'High'])
        moderate_bias_samples = len(discrepancy_data[discrepancy_data['Bias_Likelihood'] == 'Moderate'])
        total_samples = len(discrepancy_data)
        
        high_bias_pct = (high_bias_samples / total_samples * 100) if total_samples > 0 else 0
        moderate_plus_high_pct = ((high_bias_samples + moderate_bias_samples) / total_samples * 100) if total_samples > 0 else 0
        
        assessment['dataset_bias_summary']['discrepancy_analysis'] = {
            'total_samples': total_samples,
            'high_bias_samples': high_bias_samples,
            'moderate_bias_samples': moderate_bias_samples,
            'high_bias_percentage': round(high_bias_pct, 1),
            'moderate_plus_high_percentage': round(moderate_plus_high_pct, 1)
        }
        
        # Flag systematic issues
        if high_bias_pct > 50:
            assessment['extraction_quality_flags'].append("Severe: >50% of samples show high extraction bias")
        elif high_bias_pct > 25:
            assessment['extraction_quality_flags'].append("Moderate: >25% of samples show high extraction bias")
        
        if moderate_plus_high_pct > 75:
            assessment['extraction_quality_flags'].append("Critical: >75% of samples show extraction bias")
    
    # Analyze F/B ratio patterns
    if not ratio_bias_data.empty:
        extreme_ratios = len(ratio_bias_data[ratio_bias_data['Bias_Level'].isin(['High', 'Severe'])])
        total_samples = len(ratio_bias_data)
        extreme_pct = (extreme_ratios / total_samples * 100) if total_samples > 0 else 0
        
        assessment['dataset_bias_summary']['fb_ratio_analysis'] = {
            'total_samples': total_samples,
            'extreme_ratio_samples': extreme_ratios,
            'extreme_ratio_percentage': round(extreme_pct, 1)
        }
        
        if extreme_pct > 40:
            assessment['extraction_quality_flags'].append("High: >40% of samples have extreme F/B ratios")
    
    # Analyze missing taxa patterns
    if not missing_taxa_data.empty:
        missing_actino_samples = len(missing_taxa_data[missing_taxa_data['Missing_Actinomycetota']])
        total_samples = len(missing_taxa_data)
        missing_actino_pct = (missing_actino_samples / total_samples * 100) if total_samples > 0 else 0
        
        assessment['dataset_bias_summary']['missing_taxa_analysis'] = {
            'total_samples': total_samples,
            'missing_actinomycetota_samples': missing_actino_samples,
            'missing_actinomycetota_percentage': round(missing_actino_pct, 1)
        }
        
        if missing_actino_pct > 30:
            assessment['extraction_quality_flags'].append("Critical: >30% of samples missing Actinomycetota")
        elif missing_actino_pct > 10:
            assessment['extraction_quality_flags'].append("Moderate: >10% of samples missing Actinomycetota")
    
    # Generate recommendations
    assessment['recommendations'] = _generate_bias_recommendations(assessment['extraction_quality_flags'])
    
    # Overall assessment
    n_flags = len(assessment['extraction_quality_flags'])
    if n_flags == 0:
        assessment['overall_assessment'] = "Good - No systematic extraction bias detected"
    elif n_flags <= 2:
        assessment['overall_assessment'] = "Moderate - Some extraction bias indicators present"
    else:
        assessment['overall_assessment'] = "Poor - Multiple extraction bias indicators suggest protocol issues"
    
    return assessment


def generate_sample_bias_warnings(discrepancy_data: pd.DataFrame,
                                 ratio_bias_data: pd.DataFrame,
                                 missing_taxa_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate consolidated bias warnings for each sample.
    
    Parameters
    ----------
    discrepancy_data : pd.DataFrame
        Feature-abundance discrepancy analysis
    ratio_bias_data : pd.DataFrame
        F/B ratio bias assessment
    missing_taxa_data : pd.DataFrame
        Missing key taxa assessment
        
    Returns
    -------
    pd.DataFrame
        Consolidated sample-wise bias warnings
    """
    # Get all unique sample IDs
    all_samples = set()
    for df in [discrepancy_data, ratio_bias_data, missing_taxa_data]:
        if not df.empty and 'SampleID' in df.columns:
            all_samples.update(df['SampleID'].tolist())
    
    if not all_samples:
        return pd.DataFrame()
    
    warning_results = []
    
    for sample_id in all_samples:
        warnings = []
        overall_risk = "Low"
        
        # Check discrepancy analysis
        if not discrepancy_data.empty:
            disc_row = discrepancy_data[discrepancy_data['SampleID'] == sample_id]
            if not disc_row.empty:
                bias_likelihood = disc_row.iloc[0]['Bias_Likelihood']
                if bias_likelihood in ['High', 'Moderate']:
                    warnings.append(f"Extraction bias: {bias_likelihood}")
                    if bias_likelihood == 'High':
                        overall_risk = "High"
                    elif overall_risk == "Low":
                        overall_risk = "Moderate"
        
        # Check F/B ratio analysis
        if not ratio_bias_data.empty:
            ratio_row = ratio_bias_data[ratio_bias_data['SampleID'] == sample_id]
            if not ratio_row.empty:
                bias_level = ratio_row.iloc[0]['Bias_Level']
                if bias_level in ['High', 'Severe']:
                    bias_type = ratio_row.iloc[0]['Bias_Type']
                    warnings.append(f"F/B ratio: {bias_type}")
                    if bias_level == 'Severe':
                        overall_risk = "High"
                    elif overall_risk == "Low":
                        overall_risk = "Moderate"
        
        # Check missing taxa analysis
        if not missing_taxa_data.empty:
            taxa_row = missing_taxa_data[missing_taxa_data['SampleID'] == sample_id]
            if not taxa_row.empty:
                severity = taxa_row.iloc[0]['Severity']
                if severity in ['High', 'Severe']:
                    issue = taxa_row.iloc[0]['Issue_Description']
                    warnings.append(f"Missing taxa: {issue}")
                    if severity == 'Severe':
                        overall_risk = "High"
                    elif overall_risk == "Low":
                        overall_risk = "Moderate"
        
        warning_result = {
            'SampleID': sample_id,
            'Overall_Risk': overall_risk,
            'N_Warnings': len(warnings),
            'Warning_Summary': "; ".join(warnings) if warnings else "No issues detected",
            'Requires_Validation': overall_risk in ['High', 'Moderate']
        }
        warning_results.append(warning_result)
    
    return pd.DataFrame(warning_results)


def _get_clinical_flag(fb_ratio: float, bias_level: str) -> str:
    """Get clinical interpretation flag for F/B ratio."""
    if bias_level in ['High', 'Severe']:
        return "Review required"
    elif bias_level == 'Moderate':
        return "Monitor"
    else:
        return "Normal"


def _generate_bias_recommendations(extraction_flags: List[str]) -> List[str]:
    """Generate specific recommendations based on extraction quality flags."""
    recommendations = []
    
    if any("Critical" in flag for flag in extraction_flags):
        recommendations.extend([
            "CRITICAL: Re-extract samples using enhanced Gram-positive lysis protocol",
            "Use lysozyme treatment (>30 min) + mechanical disruption (bead-beating)",
            "Validate with qPCR targeting Firmicutes and Actinomycetota"
        ])
    elif any("Severe" in flag for flag in extraction_flags):
        recommendations.extend([
            "Re-extract subset of samples with optimized protocol",
            "Include mock communities with known Gram-positive bacteria",
            "Compare results with original extraction protocol"
        ])
    elif any("High" in flag or "Moderate" in flag for flag in extraction_flags):
        recommendations.extend([
            "Validate key findings with qPCR",
            "Check extraction protocol for Gram-positive optimization",
            "Consider biological vs technical variation"
        ])
    
    if extraction_flags:
        recommendations.append("Document extraction protocol details for publication")
    
    return recommendations


def assess_all_extraction_bias(sample_discrepancy: Optional[pd.DataFrame] = None,
                              fb_ratios: Optional[pd.DataFrame] = None,
                              extraction_ratios: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Perform comprehensive extraction bias assessment.
    
    Parameters
    ----------
    sample_discrepancy : pd.DataFrame, optional
        Feature-abundance discrepancy data
    fb_ratios : pd.DataFrame, optional
        F/B ratio data
    extraction_ratios : pd.DataFrame, optional
        Extraction efficiency ratio data
        
    Returns
    -------
    dict
        Complete extraction bias assessment
    """
    results = {
        'fb_ratio_bias': pd.DataFrame(),
        'missing_taxa_assessment': pd.DataFrame(),
        'systematic_bias_assessment': {},
        'sample_warnings': pd.DataFrame()
    }
    
    # F/B ratio bias assessment
    if fb_ratios is not None and not fb_ratios.empty:
        results['fb_ratio_bias'] = assess_fb_ratio_bias(fb_ratios)
    
    # Missing taxa assessment
    if extraction_ratios is not None and not extraction_ratios.empty:
        results['missing_taxa_assessment'] = detect_missing_key_taxa(extraction_ratios)
    
    # Systematic bias assessment
    if (sample_discrepancy is not None or 
        not results['fb_ratio_bias'].empty or 
        not results['missing_taxa_assessment'].empty):
        
        results['systematic_bias_assessment'] = assess_systematic_bias(
            sample_discrepancy if sample_discrepancy is not None else pd.DataFrame(),
            results['fb_ratio_bias'],
            results['missing_taxa_assessment']
        )
    
    # Generate sample warnings
    results['sample_warnings'] = generate_sample_bias_warnings(
        sample_discrepancy if sample_discrepancy is not None else pd.DataFrame(),
        results['fb_ratio_bias'],
        results['missing_taxa_assessment']
    )
    
    return results