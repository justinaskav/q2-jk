"""
Metadata parsing and group analysis functions.

This module handles metadata file processing, sample validation,
and group-based comparisons for microbiome analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import warnings


def parse_metadata_file(metadata) -> pd.DataFrame:
    """
    Parse and validate metadata from QIIME2 Metadata object.
    
    Parameters
    ----------
    metadata : qiime2.Metadata
        QIIME2 metadata object
        
    Returns
    -------
    pd.DataFrame
        Validated metadata DataFrame
    """
    # Convert QIIME2 Metadata to DataFrame
    metadata_df = metadata.to_dataframe()
    
    # Ensure index is string type for sample IDs
    metadata_df.index = metadata_df.index.astype(str)
    
    # Remove any columns with all NaN values
    metadata_df = metadata_df.dropna(axis=1, how='all')
    
    return metadata_df


def validate_sample_overlap(metadata: pd.DataFrame, analysis_samples: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Validate overlap between metadata samples and analysis samples.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata DataFrame
    analysis_samples : list of str
        Sample IDs from analysis data
        
    Returns
    -------
    tuple
        (common_samples, metadata_only, analysis_only)
    """
    metadata_samples = set(metadata.index.astype(str))
    analysis_samples_set = set(analysis_samples)
    
    common_samples = list(metadata_samples.intersection(analysis_samples_set))
    metadata_only = list(metadata_samples - analysis_samples_set)
    analysis_only = list(analysis_samples_set - metadata_samples)
    
    if len(common_samples) == 0:
        warnings.warn("No common samples found between metadata and analysis data")
    
    if len(metadata_only) > 0:
        warnings.warn(f"{len(metadata_only)} samples in metadata not found in analysis data")
    
    if len(analysis_only) > 0:
        warnings.warn(f"{len(analysis_only)} samples in analysis data not found in metadata")
    
    return common_samples, metadata_only, analysis_only


def extract_group_variables(metadata: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Extract categorical variables suitable for grouping analysis.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata DataFrame
    exclude_columns : list of str, optional
        Column names to exclude from grouping
        
    Returns
    -------
    dict
        Dictionary of group variables with their properties
    """
    if exclude_columns is None:
        exclude_columns = []
    
    group_variables = {}
    
    for column in metadata.columns:
        if column in exclude_columns:
            continue
            
        # Get non-null values for this column
        non_null_values = metadata[column].dropna()
        
        if len(non_null_values) == 0:
            continue
        
        # Check if column is suitable for grouping
        unique_values = non_null_values.unique()
        n_unique = len(unique_values)
        
        # Skip columns with too many unique values (likely continuous or ID columns)
        if n_unique > 20:
            continue
        
        # Skip columns where every value is unique (likely ID columns)
        if n_unique == len(non_null_values):
            continue
        
        # Determine data type
        if pd.api.types.is_numeric_dtype(non_null_values):
            # Check if numeric data has few unique values (could be categorical)
            if n_unique <= 10:
                var_type = "categorical_numeric"
            else:
                var_type = "continuous"
                continue  # Skip continuous variables for now
        else:
            var_type = "categorical_string"
        
        # Count samples per group
        value_counts = non_null_values.value_counts()
        
        group_variables[column] = {
            'type': var_type,
            'n_unique': n_unique,
            'unique_values': list(unique_values),
            'value_counts': value_counts.to_dict(),
            'n_samples': len(non_null_values),
            'n_missing': len(metadata) - len(non_null_values)
        }
    
    return group_variables


def create_sample_groups(metadata: pd.DataFrame, group_column: str, min_group_size: int = 3) -> Dict[str, List[str]]:
    """
    Create sample groups based on a metadata column.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata DataFrame
    group_column : str
        Column name to use for grouping
    min_group_size : int, optional
        Minimum number of samples required per group
        
    Returns
    -------
    dict
        Dictionary mapping group names to lists of sample IDs
    """
    if group_column not in metadata.columns:
        raise ValueError(f"Group column '{group_column}' not found in metadata")
    
    # Get non-null values for grouping
    non_null_data = metadata[group_column].dropna()
    
    # Create groups
    sample_groups = {}
    for group_value, group_samples in non_null_data.groupby(non_null_data):
        group_name = str(group_value)
        sample_list = group_samples.index.astype(str).tolist()
        
        # Only include groups with sufficient sample size
        if len(sample_list) >= min_group_size:
            sample_groups[group_name] = sample_list
    
    return sample_groups


def compare_groups_data(data_df: pd.DataFrame, sample_groups: Dict[str, List[str]], 
                       numeric_columns: List[str], sample_id_column: str = 'SampleID') -> pd.DataFrame:
    """
    Compare numeric data between sample groups.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing data to compare (e.g., ratio data)
    sample_groups : dict
        Dictionary mapping group names to sample ID lists
    numeric_columns : list of str
        Column names containing numeric data to compare
    sample_id_column : str, optional
        Name of column containing sample IDs
        
    Returns
    -------
    pd.DataFrame
        Group comparison statistics
    """
    comparison_results = []
    
    for column in numeric_columns:
        if column not in data_df.columns:
            continue
            
        for group_name, sample_ids in sample_groups.items():
            # Get data for this group
            group_data = data_df[data_df[sample_id_column].isin(sample_ids)][column]
            
            # Skip infinite values for statistics
            finite_data = group_data[np.isfinite(group_data)]
            
            if len(finite_data) == 0:
                continue
            
            # Calculate statistics
            stats = {
                'Variable': column,
                'Group': group_name,
                'N_Samples': len(group_data),
                'N_Finite': len(finite_data),
                'Mean': round(finite_data.mean(), 4) if len(finite_data) > 0 else None,
                'Median': round(finite_data.median(), 4) if len(finite_data) > 0 else None,
                'Std': round(finite_data.std(), 4) if len(finite_data) > 1 else None,
                'Min': round(finite_data.min(), 4) if len(finite_data) > 0 else None,
                'Max': round(finite_data.max(), 4) if len(finite_data) > 0 else None,
                'Q25': round(finite_data.quantile(0.25), 4) if len(finite_data) > 0 else None,
                'Q75': round(finite_data.quantile(0.75), 4) if len(finite_data) > 0 else None
            }
            
            comparison_results.append(stats)
    
    return pd.DataFrame(comparison_results)


def create_metadata_summary(metadata: pd.DataFrame, sample_overlap_info: Tuple[List[str], List[str], List[str]]) -> Dict[str, Any]:
    """
    Create a summary of metadata characteristics.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata DataFrame
    sample_overlap_info : tuple
        Results from validate_sample_overlap()
        
    Returns
    -------
    dict
        Metadata summary information
    """
    common_samples, metadata_only, analysis_only = sample_overlap_info
    
    # Extract group variables
    group_vars = extract_group_variables(metadata)
    
    # Create summary
    summary = {
        'total_metadata_samples': len(metadata),
        'total_columns': len(metadata.columns),
        'common_samples': len(common_samples),
        'metadata_only_samples': len(metadata_only),
        'analysis_only_samples': len(analysis_only),
        'overlap_percentage': round(len(common_samples) / max(len(metadata), 1) * 100, 2),
        'group_variables': group_vars,
        'suitable_group_columns': list(group_vars.keys())
    }
    
    return summary


def analyze_metadata_groups(metadata: Optional[pd.DataFrame], ratio_data: Optional[pd.DataFrame], 
                           sample_data: Optional[pd.DataFrame] = None, 
                           group_column: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Perform comprehensive metadata-based group analysis.
    
    Parameters
    ----------
    metadata : pd.DataFrame, optional
        Metadata DataFrame
    ratio_data : pd.DataFrame, optional
        Ratio analysis results
    sample_data : pd.DataFrame, optional
        Sample-wise analysis data
    group_column : str, optional
        Specific column to use for grouping
        
    Returns
    -------
    dict or None
        Complete metadata analysis results
    """
    if metadata is None:
        return None
    
    # Parse and validate metadata
    metadata = parse_metadata_file(metadata)
    
    # Get sample IDs from available data
    analysis_samples = []
    if ratio_data is not None and 'SampleID' in ratio_data.columns:
        analysis_samples.extend(ratio_data['SampleID'].tolist())
    elif sample_data is not None and 'SampleID' in sample_data.columns:
        analysis_samples.extend(sample_data['SampleID'].tolist())
    
    if not analysis_samples:
        return {
            'metadata_summary': create_metadata_summary(metadata, ([], [], [])),
            'group_comparisons': None,
            'selected_group_column': None
        }
    
    # Validate sample overlap
    sample_overlap = validate_sample_overlap(metadata, analysis_samples)
    common_samples = sample_overlap[0]
    
    if len(common_samples) == 0:
        return {
            'metadata_summary': create_metadata_summary(metadata, sample_overlap),
            'group_comparisons': None,
            'selected_group_column': None
        }
    
    # Create metadata summary
    metadata_summary = create_metadata_summary(metadata, sample_overlap)
    
    # Determine group column to use
    if group_column is None and metadata_summary['suitable_group_columns']:
        # Use the first suitable column
        group_column = metadata_summary['suitable_group_columns'][0]
    
    group_comparisons = None
    if group_column and group_column in metadata.columns:
        # Create sample groups
        sample_groups = create_sample_groups(metadata, group_column)
        
        # Compare groups if we have data
        if sample_groups and ratio_data is not None:
            numeric_columns = ['FB_Ratio', 'BA_Ratio', 'GramPos_GramNeg_Ratio']
            group_comparisons = compare_groups_data(ratio_data, sample_groups, numeric_columns)
    
    return {
        'metadata_summary': metadata_summary,
        'group_comparisons': group_comparisons,
        'selected_group_column': group_column,
        'sample_overlap': sample_overlap
    }