"""
Statistical utilities for microbiome analysis comparisons.

This module provides statistical testing functions for comparing
groups and assessing significance of differences in microbiome data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from scipy import stats
import warnings


def perform_group_comparisons(data_df: pd.DataFrame, 
                            group_column: str,
                            numeric_columns: List[str],
                            sample_id_column: str = 'SampleID') -> pd.DataFrame:
    """
    Perform statistical comparisons between groups for numeric variables.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing data to compare
    group_column : str
        Column name indicating group membership
    numeric_columns : list of str
        Column names containing numeric data to compare
    sample_id_column : str, optional
        Name of column containing sample IDs
        
    Returns
    -------
    pd.DataFrame
        Statistical comparison results
    """
    if group_column not in data_df.columns:
        return pd.DataFrame()
    
    # Get unique groups
    groups = data_df[group_column].unique()
    groups = [g for g in groups if pd.notna(g)]
    
    if len(groups) < 2:
        return pd.DataFrame()
    
    comparison_results = []
    
    for variable in numeric_columns:
        if variable not in data_df.columns:
            continue
        
        # Prepare data for comparisons
        group_data = {}
        for group in groups:
            group_subset = data_df[data_df[group_column] == group][variable]
            # Remove infinite values and NaN
            finite_data = group_subset[pd.isfinite(group_subset)].dropna()
            if len(finite_data) >= 3:  # Minimum sample size for meaningful statistics
                group_data[group] = finite_data.values
        
        if len(group_data) < 2:
            continue
        
        # Perform pairwise comparisons
        group_names = list(group_data.keys())
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                group1_name = group_names[i]
                group2_name = group_names[j]
                group1_data = group_data[group1_name]
                group2_data = group_data[group2_name]
                
                # Perform statistical tests
                stats_result = _compare_two_groups(group1_data, group2_data)
                
                result = {
                    'Variable': variable,
                    'Group1': group1_name,
                    'Group2': group2_name,
                    'Group1_N': len(group1_data),
                    'Group2_N': len(group2_data),
                    'Group1_Mean': round(np.mean(group1_data), 4),
                    'Group2_Mean': round(np.mean(group2_data), 4),
                    'Group1_Median': round(np.median(group1_data), 4),
                    'Group2_Median': round(np.median(group2_data), 4),
                    'Mean_Difference': round(np.mean(group1_data) - np.mean(group2_data), 4),
                    'Test_Used': stats_result['test_used'],
                    'Test_Statistic': stats_result['test_statistic'],
                    'P_Value': stats_result['p_value'],
                    'Significance': _get_significance_label(stats_result['p_value']),
                    'Effect_Size': stats_result['effect_size'],
                    'Interpretation': stats_result['interpretation']
                }
                comparison_results.append(result)
    
    return pd.DataFrame(comparison_results)


def _compare_two_groups(group1_data: np.ndarray, group2_data: np.ndarray) -> Dict[str, Any]:
    """
    Compare two groups using appropriate statistical test.
    
    Parameters
    ----------
    group1_data : np.ndarray
        Data for group 1
    group2_data : np.ndarray
        Data for group 2
        
    Returns
    -------
    dict
        Statistical comparison results
    """
    n1, n2 = len(group1_data), len(group2_data)
    
    # Check for normality (if sample size allows)
    if n1 >= 8 and n2 >= 8:
        # Shapiro-Wilk test for normality
        try:
            _, p1 = stats.shapiro(group1_data)
            _, p2 = stats.shapiro(group2_data)
            normal1 = p1 > 0.05
            normal2 = p2 > 0.05
        except:
            normal1 = normal2 = False
    else:
        normal1 = normal2 = False
    
    # Check for equal variances
    if normal1 and normal2:
        try:
            _, p_var = stats.levene(group1_data, group2_data)
            equal_var = p_var > 0.05
        except:
            equal_var = False
    else:
        equal_var = False
    
    # Choose appropriate test
    if normal1 and normal2 and equal_var:
        # Independent t-test
        try:
            t_stat, p_val = stats.ttest_ind(group1_data, group2_data)
            test_used = "Independent t-test"
            test_statistic = round(t_stat, 4)
        except:
            # Fallback to Mann-Whitney U
            u_stat, p_val = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            test_used = "Mann-Whitney U (t-test failed)"
            test_statistic = round(u_stat, 4)
    elif normal1 and normal2:
        # Welch's t-test (unequal variances)
        try:
            t_stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=False)
            test_used = "Welch's t-test"
            test_statistic = round(t_stat, 4)
        except:
            # Fallback to Mann-Whitney U
            u_stat, p_val = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            test_used = "Mann-Whitney U (Welch's t-test failed)"
            test_statistic = round(u_stat, 4)
    else:
        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, p_val = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            test_used = "Mann-Whitney U"
            test_statistic = round(u_stat, 4)
        except:
            # Fallback values
            test_used = "No test performed"
            test_statistic = None
            p_val = None
    
    # Calculate effect size
    effect_size = _calculate_effect_size(group1_data, group2_data, test_used)
    
    # Generate interpretation
    interpretation = _interpret_comparison(
        np.mean(group1_data), np.mean(group2_data), 
        p_val, effect_size, test_used
    )
    
    return {
        'test_used': test_used,
        'test_statistic': test_statistic,
        'p_value': round(p_val, 6) if p_val is not None else None,
        'effect_size': effect_size,
        'interpretation': interpretation
    }


def _calculate_effect_size(group1_data: np.ndarray, group2_data: np.ndarray, test_used: str) -> Optional[float]:
    """Calculate effect size (Cohen's d for parametric tests, rank-biserial correlation for Mann-Whitney)."""
    try:
        if 't-test' in test_used:
            # Cohen's d
            pooled_std = np.sqrt(((len(group1_data) - 1) * np.var(group1_data, ddof=1) + 
                                (len(group2_data) - 1) * np.var(group2_data, ddof=1)) / 
                               (len(group1_data) + len(group2_data) - 2))
            if pooled_std > 0:
                cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std
                return round(cohens_d, 4)
        elif 'Mann-Whitney' in test_used:
            # Rank-biserial correlation approximation
            n1, n2 = len(group1_data), len(group2_data)
            u_stat, _ = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            r = 1 - (2 * u_stat) / (n1 * n2)
            return round(r, 4)
    except:
        pass
    
    return None


def _get_significance_label(p_value: Optional[float]) -> str:
    """Get significance label based on p-value."""
    if p_value is None:
        return "Not tested"
    elif p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.1:
        return "."
    else:
        return "ns"


def _interpret_comparison(mean1: float, mean2: float, p_value: Optional[float], 
                         effect_size: Optional[float], test_used: str) -> str:
    """Generate interpretation of statistical comparison."""
    if p_value is None:
        return "Statistical test could not be performed"
    
    # Direction of difference
    if mean1 > mean2:
        direction = "higher"
    elif mean1 < mean2:
        direction = "lower"
    else:
        direction = "similar"
    
    # Significance
    if p_value < 0.05:
        significance = "significantly"
    else:
        significance = "not significantly"
    
    # Effect size interpretation
    effect_magnitude = ""
    if effect_size is not None:
        abs_effect = abs(effect_size)
        if 't-test' in test_used:
            # Cohen's d interpretation
            if abs_effect < 0.2:
                effect_magnitude = " (negligible effect)"
            elif abs_effect < 0.5:
                effect_magnitude = " (small effect)"
            elif abs_effect < 0.8:
                effect_magnitude = " (medium effect)"
            else:
                effect_magnitude = " (large effect)"
        elif 'Mann-Whitney' in test_used:
            # Rank-biserial correlation interpretation
            if abs_effect < 0.1:
                effect_magnitude = " (negligible effect)"
            elif abs_effect < 0.3:
                effect_magnitude = " (small effect)"
            elif abs_effect < 0.5:
                effect_magnitude = " (medium effect)"
            else:
                effect_magnitude = " (large effect)"
    
    return f"Group 1 is {significance} {direction} than Group 2{effect_magnitude}"


def assess_dataset_heterogeneity(data_df: pd.DataFrame, 
                                numeric_columns: List[str],
                                group_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Assess heterogeneity and distribution characteristics of the dataset.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing data to assess
    numeric_columns : list of str
        Column names containing numeric data
    group_column : str, optional
        Column name for grouping (if available)
        
    Returns
    -------
    dict
        Dataset heterogeneity assessment
    """
    assessment = {
        'variable_summaries': {},
        'distribution_characteristics': {},
        'outlier_analysis': {},
        'group_heterogeneity': {}
    }
    
    for variable in numeric_columns:
        if variable not in data_df.columns:
            continue
        
        # Get finite data
        finite_data = data_df[variable][pd.isfinite(data_df[variable])].dropna()
        
        if len(finite_data) == 0:
            continue
        
        # Basic statistics
        assessment['variable_summaries'][variable] = {
            'n_samples': len(finite_data),
            'mean': round(finite_data.mean(), 4),
            'median': round(finite_data.median(), 4),
            'std': round(finite_data.std(), 4),
            'min': round(finite_data.min(), 4),
            'max': round(finite_data.max(), 4),
            'q25': round(finite_data.quantile(0.25), 4),
            'q75': round(finite_data.quantile(0.75), 4),
            'iqr': round(finite_data.quantile(0.75) - finite_data.quantile(0.25), 4),
            'coefficient_of_variation': round(finite_data.std() / finite_data.mean(), 4) if finite_data.mean() != 0 else None
        }
        
        # Distribution characteristics
        try:
            # Test for normality
            if len(finite_data) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(finite_data)
                is_normal = shapiro_p > 0.05
            else:
                is_normal = None
                shapiro_p = None
            
            # Skewness and kurtosis
            skewness = stats.skew(finite_data)
            kurtosis = stats.kurtosis(finite_data)
            
            assessment['distribution_characteristics'][variable] = {
                'is_normal': is_normal,
                'shapiro_p_value': round(shapiro_p, 6) if shapiro_p is not None else None,
                'skewness': round(skewness, 4),
                'kurtosis': round(kurtosis, 4),
                'distribution_type': _classify_distribution(is_normal, skewness, kurtosis)
            }
        except:
            assessment['distribution_characteristics'][variable] = {
                'is_normal': None,
                'shapiro_p_value': None,
                'skewness': None,
                'kurtosis': None,
                'distribution_type': "Could not assess"
            }
        
        # Outlier analysis
        q1 = finite_data.quantile(0.25)
        q3 = finite_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = finite_data[(finite_data < lower_bound) | (finite_data > upper_bound)]
        
        assessment['outlier_analysis'][variable] = {
            'n_outliers': len(outliers),
            'outlier_percentage': round(len(outliers) / len(finite_data) * 100, 2),
            'lower_bound': round(lower_bound, 4),
            'upper_bound': round(upper_bound, 4),
            'extreme_outliers': len(outliers[outliers > upper_bound + 1.5 * iqr]) + len(outliers[outliers < lower_bound - 1.5 * iqr])
        }
    
    # Group heterogeneity analysis
    if group_column and group_column in data_df.columns:
        groups = data_df[group_column].unique()
        groups = [g for g in groups if pd.notna(g)]
        
        for variable in numeric_columns:
            if variable not in data_df.columns:
                continue
                
            group_stats = []
            for group in groups:
                group_data = data_df[data_df[group_column] == group][variable]
                finite_group_data = group_data[pd.isfinite(group_data)].dropna()
                
                if len(finite_group_data) >= 3:
                    group_stats.append({
                        'group': group,
                        'n': len(finite_group_data),
                        'mean': finite_group_data.mean(),
                        'std': finite_group_data.std(),
                        'cv': finite_group_data.std() / finite_group_data.mean() if finite_group_data.mean() != 0 else None
                    })
            
            if len(group_stats) >= 2:
                # Calculate between-group vs within-group variation
                means = [g['mean'] for g in group_stats if g['mean'] is not None]
                stds = [g['std'] for g in group_stats if g['std'] is not None]
                
                between_group_variation = np.std(means) if len(means) > 1 else 0
                within_group_variation = np.mean(stds) if stds else 0
                
                assessment['group_heterogeneity'][variable] = {
                    'n_groups': len(group_stats),
                    'between_group_std': round(between_group_variation, 4),
                    'mean_within_group_std': round(within_group_variation, 4),
                    'heterogeneity_ratio': round(between_group_variation / within_group_variation, 4) if within_group_variation > 0 else None
                }
    
    return assessment


def _classify_distribution(is_normal: Optional[bool], skewness: float, kurtosis: float) -> str:
    """Classify the distribution type based on statistical properties."""
    if is_normal is None:
        return "Unknown"
    
    if is_normal:
        return "Normal"
    
    # Non-normal distributions
    if abs(skewness) < 0.5:
        skew_desc = "symmetric"
    elif skewness > 0.5:
        skew_desc = "right-skewed"
    else:
        skew_desc = "left-skewed"
    
    if abs(kurtosis) < 0.5:
        kurt_desc = "mesokurtic"
    elif kurtosis > 0.5:
        kurt_desc = "leptokurtic (peaked)"
    else:
        kurt_desc = "platykurtic (flat)"
    
    return f"Non-normal ({skew_desc}, {kurt_desc})"


def calculate_statistical_power(group_sizes: List[int], effect_size: float, alpha: float = 0.05) -> Dict[str, float]:
    """
    Estimate statistical power for group comparisons.
    
    Parameters
    ----------
    group_sizes : list of int
        Sample sizes for each group
    effect_size : float
        Expected effect size (Cohen's d)
    alpha : float, optional
        Significance level
        
    Returns
    -------
    dict
        Power analysis results
    """
    try:
        from scipy.stats import norm
        
        if len(group_sizes) != 2:
            return {'power': None, 'interpretation': "Power analysis only available for two-group comparisons"}
        
        n1, n2 = group_sizes
        
        # Calculate pooled standard error
        pooled_se = np.sqrt((1/n1) + (1/n2))
        
        # Calculate non-centrality parameter
        ncp = effect_size / pooled_se
        
        # Calculate power
        critical_value = norm.ppf(1 - alpha/2)  # Two-tailed test
        power = 1 - norm.cdf(critical_value - ncp) + norm.cdf(-critical_value - ncp)
        
        # Interpretation
        if power >= 0.8:
            interpretation = "Adequate power (≥80%)"
        elif power >= 0.5:
            interpretation = "Moderate power (50-80%)"
        else:
            interpretation = "Low power (<50%)"
        
        return {
            'power': round(power, 3),
            'interpretation': interpretation,
            'effect_size_used': effect_size,
            'alpha': alpha
        }
    
    except ImportError:
        return {'power': None, 'interpretation': "Power analysis requires additional dependencies"}
    except:
        return {'power': None, 'interpretation': "Could not calculate power"}