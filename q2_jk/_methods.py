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
import re


# Define standard taxonomic level order
TAXONOMIC_ORDER = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']


def normalize_taxonomy_assignment(assignment):
    """Normalize a taxonomy assignment for comparison.
    
    This function handles:
    - Empty strings, NA, None values
    - Rank prefixes like 'g__', 's__', etc.
    - Incertae Sedis variations
    - Underscore/space normalization
    
    Parameters
    ----------
    assignment : str
        Raw taxonomy assignment
        
    Returns
    -------
    str
        Normalized assignment or empty string if unclassified
    """
    if pd.isna(assignment) or assignment is None:
        return ""
    
    assignment = str(assignment).strip()
    
    # Handle empty strings
    if not assignment:
        return ""
    
    # Handle rank prefixes only (e.g., "g__", "s__")
    if re.match(r'^[a-z]__$', assignment):
        return ""
    
    # Remove rank prefixes for comparison
    if re.match(r'^[a-z]__', assignment):
        assignment = assignment[3:]
    
    # Handle Incertae Sedis variations (case insensitive)
    incertae_pattern = r'^incertae[\s_]*sedis$'
    if re.match(incertae_pattern, assignment, re.IGNORECASE):
        return ""
    
    # Normalize underscores and spaces
    assignment = re.sub(r'[_\s]+', ' ', assignment).strip()
    
    # If after all normalization we have an empty string, return empty
    if not assignment:
        return ""
    
    return assignment.lower()  # Convert to lowercase for comparison


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


# Gram staining classification database
# Include both modern and legacy taxonomic names for compatibility
GRAM_POSITIVE_PHYLA = {
    # Modern names (GTDB)
    'bacillota': 'positive',           # Modern name for Firmicutes
    'actinomycetota': 'positive',      # Modern name for Actinobacteria
    # Legacy names (Silva/Greengenes)
    'firmicutes': 'positive',
    'actinobacteria': 'positive',
    'tenericutes': 'variable',         # Mycoplasma - no cell wall
    'mycoplasmatota': 'variable',      # Modern name for Tenericutes
}

GRAM_NEGATIVE_PHYLA = {
    # Modern names (GTDB)
    'pseudomonadota': 'negative',      # Modern name for Proteobacteria
    'bacteroidota': 'negative',        # Modern name for Bacteroidetes
    'campylobacterota': 'negative',    # Modern name for Epsilonproteobacteria
    'desulfobacterota': 'negative',    # Modern name for some Deltaproteobacteria
    'cyanobacteriota': 'negative',     # Modern name for Cyanobacteria
    'chloroflexota': 'negative',       # Modern name for Chloroflexi
    'spirochaetota': 'negative',       # Modern name for Spirochaetes
    'planctomycetota': 'negative',     # Modern name for Planctomycetes
    'verrucomicrobiota': 'negative',   # Modern name for Verrucomicrobia
    'acidobacteriota': 'negative',     # Modern name for Acidobacteria
    'nitrospirota': 'negative',        # Modern name for Nitrospirae
    'aquificota': 'negative',          # Modern name for Aquificae
    'thermotogota': 'negative',        # Modern name for Thermotogae
    # Legacy names (Silva/Greengenes)
    'proteobacteria': 'negative',
    'bacteroidetes': 'negative',
    'cyanobacteria': 'negative',
    'chloroflexi': 'negative',
    'spirochaetes': 'negative',
    'planctomycetes': 'negative',
    'verrucomicrobia': 'negative',
    'acidobacteria': 'negative',
    'nitrospirae': 'negative',
    'deferribacteres': 'negative',
    'aquificae': 'negative',
    'thermotogae': 'negative',
}

# Genus-level exceptions and specific classifications
GRAM_STATUS_GENUS = {
    # Firmicutes exceptions
    'mycoplasma': 'variable',
    'ureaplasma': 'variable',
    'spiroplasma': 'variable',
    
    # Actinobacteria - mostly positive
    'mycobacterium': 'variable',  # Acid-fast, weakly gram-positive
    'nocardia': 'variable',       # Acid-fast, weakly gram-positive
    'streptomyces': 'positive',
    'bifidobacterium': 'positive',
    'propionibacterium': 'positive',
    'corynebacterium': 'positive',
    
    # Well-known gram-positive genera
    'staphylococcus': 'positive',
    'streptococcus': 'positive',
    'enterococcus': 'positive',
    'lactobacillus': 'positive',
    'bacillus': 'positive',
    'clostridium': 'positive',
    'listeria': 'positive',
    
    # Well-known gram-negative genera
    'escherichia': 'negative',
    'salmonella': 'negative',
    'shigella': 'negative',
    'klebsiella': 'negative',
    'pseudomonas': 'negative',
    'acinetobacter': 'negative',
    'neisseria': 'negative',
    'haemophilus': 'negative',
    'campylobacter': 'negative',
    'helicobacter': 'negative',
    'vibrio': 'negative',
    'legionella': 'negative',
    'bacteroides': 'negative',
    'prevotella': 'negative',
}

# Family-level classifications for broader coverage
GRAM_STATUS_FAMILY = {
    # Gram-positive families
    'enterococcaceae': 'positive',
    'lactobacillaceae': 'positive',
    'leuconostocaceae': 'positive',
    'streptococcaceae': 'positive',
    'staphylococcaceae': 'positive',
    'bacillaceae': 'positive',
    'clostridiaceae': 'positive',
    'lachnospiraceae': 'positive',
    'ruminococcaceae': 'positive',
    'eubacteriaceae': 'positive',
    'peptostreptococcaceae': 'positive',
    'erysipelotrichaceae': 'positive',
    'listeriaceae': 'positive',
    'mycobacteriaceae': 'variable',
    'propionibacteriaceae': 'positive',
    'corynebacteriaceae': 'positive',
    'micrococcaceae': 'positive',
    'bifidobacteriaceae': 'positive',
    
    # Gram-negative families
    'enterobacteriaceae': 'negative',
    'pseudomonadaceae': 'negative',
    'moraxellaceae': 'negative',
    'neisseriaceae': 'negative',
    'pasteurellaceae': 'negative',
    'vibrionaceae': 'negative',
    'aeromonadaceae': 'negative',
    'campylobacteraceae': 'negative',
    'helicobacteraceae': 'negative',
    'bacteroidaceae': 'negative',
    'prevotellaceae': 'negative',
    'porphyromonadaceae': 'negative',
    'rikenellaceae': 'negative',
    'tannerellaceae': 'negative',
    'flavobacteriaceae': 'negative',
    'sphingobacteriaceae': 'negative',
    'cytophagaceae': 'negative',
}


def create_gram_staining_database():
    """Create a hierarchical gram staining classification database.
    
    Returns
    -------
    dict
        Dictionary with taxonomic levels as keys and classification dicts as values
    """
    return {
        'phylum': {**GRAM_POSITIVE_PHYLA, **GRAM_NEGATIVE_PHYLA},
        'family': GRAM_STATUS_FAMILY,
        'genus': GRAM_STATUS_GENUS
    }


def classify_gram_status(taxonomy_levels, gram_db=None):
    """Classify gram staining status using hierarchical taxonomic lookup with confidence scoring.
    
    Parameters
    ----------
    taxonomy_levels : dict
        Dictionary with taxonomic level names as keys and normalized assignments as values
    gram_db : dict, optional
        Gram staining database. If None, uses default database.
    
    Returns
    -------
    tuple
        (gram_status, classification_level, confidence_score) where:
        - gram_status: one of 'Gram Positive', 'Gram Negative', 'Variable/Unknown', 'Non-Bacterial', 'Unclassified'
        - classification_level: the taxonomic level used for classification
        - confidence_score: float between 0-1, higher = more confident
    """
    if gram_db is None:
        gram_db = create_gram_staining_database()
    
    # Define confidence scores based on taxonomic level specificity
    confidence_scores = {
        'genus': 0.9,    # High confidence - most specific
        'family': 0.7,   # Good confidence - family-level patterns
        'phylum': 0.5,   # Moderate confidence - broad patterns
        'kingdom': 0.2   # Low confidence - very broad
    }
    
    # Check if it's bacterial at kingdom level
    kingdom = taxonomy_levels.get('Kingdom', '').lower()
    if kingdom and 'bacteria' not in normalize_taxonomy_assignment(kingdom):
        if any(term in kingdom for term in ['archaea', 'eukaryot', 'virus']):
            return 'Non-Bacterial', 'Kingdom', confidence_scores['kingdom']
    
    # Hierarchical search: genus → family → phylum
    search_order = [
        ('genus', 'Genus'),
        ('family', 'Family'), 
        ('phylum', 'Phylum')
    ]
    
    for db_level, tax_level in search_order:
        if tax_level in taxonomy_levels:
            assignment = normalize_taxonomy_assignment(taxonomy_levels[tax_level])
            if assignment and db_level in gram_db:
                # Check for direct match or partial match
                for db_taxon, gram_status in gram_db[db_level].items():
                    if assignment == db_taxon or db_taxon in assignment:
                        status_map = {
                            'positive': 'Gram Positive',
                            'negative': 'Gram Negative', 
                            'variable': 'Variable/Unknown'
                        }
                        mapped_status = status_map.get(gram_status, 'Variable/Unknown')
                        confidence = confidence_scores[db_level]
                        
                        # Reduce confidence for variable classifications
                        if gram_status == 'variable':
                            confidence *= 0.8
                            
                        return mapped_status, tax_level, confidence
    
    # If we have bacterial kingdom but no specific classification
    if kingdom and 'bacteria' in kingdom:
        return 'Variable/Unknown', 'Kingdom', confidence_scores['kingdom']
    
    return 'Unclassified', 'None', 0.0


def analyze_gram_staining(taxonomy_df, feature_table=None, col_name='Taxon'):
    """Analyze gram staining composition of taxonomy data with optional abundance weighting.
    
    Parameters
    ----------
    taxonomy_df : pd.DataFrame
        DataFrame containing taxonomy data
    feature_table : pd.DataFrame, optional
        Feature abundance table. If provided, analysis will be abundance-weighted.
    col_name : str, optional
        Column containing taxonomy strings
    
    Returns
    -------
    tuple
        (summary_stats, detailed_classification, level_breakdown, unique_taxa_summary, 
         confidence_stats, abundance_weighted_stats)
        - summary_stats: Overall gram status counts and percentages
        - detailed_classification: Per-feature classification details
        - level_breakdown: Gram status distribution at each taxonomic level
        - unique_taxa_summary: Summary of unique taxa and their classifications
        - confidence_stats: Distribution of classification confidence scores
        - abundance_weighted_stats: Sample abundance-weighted statistics (if feature_table provided)
    """
    tax_levels = get_taxonomic_levels(taxonomy_df, col_name)
    gram_db = create_gram_staining_database()
    
    # Classify each feature
    classifications = []
    for idx in tax_levels.index:
        # Create dict of normalized taxonomy levels for this feature
        feature_taxonomy = {}
        for level in tax_levels.columns:
            raw_assignment = tax_levels.loc[idx, level]
            feature_taxonomy[level] = normalize_taxonomy_assignment(raw_assignment)
        
        gram_status, classification_level, confidence_score = classify_gram_status(feature_taxonomy, gram_db)
        
        # Find the deepest classified level for reporting
        deepest_level = 'Unclassified'
        deepest_taxon = ''
        for level in ['Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom']:
            if level in feature_taxonomy and feature_taxonomy[level]:
                deepest_level = level
                deepest_taxon = tax_levels.loc[idx, level]  # Use original assignment for display
                break
        
        classifications.append({
            'FeatureID': idx,
            'FullTaxonomy': taxonomy_df.loc[idx, col_name],
            'GramStatus': gram_status,
            'ClassificationLevel': classification_level,
            'ConfidenceScore': confidence_score,
            'DeepestClassifiedLevel': deepest_level,
            'DeepestTaxon': deepest_taxon
        })
    
    # Create detailed classification DataFrame
    detailed_df = pd.DataFrame(classifications)
    
    # Generate summary statistics
    gram_counts = detailed_df['GramStatus'].value_counts()
    total_features = len(detailed_df)
    
    summary_stats = pd.DataFrame({
        'Count': gram_counts,
        'Percentage': (gram_counts / total_features * 100).round(2)
    })
    summary_stats = summary_stats.sort_values('Count', ascending=False)
    
    # Generate level breakdown - show gram distribution using two different approaches
    level_breakdown = {
        'classification_source': {},  # How many features were classified BY each level
        'taxonomic_resolution': {}    # Gram distribution for features WITH taxonomy at each level
    }
    
    # Classification Source: Show how many features were classified BY each taxonomic level
    classification_source_stats = detailed_df['ClassificationLevel'].value_counts()
    for level in classification_source_stats.index:
        if level != 'None':  # Skip unclassified
            level_features = detailed_df[detailed_df['ClassificationLevel'] == level]
            if len(level_features) > 0:
                level_gram_counts = level_features['GramStatus'].value_counts()
                level_total = len(level_features)
                
                level_breakdown['classification_source'][level] = pd.DataFrame({
                    'Count': level_gram_counts,
                    'Percentage': (level_gram_counts / level_total * 100).round(2),
                    'Description': f'Features classified using {level.lower()}-level database matches'
                })
    
    # Taxonomic Resolution: Show gram distribution for features with taxonomy at each level
    for level in ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']:
        if level in tax_levels.columns:
            # Get features that have actual taxonomy assignment at this level
            level_features = detailed_df[detailed_df['DeepestClassifiedLevel'] == level]
            
            if len(level_features) > 0:
                level_gram_counts = level_features['GramStatus'].value_counts()
                level_total = len(level_features)
                
                level_breakdown['taxonomic_resolution'][level] = pd.DataFrame({
                    'Count': level_gram_counts,
                    'Percentage': (level_gram_counts / level_total * 100).round(2),
                    'Description': f'Features with taxonomic classification stopping at {level.lower()} level'
                })
    
    # Generate unique taxa summary
    unique_taxa_summary = {}
    for level in ['Genus', 'Family', 'Phylum']:
        if level in tax_levels.columns:
            level_taxa = detailed_df[detailed_df['ClassificationLevel'] == level].copy()
            if len(level_taxa) > 0:
                # Group by taxonomic assignment and gram status
                taxa_groups = level_taxa.groupby(['DeepestTaxon', 'GramStatus']).agg({
                    'FeatureID': 'count',
                    'ConfidenceScore': 'mean'
                }).rename(columns={'FeatureID': 'FeatureCount'}).reset_index()
                
                unique_taxa_summary[level] = taxa_groups
    
    # Generate confidence statistics
    confidence_stats = pd.DataFrame({
        'ClassificationLevel': detailed_df['ClassificationLevel'],
        'ConfidenceScore': detailed_df['ConfidenceScore'],
        'GramStatus': detailed_df['GramStatus']
    })
    
    confidence_summary = confidence_stats.groupby(['ClassificationLevel', 'GramStatus']).agg({
        'ConfidenceScore': ['mean', 'std', 'count']
    }).round(3)
    confidence_summary.columns = ['MeanConfidence', 'StdConfidence', 'Count']
    confidence_summary = confidence_summary.reset_index()
    
    # Generate database coverage statistics
    gram_db = create_gram_staining_database()
    database_coverage = {}
    
    for level in ['Genus', 'Family', 'Phylum']:
        if level.lower() in gram_db:
            level_col = level
            if level_col in tax_levels.columns:
                # Get unique normalized taxa at this level
                normalized_taxa = tax_levels[level_col].apply(normalize_taxonomy_assignment)
                unique_taxa = set(normalized_taxa[normalized_taxa != ""])
                
                # Check how many are in our database
                db_taxa = set(gram_db[level.lower()].keys())
                covered_taxa = unique_taxa.intersection(db_taxa)
                
                database_coverage[level] = {
                    'unique_taxa_in_data': len(unique_taxa),
                    'taxa_in_database': len(covered_taxa),
                    'coverage_percentage': (len(covered_taxa) / len(unique_taxa) * 100) if len(unique_taxa) > 0 else 0,
                    'database_size': len(db_taxa)
                }
    
    # Generate classification pipeline statistics
    pipeline_stats = []
    total_features = len(detailed_df)
    
    # Step 1: Domain classification
    bacterial_features = detailed_df[detailed_df['GramStatus'] != 'Non-Bacterial']
    non_bacterial = len(detailed_df) - len(bacterial_features)
    pipeline_stats.append({
        'Step': '1. Domain Classification',
        'Description': 'Identify bacterial vs non-bacterial features',
        'Features_Processed': total_features,
        'Features_Passed': len(bacterial_features),
        'Features_Filtered': non_bacterial,
        'Filter_Reason': 'Non-bacterial (Archaea, Eukaryota, etc.)'
    })
    
    # Step 2: Taxonomic classification
    classified_features = bacterial_features[bacterial_features['GramStatus'] != 'Unclassified']
    unclassified = len(bacterial_features) - len(classified_features)
    pipeline_stats.append({
        'Step': '2. Taxonomic Classification',  
        'Description': 'Features with sufficient taxonomic information',
        'Features_Processed': len(bacterial_features),
        'Features_Passed': len(classified_features),
        'Features_Filtered': unclassified,
        'Filter_Reason': 'Insufficient taxonomic classification'
    })
    
    # Step 3: Database matching
    definitive_features = classified_features[classified_features['GramStatus'].isin(['Gram Positive', 'Gram Negative'])]
    variable_unknown = len(classified_features) - len(definitive_features)
    pipeline_stats.append({
        'Step': '3. Database Matching',
        'Description': 'Features with definitive gram staining predictions',
        'Features_Processed': len(classified_features),
        'Features_Passed': len(definitive_features),
        'Features_Filtered': variable_unknown,
        'Filter_Reason': 'No database match or variable gram status'
    })
    
    pipeline_stats_df = pd.DataFrame(pipeline_stats)
    
    # Generate abundance-weighted statistics if feature table is provided
    abundance_weighted_stats = None
    if feature_table is not None:
        # Ensure feature IDs match between taxonomy and feature table
        common_features = list(set(detailed_df['FeatureID']).intersection(set(feature_table.index)))
        
        if common_features:
            abundance_data = []
            
            for sample_id in feature_table.columns:
                sample_totals = {}
                sample_total_reads = feature_table[sample_id].sum()
                
                for gram_status in detailed_df['GramStatus'].unique():
                    # Get features with this gram status
                    gram_features = detailed_df[detailed_df['GramStatus'] == gram_status]['FeatureID'].tolist()
                    # Get abundance for these features in this sample
                    gram_abundance = feature_table.loc[
                        [f for f in gram_features if f in common_features], sample_id
                    ].sum()
                    
                    sample_totals[gram_status] = {
                        'AbsoluteAbundance': gram_abundance,
                        'RelativeAbundance': (gram_abundance / sample_total_reads * 100) if sample_total_reads > 0 else 0
                    }
                
                sample_totals['SampleID'] = sample_id
                sample_totals['TotalReads'] = sample_total_reads
                abundance_data.append(sample_totals)
            
            # Convert to DataFrame format
            abundance_rows = []
            for sample_data in abundance_data:
                sample_id = sample_data['SampleID']
                total_reads = sample_data['TotalReads']
                
                for gram_status in detailed_df['GramStatus'].unique():
                    if gram_status in sample_data:
                        abundance_rows.append({
                            'SampleID': sample_id,
                            'GramStatus': gram_status,
                            'AbsoluteAbundance': sample_data[gram_status]['AbsoluteAbundance'],
                            'RelativeAbundance': sample_data[gram_status]['RelativeAbundance'],
                            'TotalReads': total_reads
                        })
            
            abundance_weighted_stats = pd.DataFrame(abundance_rows)
    
    return (summary_stats, detailed_df, level_breakdown, unique_taxa_summary, 
            confidence_summary, abundance_weighted_stats, database_coverage, pipeline_stats_df)


def analyze_comprehensive_taxonomic_levels(taxonomy_df: pd.DataFrame, feature_table: pd.DataFrame = None, col_name='Taxon'):
    """
    Analyze gram staining distribution comprehensively at each taxonomic level.
    
    For each taxonomic level, classify ALL features based on available taxonomy
    at that specific level, providing consistent data representation.
    
    Parameters
    ----------
    taxonomy_df : pd.DataFrame
        Taxonomy DataFrame with feature data
    feature_table : pd.DataFrame, optional
        Feature abundance table for abundance-weighted analysis
    col_name : str, optional
        Column containing taxonomy strings
        
    Returns
    -------
    dict
        Dictionary with taxonomic levels as keys and DataFrames with gram distributions
    """
    tax_levels = get_taxonomic_levels(taxonomy_df, col_name)
    gram_db = create_gram_staining_database()
    
    comprehensive_analysis = {}
    
    for level in ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']:
        if level not in tax_levels.columns:
            continue
            
        level_results = []
        
        for feature_id, row in tax_levels.iterrows():
            # For comprehensive analysis: classify using taxonomy available UP TO and INCLUDING current level
            # But if current level is empty/missing, the feature is "unclassified at this resolution"
            
            # First check if we have classification at the current level
            current_level_value = row[level] if level in tax_levels.columns else ""
            current_level_normalized = normalize_taxonomy_assignment(current_level_value)
            
            # Build taxonomy dictionary up to current level
            taxonomy_levels = {}
            for i, tax_level in enumerate(['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']):
                if tax_level in tax_levels.columns and i <= ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'].index(level):
                    taxonomy_levels[tax_level] = row[tax_level]
                else:
                    break
            
            # Special handling for Kingdom level - only do domain classification, no gram staining
            if level == 'Kingdom':
                kingdom_normalized = current_level_normalized
                if kingdom_normalized == "":
                    gram_status = 'Unclassified'
                    classification_level = 'None'
                    confidence = 0.0
                elif 'bacteria' in kingdom_normalized.lower():
                    gram_status = 'Bacterial'  # Domain classification only
                    classification_level = 'Kingdom' 
                    confidence = 1.0  # 100% confidence for domain classification
                elif kingdom_normalized.lower() in ['archaea', 'eukaryota', 'eukarya']:
                    gram_status = 'Non-Bacterial'
                    classification_level = 'Kingdom'
                    confidence = 1.0  # 100% confidence for domain classification
                else:
                    gram_status = 'Unclassified'  # Unknown domain
                    classification_level = 'None'
                    confidence = 0.0
            
            # For all other levels, determine gram staining classification
            elif current_level_normalized == "":
                # Feature doesn't have classification at this taxonomic resolution
                # Check if it's at least bacterial
                kingdom_value = taxonomy_levels.get('Kingdom', '')
                kingdom_normalized = normalize_taxonomy_assignment(kingdom_value)
                
                if kingdom_normalized == "":
                    gram_status = 'Unclassified'
                elif 'bacteria' in kingdom_normalized.lower():
                    gram_status = 'Unclassified'  # Bacterial but unclassified at this level
                elif kingdom_normalized.lower() in ['archaea', 'eukaryota', 'eukarya']:
                    gram_status = 'Non-Bacterial'  # Non-bacterial
                else:
                    gram_status = 'Unclassified'  # Unknown domain
                
                classification_level = 'None'
                confidence = 0.0
            else:
                # Feature has classification at this level - use normal classification logic
                gram_status, classification_level, confidence = classify_gram_status(taxonomy_levels)
            
            level_results.append({
                'FeatureID': feature_id,
                'GramStatus': gram_status,
                'ClassificationLevel': classification_level,
                'ConfidenceScore': confidence,
                'AvailableTaxonomy': level
            })
        
        results_df = pd.DataFrame(level_results)
        
        # Calculate distribution (abundance-weighted if feature_table provided)
        if feature_table is not None:
            # Calculate abundance-weighted distribution for this level
            abundance_by_gram = {}
            common_features = set(results_df['FeatureID']).intersection(set(feature_table.index))
            
            for gram_status in results_df['GramStatus'].unique():
                # Get features with this gram status
                gram_features = results_df[results_df['GramStatus'] == gram_status]['FeatureID'].tolist()
                # Filter to common features and calculate total abundance
                gram_features_common = [f for f in gram_features if f in common_features]
                if gram_features_common:
                    total_abundance = feature_table.loc[gram_features_common].sum().sum()
                else:
                    total_abundance = 0
                abundance_by_gram[gram_status] = total_abundance
            
            # Convert to Series for consistency
            gram_counts = pd.Series(abundance_by_gram)
            total_abundance = gram_counts.sum()
            
            distribution = pd.DataFrame({
                'Count': gram_counts,  # Actually abundance counts now
                'Percentage': (gram_counts / total_abundance * 100).round(2) if total_abundance > 0 else 0
            })
        else:
            # Use feature counts (original behavior)
            gram_counts = results_df['GramStatus'].value_counts()
            total_features = len(results_df)
            
            distribution = pd.DataFrame({
                'Count': gram_counts,
                'Percentage': (gram_counts / total_features * 100).round(2)
            })
        
        # Ensure all categories are represented (different for Kingdom level)
        if level == 'Kingdom':
            # Kingdom level uses domain classification categories
            for category in ['Bacterial', 'Non-Bacterial', 'Unclassified']:
                if category not in distribution.index:
                    distribution.loc[category] = [0, 0.0]
            distribution = distribution.reindex(['Bacterial', 'Non-Bacterial', 'Unclassified'])
        else:
            # All other levels use gram staining categories
            for category in ['Gram Positive', 'Gram Negative', 'Variable/Unknown', 'Non-Bacterial', 'Unclassified']:
                if category not in distribution.index:
                    distribution.loc[category] = [0, 0.0]
            distribution = distribution.reindex(['Gram Positive', 'Gram Negative', 'Variable/Unknown', 'Non-Bacterial', 'Unclassified'])
        distribution['TaxonomicLevel'] = level
        
        comprehensive_analysis[level] = distribution
    
    return comprehensive_analysis


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
        # Normalize assignments for comparison
        tax1_normalized = tax1_levels[level].apply(normalize_taxonomy_assignment)
        tax2_normalized = tax2_levels[level].apply(normalize_taxonomy_assignment)
        
        # Compare normalized assignments at this level
        same_mask = tax1_normalized == tax2_normalized
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
                'Tax1_Normalized': tax1_normalized.loc[diff_mask],
                'Tax2_Normalized': tax2_normalized.loc[diff_mask],
                'Full_Tax1': tax1_df.loc[diff_mask, 'Taxon'],
                'Full_Tax2': tax2_df.loc[diff_mask, 'Taxon']
            })
            
            # Create confusion matrix for this level using original (non-normalized) values for display
            # but only include truly different assignments
            unique_vals1 = set(tax1_levels.loc[diff_mask, level])
            unique_vals2 = set(tax2_levels.loc[diff_mask, level])
            all_vals = sorted(list(unique_vals1.union(unique_vals2)))
            
            # Create confusion matrix
            conf_matrix = pd.DataFrame(0, index=all_vals, columns=all_vals)
            for idx in diff_mask[diff_mask].index:
                val1 = tax1_levels.loc[idx, level]
                val2 = tax2_levels.loc[idx, level]
                if val1 in all_vals and val2 in all_vals:
                    conf_matrix.loc[val1, val2] += 1
            
            # Add diagonal elements for matching assignments
            for idx in same_mask[same_mask].index:
                val1 = tax1_levels.loc[idx, level]
                if val1 not in all_vals:
                    all_vals.append(val1)
                    # Expand confusion matrix
                    conf_matrix = conf_matrix.reindex(index=all_vals, columns=all_vals, fill_value=0)
                conf_matrix.loc[val1, val1] += 1
                
            confusion_matrices[level] = conf_matrix
            
            # Find most common mismatches (differences between taxonomies)
            # Group by normalized assignment pairs to get counts and examples
            mismatch_groups = {}
            for idx in diff_mask[diff_mask].index:
                original1 = tax1_levels.loc[idx, level]
                original2 = tax2_levels.loc[idx, level]
                normalized1 = tax1_normalized.loc[idx]
                normalized2 = tax2_normalized.loc[idx]
                
                # Use original values for display
                key = (original1, original2)
                if key not in mismatch_groups:
                    mismatch_groups[key] = {
                        'count': 0,
                        'examples': [],
                        'normalized1': normalized1,
                        'normalized2': normalized2
                    }
                mismatch_groups[key]['count'] += 1
                # Store up to 3 examples per mismatch type
                if len(mismatch_groups[key]['examples']) < 3:
                    mismatch_groups[key]['examples'].append(idx)
            
            # Convert to list and sort by count
            mismatches = []
            for (orig1, orig2), data in mismatch_groups.items():
                mismatches.append((
                    orig1, orig2, data['count'], 
                    (data['count'] / n_total) * 100,
                    '; '.join(data['examples'])  # Join example feature IDs
                ))
            
            # Sort by count (descending)
            mismatches.sort(key=lambda x: x[2], reverse=True)
            
            # Create a DataFrame for most common differences
            if mismatches:
                top_n = min(20, len(mismatches))  # Show top 20 or fewer
                common_diff_df = pd.DataFrame(
                    mismatches[:top_n], 
                    columns=[f'{tax1_name} Assignment', f'{tax2_name} Assignment', 
                            'Count', 'Percent of Features', 'Example Feature IDs']
                )
                common_diff_df['Percent of Features'] = common_diff_df['Percent of Features'].round(2)
                common_differences[level] = common_diff_df
    
    return summary_stats, confusion_matrices, common_differences
