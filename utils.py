import pandas as pd
import re
import jax
import jax.numpy as jnp
import blackjax

# Function to split values with parentheses
def parse_parentheses_uncertainty(value):
    if pd.isna(value):
        return pd.NA, pd.NA
    
    # Handle values without parentheses
    if isinstance(value, (int, float)) or '(' not in str(value):
        return value, pd.NA
    
    # Extract the value and uncertainty using regex
    match = re.match(r'([\d.]+)\s*\((\d+(?:\.\d+)?)\)', str(value))
    if match:
        return float(match.group(1)), float(match.group(2))
    return value, pd.NA

def parse_df_with_parentheses_uncertainty(df):
    # Create a new dataframe for the result
    result_df = pd.DataFrame()

    # Process each column
    for column in df.columns:
        # Check if column contains values with parentheses
        has_parentheses = any(
            isinstance(value, str) and '(' in value 
            for value in df[column].dropna().astype(str)
        )
        
        if has_parentheses:
            # Split the column into value and uncertainty
            values, uncertainties = zip(*df[column].apply(parse_parentheses_uncertainty))
            
            # Add the columns to the result dataframe
            result_df[column] = values
            result_df[f"parentheses_uncertainty_{column}"] = uncertainties
        else:
            # Keep the column as is
            result_df[column] = df[column]
    return result_df

def parentheses_uncertainty_to_std(df):
    # get pairs of value and uncertainty columns
    # where uncertainty columns are formmated as f"parentheses_uncertainty_{column}"
    value_uncertainty_pairs = [
        (column, f"parentheses_uncertainty_{column}")
        for column in df.columns
        if f"parentheses_uncertainty_{column}" in df.columns
    ]
    for value_col_name, uncertainty_col_name in value_uncertainty_pairs:
        values = pd.to_numeric(df[value_col_name], errors='coerce')
        uncertainties = df[uncertainty_col_name]
        
        # Convert uncertainties to string for valid numeric values
        str_uncertainties = uncertainties.astype(str)
        decimal_places = (str_uncertainties.str.extract(r'\.(\d+)$')[0]
            .str.len()                      # length = number of decimal places
            .fillna(0)                     # integer values get 0
            .astype(int)
        )
            
        df[f"std_{value_col_name}"] = values * (10.0 ** (-decimal_places))
    # drop the uncertainty columns
    df = df.drop(columns=[f"parentheses_uncertainty_{column}" for column in df.columns if f"parentheses_uncertainty_{column}" in df.columns])
    return df

