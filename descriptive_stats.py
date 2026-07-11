import pandas as pd
import numpy as np

# Read the original data
df = pd.read_csv('Generated_Molecules_Complete_20260622_105416.csv')

# Select only the columns needed for descriptive statistics
# Following Section 2.8.2: MW, LogP, TPSA, and GNN-predicted LogP
spss_data = df[['ID', 'MW', 'RDKit_LogP', 'GNN_LogP', 'TPSA', 'QED', 'SAscore', 'LogS']].copy()

# Rename columns to be SPSS-friendly (no spaces, max 8 characters)
spss_data.columns = ['ID', 'MW', 'LogP', 'GNN_LogP', 'TPSA', 'QED', 'SAscore', 'LogS']

# Add a grouping variable for Top 4 vs Full Set
# We'll use the composite score from the previous analysis
spss_data['Composite'] = (spss_data['QED'] * 0.4 +
                          (1 - spss_data['SAscore'] / 10) * 0.3 +
                          spss_data['LogS'] * 0.3)

# Rank and identify top 4
spss_data['Rank'] = spss_data['Composite'].rank(method='first', ascending=False)
spss_data['Top4'] = spss_data['Rank'].apply(lambda x: 'Top4' if x <= 4 else 'Other')

# Sort by ID for better presentation
spss_data = spss_data.sort_values('ID')

# Save as Excel for SPSS import
excel_file = 'SPSS_Descriptive_Statistics_Data.xlsx'
with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    # Main data sheet
    spss_data.to_excel(writer, sheet_name='NCE_Data', index=False)

    # Create a separate sheet with variable descriptions for SPSS
    var_desc = pd.DataFrame({
        'Variable': ['ID', 'MW', 'LogP', 'GNN_LogP', 'TPSA', 'QED', 'SAscore', 'LogS', 'Composite', 'Rank', 'Top4'],
        'Description': ['Compound Identifier', 'Molecular Weight (g/mol)', 'RDKit LogP (lipophilicity)',
                        'GNN-predicted LogP', 'Topological Polar Surface Area (Å²)',
                        'QED (drug-likeness score)', 'Synthetic Accessibility Score',
                        'Predicted LogS (solubility)', 'Composite Score', 'Ranking', 'Group (Top4/Other)'],
        'Type': ['String', 'Scale', 'Scale', 'Scale', 'Scale', 'Scale', 'Scale', 'Scale', 'Scale', 'Ordinal', 'Nominal']
    })
    var_desc.to_excel(writer, sheet_name='Variable_Info', index=False)

    # Create a data dictionary sheet
    data_dict = pd.DataFrame({
        'Property': ['MW', 'LogP', 'GNN_LogP', 'TPSA', 'QED', 'SAscore', 'LogS'],
        'Full Set Mean': [np.mean(spss_data['MW']), np.mean(spss_data['LogP']),
                          np.mean(spss_data['GNN_LogP']), np.mean(spss_data['TPSA']),
                          np.mean(spss_data['QED']), np.mean(spss_data['SAscore']),
                          np.mean(spss_data['LogS'])],
        'Full Set SD': [np.std(spss_data['MW'], ddof=1), np.std(spss_data['LogP'], ddof=1),
                        np.std(spss_data['GNN_LogP'], ddof=1), np.std(spss_data['TPSA'], ddof=1),
                        np.std(spss_data['QED'], ddof=1), np.std(spss_data['SAscore'], ddof=1),
                        np.std(spss_data['LogS'], ddof=1)]
    })
    data_dict.to_excel(writer, sheet_name='Quick_Stats', index=False)

print(f"Excel file created: {excel_file}")
print("\nData Preview (first 5 rows):")
print(spss_data.head())
print("\nVariables included:")
print(spss_data.columns.tolist())
print(f"\nTotal compounds: {len(spss_data)}")
print(f"Top 4 compounds: {spss_data[spss_data['Top4'] == 'Top4']['ID'].tolist()}")