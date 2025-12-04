import pandas as pd
import numpy as np
from datetime import datetime
import re

class HealthcareDataPreprocessor:
    """
    Data cleaning and preprocessing for healthcare insurance claims
    """
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.df_cleaned = None
        
    def load_data(self):
        """Load CSV data"""
        print("Loading data...")
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        return self.df
    
    def clean_column_names(self):
        """Standardize column names"""
        # Strip whitespace
        self.df.columns = self.df.columns.str.strip()
        
        # Normalize common column name variations to standard names
        column_mapping = {
            'claimid': 'ClaimID',
            'claim_id': 'ClaimID',
            'claim id': 'ClaimID',
            'patientid': 'PatientID',
            'patient_id': 'PatientID',
            'patient id': 'PatientID',
            'providerid': 'ProviderID',
            'provider_id': 'ProviderID',
            'provider id': 'ProviderID',
            'claimamount': 'ClaimAmount',
            'claim_amount': 'ClaimAmount',
            'claim amount': 'ClaimAmount',
            'claimdate': 'ClaimDate',
            'claim_date': 'ClaimDate',
            'claim date': 'ClaimDate',
            'diagnosiscode': 'DiagnosisCode',
            'diagnosis_code': 'DiagnosisCode',
            'diagnosis code': 'DiagnosisCode',
            'procedurecode': 'ProcedureCode',
            'procedure_code': 'ProcedureCode',
            'procedure code': 'ProcedureCode',
            'patientage': 'PatientAge',
            'patient_age': 'PatientAge',
            'patient age': 'PatientAge',
            'patientgender': 'PatientGender',
            'patient_gender': 'PatientGender',
            'patient gender': 'PatientGender',
            'providerspecialty': 'ProviderSpecialty',
            'provider_specialty': 'ProviderSpecialty',
            'provider specialty': 'ProviderSpecialty',
            'claimstatus': 'ClaimStatus',
            'claim_status': 'ClaimStatus',
            'claim status': 'ClaimStatus',
            'claimtype': 'ClaimType',
            'claim_type': 'ClaimType',
            'claim type': 'ClaimType',
            'diagnosisdate': 'DiagnosisDate',
            'diagnosis_date': 'DiagnosisDate',
            'diagnosis date': 'DiagnosisDate'
        }
        
        # Apply mapping (case-insensitive)
        new_columns = []
        for col in self.df.columns:
            col_lower = col.lower()
            if col_lower in column_mapping:
                new_columns.append(column_mapping[col_lower])
            else:
                new_columns.append(col)
        
        self.df.columns = new_columns
        print("Column names cleaned and normalized")
        
    def handle_missing_values(self):
        """Handle missing values based on column type"""
        print("\nHandling missing values...")
        
        # Check missing values
        missing = self.df.isnull().sum()
        print(f"Missing values per column:\n{missing[missing > 0]}")
        
        # Fill numeric columns with median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                
    def remove_duplicates(self):
        """Remove duplicate claims"""
        print("\nRemoving duplicates...")
        initial_count = len(self.df)
        
        # Try to find ClaimID column (case-insensitive)
        claim_id_col = None
        for col in self.df.columns:
            if col.strip().lower() == 'claimid':
                claim_id_col = col
                break
        
        if claim_id_col:
            self.df.drop_duplicates(subset=[claim_id_col], inplace=True)
            removed = initial_count - len(self.df)
            print(f"Removed {removed} duplicate records")
        else:
            # If no ClaimID column, try to remove duplicates based on all columns
            # or use a combination of key columns if available
            key_cols = []
            for col_name in ['PatientID', 'ProviderID', 'ClaimDate', 'ClaimAmount']:
                for col in self.df.columns:
                    if col.strip().lower() == col_name.lower():
                        key_cols.append(col)
                        break
            
            if key_cols:
                self.df.drop_duplicates(subset=key_cols, inplace=True)
                removed = initial_count - len(self.df)
                print(f"Removed {removed} duplicate records (using key columns: {', '.join(key_cols)})")
            else:
                # Last resort: remove duplicates based on all columns
                self.df.drop_duplicates(inplace=True)
                removed = initial_count - len(self.df)
                print(f"Removed {removed} duplicate records (using all columns)")
        
    def validate_dates(self):
        """Validate and convert date columns"""
        print("\nValidating dates...")
        date_columns = ['ClaimDate', 'DiagnosisDate']
        
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                invalid_dates = self.df[col].isnull().sum()
                if invalid_dates > 0:
                    print(f"Warning: {invalid_dates} invalid dates in {col}")
                    
    def validate_amounts(self):
        """Validate claim amounts"""
        print("\nValidating claim amounts...")
        
        if 'ClaimAmount' in self.df.columns:
            # Remove negative amounts
            negative_mask = self.df['ClaimAmount'] < 0
            if negative_mask.sum() > 0:
                print(f"Warning: Removing {negative_mask.sum()} negative claim amounts")
                self.df = self.df[~negative_mask]
            
            # Flag outliers (beyond 3 standard deviations)
            mean = self.df['ClaimAmount'].mean()
            std = self.df['ClaimAmount'].std()
            outliers = ((self.df['ClaimAmount'] - mean).abs() > 3 * std).sum()
            print(f"Found {outliers} potential outlier amounts")
            
    def standardize_categorical(self):
        """Standardize categorical variables"""
        print("\nStandardizing categorical variables...")
        
        categorical_mappings = {
            'PatientGender': {'M': 'Male', 'F': 'Female'},
            'ClaimStatus': {
                'Approved': 'Approved',
                'Pending': 'Pending', 
                'Denied': 'Denied'
            },
            'ClaimType': {
                'Routine': 'Routine',
                'Emergency': 'Emergency',
                'Inpatient': 'Inpatient',
                'Outpatient': 'Outpatient'
            }
        }
        
        for col, mapping in categorical_mappings.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].map(mapping).fillna(self.df[col])
                
    def create_derived_features(self):
        """Create additional features for analysis"""
        print("\nCreating derived features...")
        
        # Age groups
        if 'PatientAge' in self.df.columns:
            self.df['AgeGroup'] = pd.cut(self.df['PatientAge'], 
                                         bins=[0, 18, 35, 50, 65, 100],
                                         labels=['Child', 'Young Adult', 'Adult', 'Senior', 'Elderly'])
        
        # Claim amount categories
        if 'ClaimAmount' in self.df.columns:
            self.df['ClaimCategory'] = pd.cut(self.df['ClaimAmount'],
                                              bins=[0, 1000, 5000, 10000, float('inf')],
                                              labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Time-based features
        if 'ClaimDate' in self.df.columns:
            self.df['ClaimMonth'] = self.df['ClaimDate'].dt.month
            self.df['ClaimYear'] = self.df['ClaimDate'].dt.year
            self.df['ClaimDayOfWeek'] = self.df['ClaimDate'].dt.dayofweek
            
    def detect_anomalies(self):
        """Detect potential fraudulent patterns"""
        print("\nDetecting anomalies...")
        
        # Flag suspicious patterns
        self.df['PotentialFraud'] = False
        
        # Multiple claims same day
        if 'PatientID' in self.df.columns and 'ClaimDate' in self.df.columns:
            same_day_claims = self.df.groupby(['PatientID', 'ClaimDate']).size()
            multiple_claims = same_day_claims[same_day_claims > 2].index
            mask = self.df.set_index(['PatientID', 'ClaimDate']).index.isin(multiple_claims)
            self.df.loc[mask, 'PotentialFraud'] = True
            
        # Unusually high amounts
        if 'ClaimAmount' in self.df.columns:
            q99 = self.df['ClaimAmount'].quantile(0.99)
            self.df.loc[self.df['ClaimAmount'] > q99, 'PotentialFraud'] = True
            
        fraud_count = self.df['PotentialFraud'].sum()
        print(f"Flagged {fraud_count} potentially fraudulent claims")
        
    def generate_summary_stats(self):
        """Generate summary statistics"""
        print("\n" + "="*60)
        print("DATA SUMMARY STATISTICS")
        print("="*60)
        
        print(f"\nTotal Records: {len(self.df)}")
        print(f"Date Range: {self.df['ClaimDate'].min()} to {self.df['ClaimDate'].max()}")
        
        if 'ClaimAmount' in self.df.columns:
            print(f"\nClaim Amount Statistics:")
            print(f"  Mean: ${self.df['ClaimAmount'].mean():.2f}")
            print(f"  Median: ${self.df['ClaimAmount'].median():.2f}")
            print(f"  Total: ${self.df['ClaimAmount'].sum():.2f}")
            
        if 'ClaimStatus' in self.df.columns:
            print(f"\nClaim Status Distribution:")
            print(self.df['ClaimStatus'].value_counts())
            
        if 'ProviderSpecialty' in self.df.columns:
            print(f"\nTop 5 Provider Specialties:")
            print(self.df['ProviderSpecialty'].value_counts().head())
            
    def preprocess(self):
        """Execute full preprocessing pipeline"""
        print("Starting preprocessing pipeline...\n")
        
        self.load_data()
        self.clean_column_names()
        self.remove_duplicates()
        self.handle_missing_values()
        self.validate_dates()
        self.validate_amounts()
        self.standardize_categorical()
        self.create_derived_features()
        self.detect_anomalies()
        self.generate_summary_stats()
        
        self.df_cleaned = self.df.copy()
        print("\n✓ Preprocessing completed successfully!")
        
        return self.df_cleaned
    
    def save_cleaned_data(self, output_path='../data/processed/cleaned_healthcare_data.csv'):
        """Save cleaned data"""
        if self.df_cleaned is not None:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.df_cleaned.to_csv(output_path, index=False)
            print(f"\n✓ Cleaned data saved to {output_path}")
        else:
            print("No cleaned data to save. Run preprocess() first.")


# Usage Example
if __name__ == "__main__":
    import os
    # Get the correct path relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'raw', 'healthcare_claims.csv')
    
    # Initialize preprocessor
    preprocessor = HealthcareDataPreprocessor(data_path)
    
    # Run preprocessing
    cleaned_data = preprocessor.preprocess()
    
    # Save cleaned data
    output_path = os.path.join(project_root, 'data', 'processed', 'cleaned_healthcare_data.csv')
    preprocessor.save_cleaned_data(output_path)
    
    print("\nPreprocessing complete! Ready for Knowledge Graph construction.")