"""
Automated EDA (Exploratory Data Analysis) utilities.
Generates comprehensive data profiles and insights automatically.
Inspired by AI Data Science Team patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AutoEDA:
    """Automated exploratory data analysis."""
    
    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        self.df = df.copy()
        self.target_col = target_col
        self.report = {}
    
    def analyze(self) -> Dict[str, Any]:
        """Run full automated EDA."""
        print("🔍 Running Automated EDA...")
        
        self.report['basic_info'] = self._basic_info()
        self.report['missing_values'] = self._missing_analysis()
        self.report['data_types'] = self._dtype_analysis()
        self.report['numerical_stats'] = self._numerical_analysis()
        self.report['categorical_stats'] = self._categorical_analysis()
        self.report['correlations'] = self._correlation_analysis()
        self.report['outliers'] = self._outlier_detection()
        
        if self.target_col:
            self.report['target_analysis'] = self._target_analysis()
        
        self.report['recommendations'] = self._generate_recommendations()
        
        print("✅ EDA Complete!")
        return self.report
    
    def _basic_info(self) -> Dict:
        """Basic dataset information."""
        return {
            'shape': self.df.shape,
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'memory_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'duplicates': self.df.duplicated().sum(),
            'duplicate_pct': self.df.duplicated().mean() * 100
        }
    
    def _missing_analysis(self) -> Dict:
        """Analyze missing values."""
        missing = self.df.isnull().sum()
        missing_pct = self.df.isnull().mean() * 100
        
        missing_df = pd.DataFrame({
            'count': missing,
            'percent': missing_pct
        }).sort_values('percent', ascending=False)
        
        return {
            'total_missing_cells': missing.sum(),
            'columns_with_missing': (missing > 0).sum(),
            'details': missing_df[missing_df['count'] > 0].to_dict(),
            'high_missing_cols': missing_df[missing_df['percent'] > 50].index.tolist()
        }
    
    def _dtype_analysis(self) -> Dict:
        """Analyze data types."""
        dtypes = self.df.dtypes.astype(str).value_counts().to_dict()
        
        return {
            'dtype_counts': dtypes,
            'numerical_cols': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_cols': self.df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_cols': self.df.select_dtypes(include=['datetime64']).columns.tolist(),
            'boolean_cols': self.df.select_dtypes(include=['bool']).columns.tolist()
        }
    
    def _numerical_analysis(self) -> Dict:
        """Analyze numerical columns."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        
        stats = {}
        for col in num_cols:
            stats[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'median': self.df[col].median(),
                'skew': self.df[col].skew(),
                'kurtosis': self.df[col].kurtosis(),
                'zeros': (self.df[col] == 0).sum(),
                'negatives': (self.df[col] < 0).sum(),
                'unique': self.df[col].nunique()
            }
        
        return stats
    
    def _categorical_analysis(self) -> Dict:
        """Analyze categorical columns."""
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        stats = {}
        for col in cat_cols:
            value_counts = self.df[col].value_counts()
            stats[col] = {
                'unique': self.df[col].nunique(),
                'top_value': value_counts.index[0] if len(value_counts) > 0 else None,
                'top_freq': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'top_5': value_counts.head(5).to_dict(),
                'is_likely_id': self.df[col].nunique() == len(self.df)
            }
        
        return stats
    
    def _correlation_analysis(self) -> Dict:
        """Analyze correlations."""
        num_df = self.df.select_dtypes(include=[np.number])
        
        if num_df.empty:
            return {'message': 'No numerical columns for correlation'}
        
        corr_matrix = num_df.corr()
        
        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        # Target correlations
        target_corr = {}
        if self.target_col and self.target_col in num_df.columns:
            target_corr = corr_matrix[self.target_col].drop(self.target_col).sort_values(
                key=abs, ascending=False
            ).head(10).to_dict()
        
        return {
            'high_correlations': high_corr,
            'target_correlations': target_corr
        }
    
    def _outlier_detection(self) -> Dict:
        """Detect outliers using IQR method."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        
        outliers = {}
        for col in num_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            n_outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            
            if n_outliers > 0:
                outliers[col] = {
                    'count': n_outliers,
                    'percent': n_outliers / len(self.df) * 100,
                    'lower_bound': lower,
                    'upper_bound': upper
                }
        
        return outliers
    
    def _target_analysis(self) -> Dict:
        """Analyze target variable."""
        if self.target_col not in self.df.columns:
            return {'error': 'Target column not found'}
        
        target = self.df[self.target_col]
        
        analysis = {
            'dtype': str(target.dtype),
            'unique': target.nunique(),
            'missing': target.isnull().sum()
        }
        
        if target.dtype in [np.number, 'int64', 'float64']:
            analysis['is_classification'] = target.nunique() < 20
            analysis['mean'] = target.mean()
            analysis['std'] = target.std()
            analysis['distribution'] = target.value_counts().head(10).to_dict()
        else:
            analysis['is_classification'] = True
            analysis['class_distribution'] = target.value_counts().to_dict()
            analysis['class_balance'] = target.value_counts(normalize=True).to_dict()
        
        return analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        # Missing values
        if self.report['missing_values']['total_missing_cells'] > 0:
            high_missing = self.report['missing_values']['high_missing_cols']
            if high_missing:
                recs.append(f"⚠️ Consider dropping columns with >50% missing: {high_missing}")
            recs.append("💡 Impute missing values (median for numerical, mode for categorical)")
        
        # Duplicates
        if self.report['basic_info']['duplicates'] > 0:
            recs.append(f"🔄 Found {self.report['basic_info']['duplicates']} duplicate rows - consider removing")
        
        # High cardinality categoricals
        for col, stats in self.report.get('categorical_stats', {}).items():
            if stats.get('is_likely_id'):
                recs.append(f"🆔 Column '{col}' looks like an ID - consider excluding from features")
            elif stats.get('unique', 0) > 50:
                recs.append(f"📊 High cardinality in '{col}' ({stats['unique']} values) - consider encoding carefully")
        
        # High correlations
        high_corr = self.report.get('correlations', {}).get('high_correlations', [])
        if high_corr:
            recs.append(f"🔗 Found {len(high_corr)} highly correlated feature pairs - consider removing redundant features")
        
        # Outliers
        outlier_cols = list(self.report.get('outliers', {}).keys())
        if outlier_cols:
            recs.append(f"📈 Outliers detected in {len(outlier_cols)} columns - consider clipping or transformation")
        
        # Skewed features
        for col, stats in self.report.get('numerical_stats', {}).items():
            if abs(stats.get('skew', 0)) > 2:
                recs.append(f"↗️ Column '{col}' is highly skewed - consider log transformation")
        
        return recs
    
    def print_summary(self) -> None:
        """Print EDA summary."""
        print("\n" + "="*60)
        print("📊 AUTOMATED EDA SUMMARY")
        print("="*60)
        
        # Basic info
        info = self.report['basic_info']
        print(f"\n📁 Dataset: {info['rows']:,} rows × {info['columns']} columns")
        print(f"💾 Memory: {info['memory_mb']:.2f} MB")
        print(f"🔄 Duplicates: {info['duplicates']:,} ({info['duplicate_pct']:.1f}%)")
        
        # Data types
        dtypes = self.report['data_types']
        print(f"\n📋 Column Types:")
        print(f"   Numerical: {len(dtypes['numerical_cols'])}")
        print(f"   Categorical: {len(dtypes['categorical_cols'])}")
        print(f"   Datetime: {len(dtypes['datetime_cols'])}")
        
        # Missing
        missing = self.report['missing_values']
        print(f"\n❓ Missing Values:")
        print(f"   Total: {missing['total_missing_cells']:,} cells")
        print(f"   Columns affected: {missing['columns_with_missing']}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        for rec in self.report['recommendations']:
            print(f"   {rec}")
        
        print("\n" + "="*60)
    
    def to_html(self, output_path: str) -> str:
        """Export EDA report to HTML."""
        html = f"""
        <html>
        <head>
            <title>EDA Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
                h2 {{ color: #666; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .recommendation {{ background: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Automated EDA Report</h1>
                
                <h2>Basic Information</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Rows</td><td>{self.report['basic_info']['rows']:,}</td></tr>
                    <tr><td>Columns</td><td>{self.report['basic_info']['columns']}</td></tr>
                    <tr><td>Memory (MB)</td><td>{self.report['basic_info']['memory_mb']:.2f}</td></tr>
                    <tr><td>Duplicates</td><td>{self.report['basic_info']['duplicates']:,}</td></tr>
                </table>
                
                <h2>Recommendations</h2>
                {''.join(f'<div class="recommendation">{rec}</div>' for rec in self.report['recommendations'])}
            </div>
        </body>
        </html>
        """
        
        Path(output_path).write_text(html)
        print(f"📄 Report saved to {output_path}")
        return output_path


def quick_eda(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict:
    """Quick automated EDA."""
    eda = AutoEDA(df, target_col)
    report = eda.analyze()
    eda.print_summary()
    return report
