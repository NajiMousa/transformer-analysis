#!/usr/bin/env python3
"""
Transformer Data Analysis Script
Comprehensive analysis of transformer data from 2018-2023
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_transformer_data():
    """Load and process transformer data from Excel files"""
    
    # You can replace this with actual file loading if you have the Excel files
    # Example: df = pd.read_excel('Transformer_data_merged.xlsx')
    
    # Sample data structure - replace with your actual data loading
    print("Loading transformer data...")
    print("Note: Replace this section with pd.read_excel() calls for your actual files")
    
    return None

def create_comprehensive_analysis():
    """Create comprehensive transformer analysis"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    
    # Sample data for demonstration - replace with actual data
    sample_data = {
        'year': [2018, 2019, 2020, 2021, 2022, 2023] * 50,
        'kva_rating': np.random.choice([160, 250, 400, 630, 800, 1000, 1600, 2000], 300),
        'manufacturing_year': np.random.choice(range(1975, 2024), 300),
        'manufacturer': np.random.choice(['ELCO', 'TRAFO', 'SEM', 'ABB', 'MEKSAN', 'ARDAN'], 300),
        'type': np.random.choice(['هوائي', 'أرضي', 'خارجي'], 300),
        'body_condition': np.random.choice(['جيد', 'جيد جدا', 'صدأ', 'سيلان زيت'], 300),
        'region': ['دير البلح'] * 300
    }
    
    df = pd.DataFrame(sample_data)
    df['age'] = datetime.now().year - df['manufacturing_year']
    
    # 1. Records by Year
    plt.subplot(4, 3, 1)
    year_counts = df['year'].value_counts().sort_index()
    bars = plt.bar(year_counts.index, year_counts.values, 
                   color='#667eea', alpha=0.8, edgecolor='white', linewidth=1)
    plt.title('📊 Records by Year', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Top Manufacturers (Pie Chart)
    plt.subplot(4, 3, 2)
    manufacturer_counts = df['manufacturer'].value_counts().head(8)
    colors = plt.cm.Set3(np.linspace(0, 1, len(manufacturer_counts)))
    wedges, texts, autotexts = plt.pie(manufacturer_counts.values, 
                                      labels=manufacturer_counts.index,
                                      autopct='%1.1f%%', startangle=90,
                                      colors=colors)
    plt.title('🏭 Top Manufacturers', fontsize=14, fontweight='bold', pad=20)
    
    # 3. KVA Rating Distribution
    plt.subplot(4, 3, 3)
    plt.hist(df['kva_rating'], bins=15, color='#f093fb', alpha=0.7, 
             edgecolor='white', linewidth=1)
    plt.title('⚡ KVA Rating Distribution', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('KVA Rating')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    
    # 4. Transformer Types
    plt.subplot(4, 3, 4)
    type_counts = df['type'].value_counts()
    bars = plt.bar(type_counts.index, type_counts.values, 
                   color='#4facfe', alpha=0.8, edgecolor='white', linewidth=1)
    plt.title('🏗️ Transformer Types', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # 5. Age vs Capacity Scatter Plot
    plt.subplot(4, 3, 5)
    scatter = plt.scatter(df['age'], df['kva_rating'], 
                         c=df['year'], cmap='viridis',
                         alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    plt.title('📈 Age vs Capacity Analysis', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Age (Years)')
    plt.ylabel('Capacity (KVA)')
    plt.colorbar(scatter, label='Data Year')
    plt.grid(True, alpha=0.3)
    
    # 6. Manufacturing Timeline
    plt.subplot(4, 3, 6)
    timeline = df.groupby('manufacturing_year').size().sort_index()
    plt.plot(timeline.index, timeline.values, marker='o', linewidth=3,
            markersize=6, color='#00f2fe', markerfacecolor='white',
            markeredgewidth=2, markeredgecolor='#00f2fe')
    plt.title('⏰ Manufacturing Timeline', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Manufacturing Year')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # 7. Condition Analysis (Horizontal Bar)
    plt.subplot(4, 3, 7)
    condition_counts = df['body_condition'].value_counts()
    bars = plt.barh(condition_counts.index, condition_counts.values,
                   color='#764ba2', alpha=0.8, edgecolor='white', linewidth=1)
    plt.title('🔧 Body Condition Analysis', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Count')
    plt.ylabel('Condition')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{int(width)}', ha='left', va='center', fontweight='bold')
    
    # 8. Age Distribution by Manufacturer
    plt.subplot(4, 3, 8)
    top_manufacturers = df['manufacturer'].value_counts().head(5).index
    df_top = df[df['manufacturer'].isin(top_manufacturers)]
    
    for i, manufacturer in enumerate(top_manufacturers):
        ages = df_top[df_top['manufacturer'] == manufacturer]['age']
        plt.hist(ages, bins=10, alpha=0.6, label=manufacturer, 
                edgecolor='white', linewidth=0.5)
    
    plt.title('📊 Age Distribution by Manufacturer', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Age (Years)')
    plt.ylabel('Frequency')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    # 9. Capacity Distribution by Type
    plt.subplot(4, 3, 9)
    df.boxplot(column='kva_rating', by='type', ax=plt.gca())
    plt.title('⚡ Capacity Distribution by Type', fontsize=14, fontweight='bold', pad=20)
    plt.suptitle('')  # Remove automatic title
    plt.xlabel('Transformer Type')
    plt.ylabel('KVA Rating')
    plt.xticks(rotation=45, ha='right')
    
    # 10. Yearly Capacity Growth
    plt.subplot(4, 3, 10)
    yearly_capacity = df.groupby('year')['kva_rating'].sum()
    bars = plt.bar(yearly_capacity.index, yearly_capacity.values/1000,
                  color='#667eea', alpha=0.8, edgecolor='white', linewidth=1)
    plt.title('📈 Total Capacity by Year', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Year')
    plt.ylabel('Total Capacity (MVA)')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 11. Manufacturer Market Share Over Time
    plt.subplot(4, 3, 11)
    pivot_data = df.pivot_table(values='kva_rating', index='year', 
                               columns='manufacturer', aggfunc='count', fill_value=0)
    pivot_data.plot(kind='area', stacked=True, ax=plt.gca(), alpha=0.7)
    plt.title('🏭 Manufacturer Market Share Over Time', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Year')
    plt.ylabel('Number of Transformers')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 12. Average Age by Manufacturer
    plt.subplot(4, 3, 12)
    avg_age = df.groupby('manufacturer')['age'].mean().sort_values(ascending=True)
    bars = plt.bar(range(len(avg_age)), avg_age.values,
                  color='#f5576c', alpha=0.8, edgecolor='white', linewidth=1)
    plt.title('⏰ Average Age by Manufacturer', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Manufacturer')
    plt.ylabel('Average Age (Years)')
    plt.xticks(range(len(avg_age)), avg_age.index, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig, df

def generate_summary_statistics(df):
    """Generate comprehensive summary statistics"""
    
    print("\n" + "="*60)
    print("           TRANSFORMER DATA ANALYSIS SUMMARY")
    print("="*60)
    
    # Basic Statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   • Total Transformers: {len(df):,}")
    print(f"   • Total Capacity: {df['kva_rating'].sum():,} KVA ({df['kva_rating'].sum()/1000:.1f} MVA)")
    print(f"   • Average Capacity: {df['kva_rating'].mean():.1f} KVA")
    print(f"   • Capacity Range: {df['kva_rating'].min()} - {df['kva_rating'].max()} KVA")
    print(f"   • Average Age: {df['age'].mean():.1f} years")
    print(f"   • Age Range: {df['age'].min()} - {df['age'].max()} years")
    
    # Manufacturer Analysis
    print(f"\n🏭 MANUFACTURER ANALYSIS:")
    manufacturer_stats = df.groupby('manufacturer').agg({
        'kva_rating': ['count', 'sum', 'mean'],
        'age': 'mean'
    }).round(2)
    
    manufacturer_stats.columns = ['Count', 'Total_KVA', 'Avg_KVA', 'Avg_Age']
    manufacturer_stats = manufacturer_stats.sort_values('Count', ascending=False)
    
    print(f"   • Number of Manufacturers: {df['manufacturer'].nunique()}")
    print(f"   • Top 5 Manufacturers by Count:")
    for idx, (mfr, row) in enumerate(manufacturer_stats.head().iterrows(), 1):
        print(f"     {idx}. {mfr}: {int(row['Count'])} transformers, "
              f"{row['Total_KVA']:.0f} KVA total, {row['Avg_Age']:.1f} years avg age")
    
    # Type Analysis
    print(f"\n🏗️ TYPE ANALYSIS:")
    type_stats = df.groupby('type').agg({
        'kva_rating': ['count', 'sum', 'mean']
    }).round(2)
    type_stats.columns = ['Count', 'Total_KVA', 'Avg_KVA']
    
    for type_name, row in type_stats.iterrows():
        percentage = (row['Count'] / len(df)) * 100
        print(f"   • {type_name}: {int(row['Count'])} units ({percentage:.1f}%), "
              f"{row['Total_KVA']:.0f} KVA total")
    
    # Yearly Analysis
    print(f"\n📅 YEARLY ANALYSIS:")
    yearly_stats = df.groupby('year').agg({
        'kva_rating': ['count', 'sum', 'mean'],
        'age': 'mean'
    }).round(2)
    yearly_stats.columns = ['Count', 'Total_KVA', 'Avg_KVA', 'Avg_Age']
    
    for year, row in yearly_stats.iterrows():
        print(f"   • {year}: {int(row['Count'])} records, "
              f"{row['Total_KVA']:.0f} KVA total, {row['Avg_Age']:.1f} years avg age")
    
    # Age Categories
    print(f"\n⏰ AGE CATEGORIES:")
    df['age_category'] = pd.cut(df['age'], bins=[0, 10, 25, 50, 100], 
                               labels=['New (0-10)', 'Medium (10-25)', 'Old (25-50)', 'Very Old (50+)'])
    age_dist = df['age_category'].value_counts()
    
    for category, count in age_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   • {category}: {count} transformers ({percentage:.1f}%)")
    
    # Condition Analysis
    print(f"\n🔧 CONDITION ANALYSIS:")
    condition_counts = df['body_condition'].value_counts()
    print(f"   • Condition Categories: {len(condition_counts)}")
    for condition, count in condition_counts.head().items():
        percentage = (count / len(df)) * 100
        print(f"     - {condition}: {count} units ({percentage:.1f}%)")
    
    # Capacity Categories
    print(f"\n⚡ CAPACITY CATEGORIES:")
    df['capacity_category'] = pd.cut(df['kva_rating'], 
                                   bins=[0, 250, 500, 1000, 5000], 
                                   labels=['Small (≤250)', 'Medium (250-500)', 
                                          'Large (500-1000)', 'Very Large (>1000)'])
    capacity_dist = df['capacity_category'].value_counts()
    
    for category, count in capacity_dist.items():
        percentage = (count / len(df)) * 100
        avg_kva = df[df['capacity_category'] == category]['kva_rating'].mean()
        print(f"   • {category}: {count} units ({percentage:.1f}%), avg: {avg_kva:.0f} KVA")
    
    print("\n" + "="*60)

def export_processed_data(df):
    """Export processed data to various formats"""
    
    # Export to CSV
    df.to_csv('transformer_analysis_results.csv', index=False)
    print("✅ Data exported to 'transformer_analysis_results.csv'")
    
    # Export summary statistics
    with open('transformer_summary_report.txt', 'w', encoding='utf-8') as f:
        f.write("TRANSFORMER DATA ANALYSIS SUMMARY REPORT\n")
        f.write("="*50 + "\n\n")
        
        # Basic stats
        f.write(f"Total Transformers: {len(df):,}\n")
        f.write(f"Total Capacity: {df['kva_rating'].sum():,} KVA\n")
        f.write(f"Average Age: {df['age'].mean():.1f} years\n")
        f.write(f"Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Top manufacturers
        f.write("TOP MANUFACTURERS:\n")
        for mfr, count in df['manufacturer'].value_counts().head().items():
            f.write(f"- {mfr}: {count} transformers\n")
    
    print("✅ Summary report exported to 'transformer_summary_report.txt'")

def create_dashboard_html():
    """Create an HTML dashboard similar to the web version"""
    
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Transformer Analysis Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { text-align: center; color: #333; }
            .stats { display: flex; justify-content: space-around; margin: 20px 0; }
            .stat-card { 
                background: #f0f8ff; 
                padding: 20px; 
                border-radius: 10px; 
                text-align: center; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .stat-number { font-size: 2em; color: #667eea; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔌 Transformer Data Analysis Dashboard</h1>
            <p>Generated by Python Analysis Script</p>
        </div>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{total_transformers}</div>
                <div>Total Transformers</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{total_capacity}</div>
                <div>Total Capacity (KVA)</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{avg_age}</div>
                <div>Average Age (Years)</div>
            </div>
        </div>
        <p>📊 Charts have been saved as PNG files in the current directory.</p>
        <p>📋 Detailed data available in 'transformer_analysis_results.csv'</p>
        <p>📄 Summary report available in 'transformer_summary_report.txt'</p>
    </body>
    </html>
    '''
    return html_template

def main():
    """Main execution function"""
    
    print("🔌 TRANSFORMER DATA ANALYSIS TOOL")
    print("=" * 50)
    
    # Create comprehensive analysis
    fig, df = create_comprehensive_analysis()
    
    # Generate summary statistics
    generate_summary_statistics(df)
    
    # Save the plot
    plt.savefig('transformer_analysis_dashboard.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Dashboard saved as 'transformer_analysis_dashboard.png'")
    
    # Export processed data
    export_processed_data(df)
    
    # Create HTML dashboard
    html_content = create_dashboard_html().format(
        total_transformers=f"{len(df):,}",
        total_capacity=f"{df['kva_rating'].sum():,}",
        avg_age=f"{df['age'].mean():.1f}"
    )
    
    with open('transformer_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ HTML dashboard saved as 'transformer_dashboard.html'")
    
    # Show the plot
    plt.show()
    
    print("\n🎉 Analysis Complete!")
    print("Files generated:")
    print("  • transformer_analysis_dashboard.png - Visual dashboard")
    print("  • transformer_analysis_results.csv - Processed data")
    print("  • transformer_summary_report.txt - Summary statistics")
    print("  • transformer_dashboard.html - HTML dashboard")

if __name__ == "__main__":
    main()