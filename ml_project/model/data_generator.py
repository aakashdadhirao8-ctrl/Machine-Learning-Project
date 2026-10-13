import numpy as np
import pandas as pd

np.random.seed(42)

def generate_customer_data(n_samples=3000):
    """Generate synthetic multi-class customer dataset (fallback if real Kaggle CSV not found)"""
    age = np.random.normal(38, 12, n_samples).clip(18, 75).astype(int)
    income = np.random.normal(65000, 25000, n_samples).clip(20000, 150000)
    spending_score = np.random.normal(50, 20, n_samples).clip(1, 100)
    tenure_months = np.random.exponential(24, n_samples).clip(1, 120).astype(int)
    num_purchases = np.random.poisson(15, n_samples) + np.random.randint(0, 50, n_samples)
    avg_order_value = np.random.normal(120, 60, n_samples).clip(10, 500)
    recency_days = np.random.exponential(30, n_samples).clip(1, 365).astype(int)
    web_visits = np.random.poisson(20, n_samples) + np.random.randint(0, 100, n_samples)
    support_tickets = np.random.poisson(1.5, n_samples)
    gender = np.random.choice(['Male','Female','Other'], n_samples, p=[0.48,0.48,0.04])
    region = np.random.choice(['North','South','East','West','Central'], n_samples)
    membership = np.random.choice(['Bronze','Silver','Gold','Platinum'], n_samples, p=[0.4,0.3,0.2,0.1])
    
    df = pd.DataFrame({
        'Age': age,
        'AnnualIncome': income,
        'SpendingScore': spending_score,
        'TenureMonths': tenure_months,
        'NumPurchases': num_purchases,
        'AvgOrderValue': avg_order_value,
        'RecencyDays': recency_days,
        'WebVisitsPerMonth': web_visits,
        'SupportTickets': support_tickets,
        'Gender': gender,
        'Region': region,
        'MembershipTier': membership
    })
    
    for col in ['AnnualIncome','SpendingScore','AvgOrderValue','RecencyDays']:
        mask = np.random.rand(n_samples) < 0.05
        df.loc[mask, col] = np.nan

    categories = []
    for i in range(n_samples):
        inc = df.loc[i, 'AnnualIncome'] if not pd.isna(df.loc[i,'AnnualIncome']) else 65000
        spend = df.loc[i, 'SpendingScore'] if not pd.isna(df.loc[i,'SpendingScore']) else 50
        tenure = df.loc[i,'TenureMonths']
        recency = df.loc[i,'RecencyDays'] if not pd.isna(df.loc[i,'RecencyDays']) else 30
        purchases = df.loc[i,'NumPurchases']
        aov = df.loc[i,'AvgOrderValue'] if not pd.isna(df.loc[i,'AvgOrderValue']) else 120
        visits = df.loc[i,'WebVisitsPerMonth']
        
        score_premium = (inc/150000)*0.4 + (aov/500)*0.3 + (spend/100)*0.3
        score_loyal = (tenure/120)*0.4 + (purchases/65)*0.3 + (1 - recency/365)*0.3
        score_budget = (1 - inc/150000)*0.5 + (1 - aov/500)*0.3 + (spend/100)*0.2
        score_at_risk = (recency/365)*0.5 + (df.loc[i,'SupportTickets']/10)*0.3 + (1 - visits/120)*0.2
        
        if score_at_risk > 0.65 and recency > 90:
            cat = 'At-Risk'
        elif score_premium > 0.7 and inc > 90000 and aov > 180:
            cat = 'Premium'
        elif score_loyal > 0.65 and tenure > 36 and purchases > 25:
            cat = 'Loyal'
        elif score_budget > 0.6 and inc < 50000:
            cat = 'Budget'
        else:
            cat = 'Standard'
        
        if np.random.rand() < 0.12:
            cat = np.random.choice(['Premium','Loyal','Standard','Budget','At-Risk'])
        categories.append(cat)
    
    df['CustomerCategory'] = categories
    return df

if __name__ == "__main__":
    df = generate_customer_data()
    print(df['CustomerCategory'].value_counts())
    df.to_csv('artifacts/customers.csv', index=False)
