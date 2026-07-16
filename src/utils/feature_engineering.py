def feature_engineer(df):
    df = df.copy()
    df['has_agent'] = df['agent'].notna().astype(int)
    df['has_company'] = df['company'].notna().astype(int)
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['total_guests'] = df['adults'] + df['children'].fillna(0) + df['babies']
    df = df.drop(columns=['agent', 'company'])
    return df