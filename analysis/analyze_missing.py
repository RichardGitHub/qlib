import pandas as pd

# 读取缺失样本明细
df = pd.read_csv('feature_missing_rows_csi300.csv')

# 统计每个特征的缺失次数
missing_per_feature = df.isnull().sum().sort_values(ascending=False)
print("各特征缺失次数：\n", missing_per_feature)

# 统计缺失最多的前10个特征
print("缺失最多的特征Top10：\n", missing_per_feature.head(10))

# 统计每只股票缺失样本数
if 'instrument' in df.columns:
    missing_per_stock = df['instrument'].value_counts().head(10)
    print("缺失最多的股票Top10：\n", missing_per_stock)

# 统计每个日期缺失样本数
if 'datetime' in df.columns:
    missing_per_date = df['datetime'].value_counts().head(10)
    print("缺失最多的日期Top10：\n", missing_per_date)