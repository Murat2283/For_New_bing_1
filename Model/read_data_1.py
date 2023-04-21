
import pandas as pd

df = pd.read_csv(r'..\data\半导体\20_bdt_stock_data.csv')


def get_stock_data(df, data, x, y):
    """
    统计指定日期内有多少股票信息，然后提取指定日期前 x 天和后 y 天的股票数据分别返回。

    参数：
    df: pandas.DataFrame，包含股票数据的 DataFrame。
    data: str，日期字符串，格式为 YYYY-MM-DD，指定要查询的日期。
    x: int，查询日期前 x 天的数据。
    y: int，查询日期后 y 天的数据。

    返回值：
    num_stocks: int，指定日期内股票信息的数量。
    df_before: pandas.DataFrame，查询日期前 x 天的股票数据。
    df_after: pandas.DataFrame，查询日期后 y 天的股票数据。
    """

    # 将索引设置为日期和s_index
    df.set_index(['Date', 's_index'], inplace=True)

    # 统计指定日期内股票信息的数量
    df_data = df.loc[data]
    num_stocks = len(df_data)

    # 将日期字符串转换为日期类型
    date = pd.to_datetime(data)

    # 计算起始日期和结束日期
    start_date = date - pd.DateOffset(days=x)
    end_date = date + pd.DateOffset(days=y)

    # 筛选前x天的数据
    before_dates = pd.date_range(start=start_date, end=date).strftime('%Y-%m-%d').tolist()
    df_before = df[df.index.get_level_values('Date').isin(before_dates)]

    # 筛选后y天的数据
    after_dates = pd.date_range(start=date, end=end_date).strftime('%Y-%m-%d').tolist()
    df_after = df[df.index.get_level_values('Date').isin(after_dates)]

    # 返回结果
    return num_stocks, df_before, df_after

num_stocks, df_before, df_after = get_stock_data(df,'2023-04-21', 7, 7)



print(f"指定日期内股票信息的数量为：{num_stocks}")
print("查询日期前 7 天的股票数据：")
print(df_before)
print("查询日期后 7 天的股票数据：")
print(df_after)



