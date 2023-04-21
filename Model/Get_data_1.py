import tushare as ts
import numpy as np
import pandas as pd


# 读取txt文件并将其转换为DataFrame对象
# stocks = pd.read_csv(r'..\data\半导体\20_bdt.txt', sep='\t', header=None, names=['company'])
stocks = ['688012', '002371', '300604', '300567', '300545', '600641', '300316', '600745', '601908', 
          '688072', '688082', '603690', '688037', '688200', '688328', '688383', '688120', '688596', 
          '605358', '688126']


result = {}
for stock in stocks:
    df = ts.get_hist_data(stock)
    
    # 将股票日线数据保存为字典形式
    if df is not None:
        result[stock] = df.to_dict(orient='index')
    else:
        print(f"No data available for {stock}")

# 将字典形式的数据转换为三维结构
df = pd.DataFrame(result).stack().apply(pd.Series)
df.index.set_names(['Date', 's_index'], inplace=True)


# 保存数据到文件中
df.to_csv(r'..\data\半导体\20_bdt_stock_data.csv')


#读取某个日期的数据
date = '2022-04-20' # 要查询的日期
data_on_date = df.loc[date, :] # 获取该日期所有股票的数据

#读取某段日期的数据
start_date = '2022-04-18' # 起始日期
end_date = '2023-04-20' # 结束日期
# df.index = pd.to_datetime(df.index)
data_in_range = df.loc[start_date:end_date]



if isinstance(df.index, pd.DatetimeIndex):
    print("行索引是时间序列类型")
else:
    print("行索引不是时间序列类型")
    
    
    
    
    
#读取某个日期 前30个 后 4个数据
# 读取该日期前30个数据
# date = '2022-04-20'
# start_date = pd.to_datetime(date) - pd.DateOffset(days=30)
# df_before = df.loc[start_date:date]

# # 读取该日期后4个数据
# end_date = pd.to_datetime(date) + pd.DateOffset(days=4)
# df_after = df.loc[date:end_date]


# 读取该日期前30个数据

df.index.set_names(['Date', 's_index'], inplace=True)
date = '2022-04-20'
start_date = pd.to_datetime(date) - pd.DateOffset(days=30)
before_dates = pd.date_range(start=start_date, end=date).strftime('%Y-%m-%d').tolist()
df_before = df[df.index.get_level_values('Date').isin(before_dates)]

# 读取该日期后4个数据
end_date = pd.to_datetime(date) + pd.DateOffset(days=4)
after_dates = pd.date_range(start=date, end=end_date).strftime('%Y-%m-%d').tolist()
df_after = df[df.index.get_level_values('Date').isin(after_dates)]






def get_before_after_data(df,date_str, x, y):
    """
    获取指定日期前 x 天和后 y 天的数据。

    参数：
    date_str: str，日期字符串，格式为 YYYY-MM-DD，指定要查询的日期。
    x: int，查询日期前 x 天的数据。
    y: int，查询日期后 y 天的数据。

    返回值：
    df_before: pandas.DataFrame，查询日期前 x 天的数据。
    df_after: pandas.DataFrame，查询日期后 y 天的数据。
    """
    # 将索引设置为日期和s_index
    df.index.set_names(['Date', 's_index'], inplace=True)

    # 将日期字符串转换为日期类型
    date = pd.to_datetime(date_str)

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
    return df_before, df_after


df_before, df_after = get_before_after_data(df,'2022-04-20', 30, 4)

# print(type(df.index))
# print(df.index.levels)  # 查看每个层级的取值范围
# print(df.index.names)   # 查看每个层级的名称





