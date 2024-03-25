import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

Nominal_Attribute = ['LocationAbbr', 'LocationDesc', 'Class', 'Topic',
                         'Question', 'DataValueTypeID', 'Stratification1',
                         'StratificationCategory2', 'ClassID', 'QuestionID',
                         'StratificationCategoryID2', 'StratificationID2']
Number_Attribute = ['YearStart', 'YearEnd', 'Data_Value', 'Data_Value_Alt',
                     'LocationID']
file = 'datasets/Alzheimer Disease and Healthy Aging Data In US.csv'


# 封装打印标称属性分布的函数
def print_nominal_attributes(data, nominal_attributes):
    print('标称属性:')
    for attribute in nominal_attributes:
        print('-------------------------------------------------')
        print(attribute + ":")
        print(data[attribute].value_counts())

# 封装计算五数概括的函数
def five_number_summary(data, attribute):
    data_attribute = data.dropna(subset=[attribute])[attribute]
    Min_num = min(data_attribute)
    Max_num = max(data_attribute)
    Q1_num = np.percentile(data_attribute, 25)
    Median_num = np.median(data_attribute)
    Q3_num = np.percentile(data_attribute, 75)
    return Min_num, Max_num, Q1_num, Median_num, Q3_num


# 封装打印数值属性信息的函数
def print_numeric_attributes(data, numeric_attributes):
    print('数值属性:')
    for attribute in numeric_attributes:
        print('-----------------------------------------------------')
        print(attribute + ":")
        print('缺失值个数：')
        print(data[attribute].isnull().sum())
        print('五值概括:')
        summary = five_number_summary(data, attribute)
        print(summary)


# 封装绘制直方图的函数
def plot_histogram(data, column_name, bins=48, width=0.5, color='blue', figsize=(10, 6)):
    plt.figure(figsize=figsize)
    sub_data = data.dropna(subset=[column_name])[column_name]
    plt.xticks(rotation=90)
    plt.hist(sub_data, bins=bins, width=width, color=color)
    plt.grid(alpha=0.5, linestyle='-.')
    plt.xlabel(f'name of {column_name}')
    plt.ylabel('frequencies')
    plt.title(f'the frequency of {column_name}')
    plt.show()


# 封装绘制箱线图的函数
def plot_boxplot(data, column_name, figsize=(6, 8)):
    plt.figure(figsize=figsize)
    sub_data = data.dropna(subset=[column_name])[column_name]
    sub_data.plot.box(title='box plot')
    plt.grid(linestyle="--", alpha=0.5)
    plt.ylabel('values')
    plt.title(f'Box plot of column "{column_name}"')
    plt.show()


# 打印缺失数据列表
def print_missing_data(df, column_name):
    print('缺失数据列表：')
    print(df[df[column_name].isnull()].iloc[:10, 10:])


# 剔除缺失数据
def drop_missing_data(df, column_name):
    return df.dropna(subset=[column_name])


# 用最高频率值来填补缺失值
def fill_missing_with_median(df, column_name):
    df_copy = df.copy(deep=True)
    df_copy[column_name] = df_copy[column_name].fillna(
        np.median(df_copy.dropna(subset=[column_name])[column_name]), inplace=False)
    return df_copy


# 通过属性的相关关系来填补缺失值（使用interpolate方法）
def fill_missing_with_interpolate(df, column_name):
    df_copy = df.copy(deep=True)
    df_copy[column_name] = df_copy[column_name].interpolate()
    return df_copy


# 通过数据对象之间的相似性来填补缺失值（使用均值）
def fill_missing_with_mean(df, column_name):
    df_copy = df.copy(deep=True)
    df_copy[column_name] = df_copy[column_name].fillna(
        np.mean(df_copy[column_name]), inplace=False)
    return df_copy


if __name__ == "__main__" :
    data = pd.read_csv(file)

    print("数据摘要")
    print_nominal_attributes(data, Nominal_Attribute)
    print_numeric_attributes(data, Number_Attribute)

    print("数据可视化")
    # 绘制的直方图
    plot_histogram(data, 'Class')
    plot_histogram(data, 'LocationAbbr')
    plot_histogram(data, 'Topic')

    # 绘制箱线图
    plot_boxplot(data, 'YearStart')
    plot_boxplot(data, 'YearEnd')
    plot_boxplot(data, 'Data_Value')
    plot_boxplot(data, 'Data_Value_Alt')

    print("数据缺失的处理")
    # 打印原始缺失数据列表
    print_missing_data(data, 'Data_Value')

    # 剔除缺失数据
    data1 = drop_missing_data(data, 'Data_Value')
    print_missing_data(data1, 'Data_Value')

    # 用最高频率值来填补缺失值
    data2 = fill_missing_with_median(data, 'Data_Value')
    print_missing_data(data2, 'Data_Value')

    # 通过属性的相关关系来填补缺失值
    data3 = fill_missing_with_interpolate(data, 'Data_Value')
    print_missing_data(data3, 'Data_Value')

    # 通过数据对象之间的相似性来填补缺失值
    data4 = fill_missing_with_mean(data, 'Data_Value')
    print_missing_data(data4, 'Data_Value')

