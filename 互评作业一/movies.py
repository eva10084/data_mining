import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


Nominal_Attribute = ['director', 'industry', 'language', 'title', 'writer']
Number_Attribute = ['IMDb-rating', 'downloads', 'id', 'views']
file = 'datasets/movies_dataset.csv'


def analyze_dataset(data, Nominal_Attribute, Number_Attribute):
    # 打印标称属性信息
    print('标称属性:')
    for attribute in Nominal_Attribute:
        print('---------------------------------------------')
        print(attribute + ":")
        print(data[attribute].value_counts())

    # 定义五数概括的函数
    def five_number(data_series):
        Min_num = min(data_series)
        Max_num = max(data_series)
        Q1_num = np.percentile(data_series, 25)
        Median_num = np.median(data_series)
        Q3_num = np.percentile(data_series, 75)
        return Min_num, Max_num, Q1_num, Median_num, Q3_num

    print('数值属性:')

    # 处理数值属性的逗号并转换为浮点数
    for attribute in Number_Attribute:
        # 首先，确保列是字符串类型
        if not pd.api.types.is_string_dtype(data[attribute]):
            data[attribute] = data[attribute].astype(str)

        data[attribute] = data[attribute].str.replace(',', '').astype(float)

        # data[attribute] = data[attribute].str.replace(',', '').astype(float)
        # 打印数值属性的统计信息
        print('-------------------------------------------------')
        print(attribute + ":")
        print('缺失值个数：')
        print(data[attribute].isnull().sum())
        print('五值概括:')
        non_null_data = data.dropna(subset=[attribute])[attribute]
        print(five_number(non_null_data))





def plot_histogram(data, column_name, bins=48, width=0.5, color='blue', figsize=(10, 6)):
    plt.figure(figsize=figsize)
    sub_data = data.dropna(subset=[column_name])[column_name]
    plt.xticks(rotation=90)
    plt.hist(sub_data, bins=bins, width=width, color=color)
    plt.grid(alpha=0.5, linestyle='-.')
    plt.xlabel(f'name of {column_name.lower()}s')
    plt.ylabel('frequencies')
    plt.title(f'the frequency of {column_name}')
    plt.show()


def generate_and_plot_histogram(data, name, bins=48, width=0.5, color='blue', figsize=(30, 6)):
    attr_list = []
    Attribute_list = data.dropna(subset=[name])[name]
    for attribute_list in Attribute_list:
        Attributes = attribute_list.strip().split(',')
        for attr in Attributes:
            attr_list.append(attr)

    plot_histogram(pd.DataFrame(attr_list), 0, bins=bins, width=width, color=color, figsize=figsize)


def plot_boxplot(data, column_name, figsize=(6, 8)):
    plt.figure(figsize=figsize)
    sub_data = data.dropna(subset=[column_name])[column_name]
    sub_data.plot.box(title='box plot')
    plt.grid(linestyle="--", alpha=0.5)
    plt.ylabel('values')
    plt.title(f'Box plot of column "{column_name}"')
    plt.show()

# 打印指定列中缺失值的行
def print_missing_values(data, columns):
    print('缺失数据列表：')
    print(data[data[columns].isnull().any(axis=1)].head(10))

# 删除数据集中包含缺失值的所有行
def drop_missing_values(data):
    return data.dropna()

# 使用指定列的中位数填充缺失值
def fill_missing_values_with_median(data, column):
    data_copy = data.copy(deep=True)
    median_value = np.median(data_copy.dropna(subset=[column])[column])
    data_copy[column].fillna(median_value, inplace=True)
    return data_copy

# 使用插值法填充指定列的缺失值
def interpolate_missing_values(data, column):
    data_copy = data.copy(deep=True)
    data_copy[column].interpolate(method='linear', inplace=True)
    return data_copy

# 使用指定列的均值填充缺失值
def fill_missing_values_with_mean(data, column):
    data_copy = data.copy(deep=True)
    mean_value = np.mean(data_copy[column])
    data_copy[column].fillna(mean_value, inplace=True)
    return data_copy



if __name__ == "__main__":
    data = pd.read_csv(file)
    print("数据摘要")
    analyze_dataset(data, Nominal_Attribute, Number_Attribute)

    print("数据可视化")
    # 绘制 industry 列的直方图
    plot_histogram(data, 'industry')
    # 生成并绘制 language 列的属性直方图
    # generate_and_plot_histogram(data, 'language')
    # 绘制 IMDb-rating 列的箱线图
    plot_boxplot(data, 'IMDb-rating')
    # 绘制 downloads 列的箱线图
    plot_boxplot(data, 'downloads')
    # 绘制 id 列的箱线图
    plot_boxplot(data, 'id')
    # 绘制 view 列的箱线图
    plot_boxplot(data, 'views')

    print("数据处理")
    # 打印包含 'IMDb-rating' 列缺失值的行
    print_missing_values(data, ['IMDb-rating'])

    # 删除包含缺失值的行
    data_no_na = drop_missing_values(data)

    # 使用 'IMDb-rating' 列的中位数填充缺失值
    data_median_filled = fill_missing_values_with_median(data, 'IMDb-rating')

    # 使用插值法填充 'IMDb-rating' 列的缺失值
    data_interpolated = interpolate_missing_values(data, 'IMDb-rating')

    # 使用 'IMDb-rating' 列的均值填充缺失值
    data_mean_filled = fill_missing_values_with_mean(data, 'IMDb-rating')
