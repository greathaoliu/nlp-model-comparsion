import pandas as pd
import csv
import os

def load_prediction_data(file_path):
    def read_csv_robust(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
            headers = next(reader)
            data = []
            for i, row in enumerate(reader, start=2):
                try:
                    if len(row) == len(headers):
                        data.append(row)
                    else:
                        print(f"Skipping malformed row {i}: {row}")
                except Exception as e:
                    print(f"Error reading row {i}: {e}")
        return pd.DataFrame(data, columns=headers)
    
    try:
        df = pd.read_csv(file_path, quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True, escapechar='\\')
    except Exception as e:
        print(f"Error reading CSV with pandas: {e}")
        print("Attempting to read file with custom method...")
        df = read_csv_robust(file_path)
    
    return df


if __name__=='__main__':
    models = {'xlnet', 'roberta'}
    years = {'2011-2019', '2020-2022', '2023'}
    scores = {'score1', 'score2'}
    thres = [
        ('less', 2),
        ('greater', 7),
        ('greater', 8)
     ]

    for model in models:
        for year in years:
            for score in scores:
                for less_or_greater, threshold in thres:
                    # 读取CSV文件
                    input_file = f'./predict_csv/{year}-{model}.csv'  # 请替换为您的输入文件名
                    output_file = f'./filtered_csv/{model}/{score}/{year}-{model}-{score}_{less_or_greater}_than_{threshold}.csv'  # 输出文件名

                    if not os.path.exists(input_file):
                        continue

                    if os.path.exists(output_file):
                        print(f'{output_file} 已存在')
                        continue

                    # 读取CSV文件
                    df = load_prediction_data(input_file)

                    # 将score列转换为数值类型，无法转换的值设为NaN
                    df[f'{score}_预测'] = pd.to_numeric(df[f'{score}_预测'], errors='coerce')

                    if less_or_greater == 'less':
                        filtered_df = df[df[f'{score}_预测'] < threshold].dropna(subset=[f'{score}_预测'])
                    else:
                        filtered_df = df[df[f'{score}_预测'] > threshold].dropna(subset=[f'{score}_预测'])

                    # 将结果写入新的CSV文件
                    filtered_df.to_csv(output_file, index=False)

                    print(f"已将{score} {less_or_greater} than {threshold}的行写入到 {output_file}")
                    print(f"总行数: {len(df)}, 筛选后行数: {len(filtered_df)}")