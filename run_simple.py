# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
import os
import re

# ==================== 1. 列名中英文对照表 ====================
COLUMN_NAME_MAPPING = {
    'record_time': '记录时间',
    'tau': '动量通量',
    'sonic_temperature': '超声温度',
    'air_temperature': '空气温度',
    'air_pressure': '气压',
    'air_density': '空气密度',
    'air_heat_capacity': '空气热容',
    'air_molar_volume': '空气摩尔体积',
    'et': '蒸散发',
    'water_vapor_density': '水汽密度',
    'e': '水汽分压',
    'es': '饱和水汽分压',
    'specific_humidity': '比湿',
    'rh': '相对湿度',
    'vpd': '饱和水汽压差',
    'tdew': '露点温度',
    'wind_speed': '平均风速',
    'wind_dir': '风向',
    'u_': '摩擦速率',
    'l': '莫宁-奥布霍夫长度',
    'v_var': '垂直风速方差',
    'dew_point_mean': '平均露点温度',
    'lwin_1_1_1': '入射长波辐射',
    'lwout_1_1_1': '反射长波辐射',
    'ppfd_1_1_1': '入射光合有效辐射',
    'p_rain_1_1_1': '下层降水量',
    'p_rain_1_2_1': '上层降水量',
    'rh_1_1_1': '下层空气相对湿度',
    'rh_1_2_1': '上层空气相对湿度',
    'rn_1_1_1': '净辐射',
    'rlnet_1_1_1': '长波净辐射',
    'rsnet_1_1_1': '短波净辐射',
    'r_uva_1_1_1': '紫外辐射',
    'swc_1_1_1': '土壤含水量1',
    'swc_1_2_1': '土壤含水量2',
    'swc_1_3_1': '土壤含水量3',
    'swc_1_4_1': '土壤含水量4',
    'swc_1_5_1': '土壤含水量5',
    'swdif_1_1_1': '散射辐射',
    'swin_1_1_1': '向下短波辐射',
    'swout_1_1_1': '向上短波辐射',
    'trn_1_1_1': '净辐射表本体温度',
    'ts_1_1_1': '土壤温度1',
    'ts_1_2_1': '土壤温度2',
    'ts_1_3_1': '土壤温度3',
    'ts_1_4_1': '土壤温度4',
    'ts_1_5_1': '土壤温度5',
    'ta_1_1_1': '下层空气温度',
    'ta_1_2_1': '上层空气温度',
    'ws_1_1_1': '下层风速',
    'ws_1_2_1': '上层风速',
    'ppfd_1_1_2': '向上光合有效辐射',
    'nee_ustar_f': 'CO2净通量',
    'gpp_ustar_f': '总初级生产力',
    'reco_ustar': '生态系统呼吸',
}

# ==================== 2. 逻辑阈值（基于生态学知识） ====================
DOMAIN_THRESHOLDS = {
    'air_temperature': [-35, 45],
    'sonic_temperature': [-35, 45],
    'ta_1_1_1': [-35, 45],
    'ta_1_2_1': [-35, 45],
    'ts_1_1_1': [-30, 35],
    'ts_1_2_1': [-30, 35],
    'ts_1_3_1': [-30, 35],
    'ts_1_4_1': [-30, 35],
    'ts_1_5_1': [-30, 35],
    'trn_1_1_1': [-40, 50],
    'air_pressure': [70000, 100000],
    'rh': [0, 100],
    'rh_1_1_1': [0, 100],
    'rh_1_2_1': [0, 100],
    'wind_speed': [0, 30],
    'ws_1_1_1': [0, 30],
    'ws_1_2_1': [0, 30],
    'wind_dir': [0, 360],
    'rn_1_1_1': [-150, 900],
    'rsnet_1_1_1': [-100, 800],
    'rlnet_1_1_1': [-150, 200],
    'lwin_1_1_1': [150, 550],
    'lwout_1_1_1': [200, 600],
    'swin_1_1_1': [0, 1200],
    'swout_1_1_1': [0, 400],
    'swdif_1_1_1': [0, 400],
    'r_uva_1_1_1': [0, 80],
    'ppfd_1_1_1': [0, 2500],
    'ppfd_1_1_2': [0, 2500],
    'p_rain_1_1_1': [0, 0.3],
    'p_rain_1_2_1': [0, 0.3],
    'swc_1_1_1': [0.01, 0.8],
    'swc_1_2_1': [0.01, 0.8],
    'swc_1_3_1': [0.01, 0.8],
    'swc_1_4_1': [0.01, 0.8],
    'swc_1_5_1': [0.01, 0.8],
    'e': [0, 8000],
    'es': [0, 8000],
    'vpd': [-500, 4000],
    'nee_ustar_f': [-50, 50],
    'gpp_ustar_f': [-5, 60],
    'reco_ustar': [0, 40],
    'tau': [-2, 2],
    'et': [-0.5, 3],
    'u_': [0, 2],
    'l': [-10000, 10000],
    'v_var': [-10, 30],
    'tdew': [-40, 35],
    'dew_point_mean': [-40, 35],
    'air_density': [0.8, 1.5],
    'air_heat_capacity': [950, 1050],
    'air_molar_volume': [0.020, 0.035],
    'water_vapor_density': [0, 0.03],
    'specific_humidity': [0, 0.03],
}


def extract_base_name(col_name):
    match = re.match(r'^([a-zA-Z_0-9]+)', col_name)
    return match.group(1) if match else col_name


def calculate_modified_zscore(series):
    """
    计算Modified Z-score
    Modified Z-score = 0.6745 * (x - median) / MAD
    当|MoD Z-score| > 5.0 时认为是离群点
    """
    numeric_vals = pd.to_numeric(series, errors='coerce')
    valid_vals = numeric_vals.dropna()
    
    if len(valid_vals) < 10:
        return pd.Series([np.nan] * len(series), index=series.index)
    
    median = valid_vals.median()
    mad = (valid_vals - median).abs().median()
    
    if mad == 0:
        return pd.Series([np.nan] * len(series), index=series.index)
    
    modified_z = 0.6745 * (numeric_vals - median) / mad
    return modified_z


def diagnose_column_improved(df, base_col, full_col_name):
    anomaly_dates = []
    series = df[full_col_name]
    times = pd.to_datetime(df['record_time'], errors='coerce')
    numeric_values = pd.to_numeric(series, errors='coerce')
    
    # 1. 逻辑阈值检测
    domain_threshold = DOMAIN_THRESHOLDS.get(base_col, None)
    domain_violated = pd.Series([False] * len(series), index=series.index)
    
    if domain_threshold is not None:
        valid_mask = ~(numeric_values.isna() | series.astype(str).str.strip().eq(''))
        domain_violated = valid_mask & (
            (numeric_values < domain_threshold[0]) | 
            (numeric_values > domain_threshold[1])
        )
    
    # 2. Modified Z-score检测
    modified_z = calculate_modified_zscore(numeric_values)
    valid_for_z = ~(numeric_values.isna() | series.astype(str).str.strip().eq(''))
    zscore_outlier = valid_for_z & (modified_z.abs() > 5.0)
    
    # 3. 组合判定
    final_anomaly = domain_violated | zscore_outlier
    
    # 收集异常日期
    anomaly_mask = final_anomaly & times.notna()
    if anomaly_mask.any():
        anomaly_times = times[anomaly_mask]
        anomaly_dates = [(t.year, t.month, t.day) for t in anomaly_times]
    
    return anomaly_dates


def format_output_ymd(dates_by_ymd):
    if not dates_by_ymd:
        return None
    
    year_month_days = {}
    for year, month, day in dates_by_ymd:
        key = (year, month)
        if key not in year_month_days:
            year_month_days[key] = set()
        year_month_days[key].add(day)
    
    if not year_month_days:
        return None
    
    result_parts = []
    for (year, month) in sorted(year_month_days.keys()):
        days = sorted(list(year_month_days[(year, month)]))
        days_str = '、'.join(map(str, days))
        result_parts.append(f"{year}年{month}月：{days_str}")
    
    return '\n'.join(result_parts)


def main():
    DATA_DIR = r"C:\Users\Administrator\.openclaw\workspace-client-d\data"
    OUTPUT_FILE = r"C:\Users\Administrator\.openclaw\workspace-client-d\数据诊断报告.txt"
    
    print("=" * 60)
    print("BEON百华山通量站数据诊断程序")
    print("改进版：逻辑阈值 + Modified Z-score")
    print("=" * 60)
    print()
    
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    csv_files.sort()
    
    dfs = []
    for f in csv_files:
        filepath = os.path.join(DATA_DIR, f)
        df = pd.read_csv(filepath)
        dfs.append(df)
        print(f"已读取: {f}, 行数: {len(df)}")
    
    all_data = pd.concat(dfs, ignore_index=True)
    print(f"\n合并后总行数: {len(all_data)}")
    
    column_mapping = {}
    for col in all_data.columns:
        base = extract_base_name(col)
        if base not in column_mapping:
            chinese = COLUMN_NAME_MAPPING.get(base, base)
            column_mapping[base] = (col, chinese)
    
    print(f"识别到 {len(column_mapping)} 个有效数据列")
    print()
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BEON百华山通量站数据诊断报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"数据文件: {', '.join(csv_files)}")
    report_lines.append(f"总记录数: {len(all_data)}")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("【检测方法】")
    report_lines.append("  1. 逻辑阈值法：基于生态学知识制定各指标的物理/生物学合理范围")
    report_lines.append("  2. Modified Z-score：统计方法，检测偏离中位数超过5.0倍MAD的离群点")
    report_lines.append("  异常判定：逻辑阈值违规 OR |Modified Z-score| > 5.0")
    report_lines.append("")
    report_lines.append("【各指标数据异常明细】（仅列出存在异常的指标）")
    report_lines.append("")
    
    anomaly_count = 0
    
    for base_col in sorted(column_mapping.keys()):
        full_col_name, chinese_name = column_mapping[base_col]
        print(f"正在诊断: {chinese_name} ({base_col})...")
        
        anomaly_dates = diagnose_column_improved(all_data, base_col, full_col_name)
        
        if len(anomaly_dates) > 0:
            anomaly_count += 1
            report_lines.append(f"【{anomaly_count}、{chinese_name}】（{base_col}）")
            report_lines.append(f"  异常值数量: {len(anomaly_dates)}")
            
            anomaly_output = format_output_ymd(anomaly_dates)
            if anomaly_output:
                report_lines.append(f"  异常值日期：")
                report_lines.append(f"  {anomaly_output}")
            
            report_lines.append("")
    
    report_content = '\n'.join(report_lines)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n报告已生成: {OUTPUT_FILE}")
    print(f"共有 {anomaly_count} 个指标存在异常")
    
    return report_content


if __name__ == "__main__":
    main()
