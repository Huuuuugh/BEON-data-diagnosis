# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
import os
import re

# 列名中英文对照表（不带单位后缀）
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

# 异常值判定阈值 [min, max]
ANOMALY_THRESHOLDS = {
    'tau': [-2, 2],
    'sonic_temperature': [-50, 50],
    'air_temperature': [-50, 50],
    'air_pressure': [50000, 110000],
    'air_density': [0.5, 2.0],
    'air_heat_capacity': [900, 1100],
    'air_molar_volume': [0.020, 0.030],
    'et': [-0.5, 1],
    'water_vapor_density': [0, 0.1],
    'e': [0, 10000],
    'es': [0, 10000],
    'specific_humidity': [0, 0.05],
    'rh': [0, 100],
    'vpd': [-500, 5000],
    'tdew': [-50, 40],
    'wind_speed': [0, 35],
    'wind_dir': [0, 360],
    'u_': [0, 10],
    'l': [-10000, 10000],
    'v_var': [-50, 50],
    'dew_point_mean': [-50, 40],
    'lwin_1_1_1': [100, 500],
    'lwout_1_1_1': [100, 500],
    'ppfd_1_1_1': [0, 3000],
    'p_rain_1_1_1': [0, 0.5],
    'p_rain_1_2_1': [0, 0.5],
    'rh_1_1_1': [0, 100],
    'rh_1_2_1': [0, 100],
    'rn_1_1_1': [-200, 1000],
    'rlnet_1_1_1': [-200, 200],
    'rsnet_1_1_1': [-200, 1000],
    'r_uva_1_1_1': [0, 100],
    'swc_1_1_1': [0, 1],
    'swc_1_2_1': [0, 1],
    'swc_1_3_1': [0, 1],
    'swc_1_4_1': [0, 1],
    'swc_1_5_1': [0, 1],
    'swdif_1_1_1': [0, 1000],
    'swin_1_1_1': [0, 1500],
    'swout_1_1_1': [0, 1000],
    'trn_1_1_1': [-50, 50],
    'ts_1_1_1': [-40, 40],
    'ts_1_2_1': [-40, 40],
    'ts_1_3_1': [-40, 40],
    'ts_1_4_1': [-40, 40],
    'ts_1_5_1': [-40, 40],
    'ta_1_1_1': [-50, 50],
    'ta_1_2_1': [-50, 50],
    'ws_1_1_1': [0, 35],
    'ws_1_2_1': [0, 35],
    'ppfd_1_1_2': [0, 3000],
    'nee_ustar_f': [-50, 50],
    'gpp_ustar_f': [-10, 60],
    'reco_ustar': [-10, 50],
}


def extract_base_name(col_name):
    """从带单位的列名中提取基础列名"""
    match = re.match(r'^([a-zA-Z_0-9]+)', col_name)
    if match:
        return match.group(1)
    return col_name


def diagnose_column_vectorized(df, base_col, full_col_name):
    """使用pandas向量化操作诊断列"""
    missing_dates = []
    anomaly_dates = []
    
    threshold = ANOMALY_THRESHOLDS.get(base_col, None)
    series = df[full_col_name]
    
    # 解析时间
    try:
        times = pd.to_datetime(df['record_time'])
    except:
        return missing_dates, anomaly_dates
    
    # 找出缺失值（NaN或空字符串）
    missing_mask = series.isna() | (series == '') | (series.astype(str).str.strip() == '')
    
    # 获取缺失值的日期
    if missing_mask.any():
        missing_times = times[missing_mask]
        missing_dates = [(t.year, t.month, t.day) for t in missing_times]
    
    # 如果有阈值，检查异常值
    if threshold is not None:
        # 转换为数值，非数值变为NaN
        numeric_values = pd.to_numeric(series, errors='coerce')
        
        # 异常值：不是缺失但超出范围
        anomaly_mask = (~missing_mask) & (
            (numeric_values < threshold[0]) | (numeric_values > threshold[1])
        )
        
        if anomaly_mask.any():
            anomaly_times = times[anomaly_mask]
            anomaly_dates = [(t.year, t.month, t.day) for t in anomaly_times]
    
    return missing_dates, anomaly_dates


def format_output(dates_by_ymd):
    """格式化输出：月份：日期列表"""
    if not dates_by_ymd:
        return None
    
    # 按月份分组
    month_dates = {}
    for year, month, day in dates_by_ymd:
        if month not in month_dates:
            month_dates[month] = set()
        month_dates[month].add(day)
    
    if not month_dates:
        return None
    
    result_parts = []
    for month in sorted(month_dates.keys()):
        days = sorted(list(month_dates[month]))
        days_str = '、'.join(map(str, days))
        result_parts.append(f"{month}月：{days_str}")
    
    return '\n'.join(result_parts)


def main():
    DATA_DIR = r"C:\Users\Administrator\.openclaw\workspace-client-d\data"
    OUTPUT_FILE = r"C:\Users\Administrator\.openclaw\workspace-client-d\数据诊断报告.txt"
    
    print("=" * 60)
    print("BEON百华山通量站数据诊断程序")
    print("=" * 60)
    print()
    
    # 查找所有CSV文件
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    csv_files.sort()
    
    # 读取并合并所有CSV
    dfs = []
    for f in csv_files:
        filepath = os.path.join(DATA_DIR, f)
        df = pd.read_csv(filepath)
        dfs.append(df)
        print(f"已读取: {f}, 行数: {len(df)}")
    
    # 合并数据
    all_data = pd.concat(dfs, ignore_index=True)
    print(f"\n合并后总行数: {len(all_data)}")
    print(f"原始列数: {len(all_data.columns)}")
    
    # 建立列名映射
    column_mapping = {}
    for col in all_data.columns:
        base = extract_base_name(col)
        if base not in column_mapping:
            chinese = COLUMN_NAME_MAPPING.get(base, base)
            column_mapping[base] = (col, chinese)
    
    print(f"识别到 {len(column_mapping)} 个有效数据列")
    print()
    
    # 存储报告内容
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BEON百华山通量站数据诊断报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"数据文件: {', '.join(csv_files)}")
    report_lines.append(f"总记录数: {len(all_data)}")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("【一、数据概览】")
    report_lines.append(f"  - 数据列数: {len(column_mapping)}")
    report_lines.append("")
    report_lines.append("【二、各指标数据异常明细】")
    report_lines.append("")
    
    # 遍历每列进行诊断
    for base_col in sorted(column_mapping.keys()):
        full_col_name, chinese_name = column_mapping[base_col]
        print(f"正在诊断: {chinese_name} ({base_col})...")
        
        missing_dates, anomaly_dates = diagnose_column_vectorized(all_data, base_col, full_col_name)
        
        # 列标题
        report_lines.append(f"【{chinese_name}】（{base_col}）")
        
        # 统计信息
        total_missing = len(missing_dates)
        total_anomaly = len(anomaly_dates)
        total_issues = total_missing + total_anomaly
        
        report_lines.append(f"  缺失值数量: {total_missing}  |  异常值数量: {total_anomaly}")
        
        if total_issues == 0:
            report_lines.append("  [OK] 数据正常，无异常")
        else:
            if total_missing > 0:
                missing_output = format_output(missing_dates)
                if missing_output:
                    report_lines.append(f"  缺失值日期：")
                    report_lines.append(f"  {missing_output}")
            
            if total_anomaly > 0:
                anomaly_output = format_output(anomaly_dates)
                if anomaly_output:
                    report_lines.append(f"  异常值日期：")
                    report_lines.append(f"  {anomaly_output}")
        
        report_lines.append("")
    
    # 写入报告文件
    report_content = '\n'.join(report_lines)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n报告已生成: {OUTPUT_FILE}")
    print("\n" + "=" * 60)
    print("报告预览（前80行）：")
    print("=" * 60)
    for line in report_lines[:80]:
        print(line)
    
    return report_content


if __name__ == "__main__":
    main()
