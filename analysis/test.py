import numpy as np
import pandas as pd
import qlib
from qlib.data.dataset import DatasetH
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from qlib.contrib.data.handler import DynamicAlphaCustom
from qlib.config import REG_CN
from multiprocessing import freeze_support

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def prepare_prediction_data(
    instruments: str,
    start_date: str,
    end_date: str,
    lookback_days: int = 60
) -> Optional[DatasetH]:
    """
    准备预测所需的数据集
    :param instruments: 股票池名称
    :param start_date: 开始日期 (YYYY-MM-DD)
    :param end_date: 结束日期 (YYYY-MM-DD)
    :param lookback_days: 回看天数 (用于特征计算)
    :return: 准备好的数据集对象
    """
    try:
        # 动态调整开始日期以确保有足够的历史数据
        actual_start = (datetime.strptime(start_date, "%Y-%m-%d") - 
                      timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        logger.info(f"回测区间: {start_date} ~ {end_date}, 实际数据起始: {actual_start}, 特征最大窗口期: {lookback_days}天")
        
        # 使用与训练时相同的特征处理器
        handler = DynamicAlphaCustom(
            instruments=instruments,
            start_time=actual_start,
            end_time=end_date,
            fit_start_time=actual_start,
            fit_end_time=end_date
        )
        
        # 创建预测数据集 (仅测试集)
        dataset = DatasetH(
            handler=handler,
            segments={
                "test": (actual_start, end_date)
            }
        )
        
        # 新增：数据完整性检查
        try:
            test_data = dataset.prepare("test", col_set=["feature", "label"])
            if test_data.isnull().any().any():
                nan_info = test_data.isnull().sum()
                logger.warning(f"数据中存在NaN，各列NaN数量:\n{nan_info}")
                nan_rows = test_data[test_data.isnull().any(axis=1)]
                logger.warning(f"含NaN的样本（前10行）：\n{nan_rows.head(10)}")
                # 新增：打印含NaN样本的股票、日期及其前lookback_days天行情数据
                try:
                    import qlib
                    for idx in nan_rows.index[:3]:  # 只打印前3个
                        code = idx[1]
                        date = idx[0]
                        start_hist = (pd.to_datetime(date) - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
                        end_hist = pd.to_datetime(date).strftime('%Y-%m-%d')
                        # 取close、volume等基础行情
                        raw = qlib.data.D.features([code], ["$close", "$volume"], start_hist, end_hist)
                        logger.info(f"{code} {date} 前{lookback_days}天行情数据:\n{raw}")
                except Exception as e:
                    logger.warning(f"打印窗口期原始行情数据失败: {e}")
                before = len(test_data)
                test_data = test_data.dropna()
                after = len(test_data)
                logger.warning(f"已自动剔除含NaN样本，剩余样本数: {after}（剔除{before-after}）")
                if after == 0:
                    logger.error("剔除NaN后无有效样本")
                    return None
        except Exception as e:
            logger.error(f"数据完整性检查未通过: {e}")
            if 'test_data' in locals():
                nan_info = test_data.isnull().sum()
                logger.error(f"各列NaN数量:\n{nan_info}")
                nan_rows = test_data[test_data.isnull().any(axis=1)]
                logger.error(f"含NaN的样本（前10行）：\n{nan_rows.head(10)}")
            return None
        # 检查测试集内容
       
        # 新增：打印数据集基本信息
        logger.info(f"数据集基本信息:")
        # logger.info(f"总样本数: {len(dataset)}")  # DatasetH 无法直接取 len()
        logger.info(f"测试集样本数: {len(test_data)}")
        # 兼容 MultiIndex 和普通列名，输出特征列和标签列
        if isinstance(test_data.columns, pd.MultiIndex):
            feature_cols = [col for col in test_data.columns if col[0] == 'feature']
            label_cols = [col for col in test_data.columns if col[0] == 'label']
        else:
            feature_cols = [col for col in test_data.columns if 'feature' in str(col)]
            label_cols = [col for col in test_data.columns if 'label' in str(col)]
        logger.info(f"特征列: {feature_cols}")
        logger.info(f"标签列: {label_cols}")
        logger.info(f"测试集特征列: {test_data.columns}")
        logger.info(f"Prepared prediction data for {instruments} from {start_date} to {end_date}")
        # 假设 test_data 已经准备好
        if isinstance(test_data.index, pd.MultiIndex):
            sh689009_data = test_data.xs("SH689009", level=1, drop_level=False)
        else:
            sh689009_data = test_data[test_data.index.get_level_values(1) == "SH689009"]

        logger.info(f'SH689009 全部数据（前10行）：\n{sh689009_data.head(10)}')
        return dataset
    except Exception as e:
        logger.error(f"Failed to prepare data: {str(e)}")
        return None

if __name__ == '__main__':
    freeze_support()
    
    qlib.init(provider_uri="./qlib_data/cn_data", region=REG_CN)
    today = datetime.now().strftime("%Y-%m-%d")
    start_test = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    backtest_start_date = start_test
    backtest_end_date = today

    dataset = prepare_prediction_data(instruments="csi500", start_date=backtest_start_date, end_date=backtest_end_date)


