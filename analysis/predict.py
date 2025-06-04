import qlib
import numpy as np
import pandas as pd
from qlib.config import REG_CN
from qlib.contrib.model.double_ensemble import DEnsembleModel
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.model.pytorch_transformer import TransformerModel
from qlib.contrib.model.xgboost import XGBModel
from qlib.contrib.data.handler import Alpha360, Alpha158, AlphaSimpleCustom
from qlib.data.dataset import DatasetH
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from datetime import datetime, timedelta
from multiprocessing import freeze_support
import os

# 导入自定义工具模块
import model_utils
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qlib_predict.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def safe_model_training(model, dataset, **kwargs) -> Optional[Any]:
    """安全的模型训练包装器"""
    try:
        logger.info(f"开始训练模型: {model.__class__.__name__}")
        model.fit(dataset, **kwargs)
        logger.info("模型训练完成")
        return model
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        return None

def safe_feature_calculation(handler, **kwargs):
    """安全的特征计算包装器"""
    try:
        logger.info("开始验证特征配置")
        # get_feature_config返回的是(fields, names)元组，不是字典
        fields, names = handler.get_feature_config()
        logger.info(f"特征配置验证完成，共生成 {len(fields)} 个特征: {names}")
        return handler
    except Exception as e:
        logger.error(f"特征配置验证失败: {e}")
        return handler

@contextmanager
def timer(description):
    """计时上下文管理器"""
    start = time.time()
    logger.info(f"开始 {description}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"{description} 完成，耗时: {elapsed:.2f}秒")

def validate_data_quality(dataset) -> Dict[str, Any]:
    """数据质量验证"""
    stats = {
        'total_samples': 0,
        'missing_values': 0,
        'feature_count': 0,
        'inf_values': 0
    }
    
    try:
        # 检查数据完整性
        with timer("数据质量检查"):
            # 获取训练数据
            train_data = dataset.prepare(segments=['train'], col_set='feature')
            if train_data is not None and len(train_data) > 0:
                stats['total_samples'] = len(train_data)
                stats['feature_count'] = train_data.shape[1] if hasattr(train_data, 'shape') else 0
                
                # 检查 NaN 值
                if hasattr(train_data, 'isnull'):
                    stats['missing_values'] = train_data.isnull().sum().sum()
                
                # 检查无限值
                if hasattr(train_data, 'select_dtypes'):
                    numeric_data = train_data.select_dtypes(include=[np.number])
                    stats['inf_values'] = np.isinf(numeric_data).sum().sum()
                
                if stats['missing_values'] > 0:
                    logger.warning(f"发现 {stats['missing_values']} 个缺失值")
                if stats['inf_values'] > 0:
                    logger.warning(f"发现 {stats['inf_values']} 个无限值")
            else:
                logger.info("训练数据为空或无法访问")
                    
    except Exception as e:
        logger.warning(f"数据质量检查失败: {e}")
    
    return stats

MODEL_PATH = "./saved_models/densemble_model.pkl"

def run_daily_prediction_and_trading(model, dataset, norm_method="zscore"):
    today = datetime.today().strftime("%Y-%m-%d")
    start_test = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")

    # 配置最新数据段
    dataset.config(segments={"test": (start_test, today)})

    # 预测
    pred = model.predict(dataset, segment="test")
    test_data = dataset.prepare("test")
    pred_df = pd.DataFrame(pred, index=test_data.index, columns=['score']).reset_index()

    # 归一化预测分数
    pred_df = model_utils.normalize_scores(pred_df, method=norm_method, group_by_date=True)

    # 生成调仓建议，使用归一化后的score_norm排序
    top_stocks = pred_df.sort_values(by='score_norm', ascending=False).head(50)
    top_stocks.to_csv(f"daily_trading_plan_{today}.csv", index=False)
    print(f"\n[每日调仓建议] {today}")
    print(top_stocks[['datetime', 'instrument', 'score_norm']])

    # 回测最近一段时间策略表现
    strategy = TopkDropoutStrategy(signal=pred, topk=50, n_drop=5)
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }
    report, positions = backtest_daily(
        start_time=start_test, end_time=today, strategy=strategy, executor=executor_config
    )

    return_series = pd.to_numeric(report['return'], errors='coerce').dropna()
    if not isinstance(return_series.index, pd.DatetimeIndex):
        return_series.index = pd.to_datetime(return_series.index, errors='coerce')

    try:
        analysis = risk_analysis(return_series, freq="day")
    except Exception as e:
        print("[Warning] risk_analysis fallback due to:", e)
        mean_return = float(return_series.mean())
        std_return = float(return_series.std())
        analysis = pd.Series({
            "mean": mean_return,
            "std": std_return,
            "annualized_return": mean_return * 238,
            "information_ratio": mean_return / std_return * np.sqrt(238) if std_return != 0 else 0,
            "max_drawdown": float((return_series.cumsum() - return_series.cumsum().cummax()).min()),
        }).to_frame("risk")

    print("\n[风险指标分析]")
    print(analysis)

def main(device="cpu"):
    print(f"[Main] 使用设备: {device}")
    qlib.init(provider_uri="./qlib_data/cn_data", region=REG_CN)
    instruments = "csi300"
    try:
        handler = AlphaSimpleCustom(
            start_time="2015-01-01",
            end_time="2025-12-31",
            fit_start_time="2015-01-01",
            fit_end_time="2022-12-31",
            instruments=instruments,
            label=["Ref($close, -5)/$close - 1 > 0.02"]  # 分类标签
        )
        
        # 添加特征计算验证，防止Rolling(ATTR, 0)警告
        handler = safe_feature_calculation(handler)
        
        # 获取实际特征数量
        fields, names = handler.get_feature_config()
        actual_feat_count = len(fields)
        logger.info(f"实际特征数量: {actual_feat_count}")
        
    except Exception as e:
        logger.error(f"Handler 初始化失败: {e}")
        return

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ("2015-01-01", "2022-12-31"),
            "valid": ("2023-01-01", "2023-12-31"),
            "test": ("2024-01-01", "2025-12-31")
        }
    )
    
    # 数据质量验证
    data_stats = validate_data_quality(dataset)
    logger.info(f"数据统计信息: {data_stats}")

    # 判断是否已有训练好的模型，复用避免重复训练
    if os.path.exists(MODEL_PATH):
        print("[Main] 发现已有模型文件，尝试加载...")
        with timer("模型加载"):
            model = model_utils.load_model(MODEL_PATH)
    else:
        print("[Main] 未检测到模型文件，开始训练新模型...")
        model = DEnsembleModel(
            models=[
                (LGBModel(), 0.3),
                (TransformerModel(d_feat=actual_feat_count, GPU=0 if device.startswith("cuda") else -1), 0.4)
            ],
            method="average",
            decay=0.5
        )
        # 设定设备
        model_utils.set_device_to_model(model, device=device)
        
        with timer("模型训练"):
            model = safe_model_training(model, dataset)
            if model is None:
                logger.error("模型训练失败，程序退出")
                return
        
        model_utils.save_model(model, MODEL_PATH)

    # 预测前也设置设备（如果加载模型，防止设备不匹配）
    model_utils.set_device_to_model(model, device=device)
    
    # 运行每日预测和调仓逻辑，归一化方法支持切换
    with timer("预测和回测"):
        run_daily_prediction_and_trading(model, dataset, norm_method="zscore")
    
    logger.info("程序执行完成")

if __name__ == "__main__":
    freeze_support()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="Device to use, e.g., 'cpu' or 'cuda' or 'cuda:0'")
    args = parser.parse_args()
    main(device=args.device)
