import qlib
import numpy as np
import pandas as pd
from qlib.config import REG_CN
from qlib.contrib.model.double_ensemble import DEnsembleModel
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.model.pytorch_transformer import TransformerModel
from qlib.contrib.model.xgboost import XGBModel
from qlib.contrib.data.handler import Alpha360
from qlib.data.dataset import DatasetH
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from datetime import datetime, timedelta
from multiprocessing import freeze_support
import os

# 导入自定义工具模块
import model_utils

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

    handler = Alpha360(
        start_time="2015-01-01",
        end_time="2025-12-31",
        fit_start_time="2015-01-01",
        fit_end_time="2022-12-31",
        instruments="csi300",
        label=["Ref($close, -5)/$close - 1 > 0.02"]  # 分类标签
    )

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ("2015-01-01", "2021-12-31"),
            "valid": ("2022-01-01", "2022-12-31"),
            "test": ("2023-01-01", "2025-12-31")
        }
    )

    # 判断是否已有训练好的模型，复用避免重复训练
    if os.path.exists(MODEL_PATH):
        print("[Main] 发现已有模型文件，尝试加载...")
        model = model_utils.load_model(MODEL_PATH)
    else:
        print("[Main] 未检测到模型文件，开始训练新模型...")
        model = DEnsembleModel(
            models=[
                (LGBModel(), 0.3),
                (XGBModel(), 0.3),
                (TransformerModel(d_feat=360, GPU=0 if device.startswith("cuda") else -1), 0.4)
            ],
            method="average",
            decay=0.5
        )
         # 设定设备
        model_utils.set_device_to_model(model, device=device)
        model.fit(dataset)
        model_utils.save_model(model, MODEL_PATH)

    # 预测前也设置设备（如果加载模型，防止设备不匹配）
    model_utils.set_device_to_model(model, device=device)
    # 运行每日预测和调仓逻辑，归一化方法支持切换
    run_daily_prediction_and_trading(model, dataset, norm_method="zscore")

if __name__ == "__main__":
    freeze_support()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="Device to use, e.g., 'cpu' or 'cuda' or 'cuda:0'")
    args = parser.parse_args()
    main(device=args.device)
