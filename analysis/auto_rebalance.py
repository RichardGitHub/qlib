import qlib
from qlib.config import REG_CN
from datetime import datetime, timedelta
import pandas as pd
from stock_predictor import StockPredictor
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest_daily
import logging
import os
from progressive_train import SafeTransformerModel
import glob
from stock_predictor import dynamic_topk_n_drop

def run_daily_rebalance(
    instruments="csi800",
    target_date=None,
    topk=5,
    n_drop=2,
    model_dir="./saved_models",
    lookback_days=90,
    days_window=7,
    holdings_dir="holdings"
):
    """
    自动调仓主流程，支持指定股票池和调仓日期。
    :param instruments: 股票池名称，如"csi800"、"csi500"、"csi300"、"all"等
    :param target_date: 调仓日期（字符串，格式YYYY-MM-DD），如"2025-06-10"
    :param topk: 最多持仓股票数
    :param n_drop: 每天最多调仓股票数
    :param model_dir: 模型目录
    :param lookback_days: 特征窗口期
    :param days_window: 用于信号生成的数据区间长度（天数）
    :param holdings_dir: 持仓csv目录
    """
    # 日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 1. 初始化QLib
    qlib.init(provider_uri="./qlib_data/cn_data", region=REG_CN)

    # 2. 日期区间
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=days_window)).strftime("%Y-%m-%d")
    end_date = target_date

    # 3. 加载模型与数据
    predictor = StockPredictor(model_dir=model_dir)
    if not predictor.load_latest_model(instruments):
        logger.error("模型加载失败")
        return

    dataset = predictor.prepare_prediction_data(
        instruments=instruments,
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days
    )
    if dataset is None:
        logger.error("数据准备失败")
        return

    # 4. 预测并归一化信号
    pred = predictor.current_model.predict(dataset, segment="test")
    test_data = dataset.prepare("test", col_set=["feature", "label"])
    pred_df = pd.DataFrame({
        "instrument": test_data.index.get_level_values("instrument"),
        "date": test_data.index.get_level_values("datetime"),
        "score": pred
    })
    filtered_df = predictor._filter_predictions(pred_df)
    if len(filtered_df) == 0:
        logger.error("无有效信号")
        return

    # 5. 构建信号Series
    signal = filtered_df.set_index(["date", "instrument"])["score_norm"].sort_index()

    # 6. 构建调仓策略
    strategy = TopkDropoutStrategy(
        signal=signal,
        topk=topk,
        n_drop=n_drop
    )

    # 7. 回测/模拟调仓（可选，实盘可跳过）
    report, positions = backtest_daily(
        start_time=start_date,
        end_time=end_date,
        strategy=strategy,
        executor={
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
                "verbose": False
            }
        }
    )

    # 8. 生成指定日期应买入/卖出股票列表
    holdings = []
    if isinstance(positions, dict) and len(positions) > 0:
        for date, v in positions.items():
            # 参考StockPredictor.py的健壮处理
            if isinstance(v, dict):
                pass  # 直接用
            elif hasattr(v, 'to_dict'):
                v_dict = v.to_dict()
                if not v_dict:
                    v_dict = v.__dict__
                v = v_dict
            elif hasattr(v, '__dict__'):
                v = v.__dict__
            else:
                # 无法处理的类型，跳过
                continue
            if 'position' in v and isinstance(v['position'], dict):
                pos_dict = v['position']
                for inst, detail in pos_dict.items():
                    if str(inst).lower() in ['cash', 'now_account_value']:
                        continue
                    if not (isinstance(detail, dict) or hasattr(detail, 'keys')):
                        continue
                    holdings.append({
                        'date': date,
                        'instrument': inst,
                        'weight': detail.get('weight', None)
                    })
    holdings_df = pd.DataFrame(holdings)
    holdings_df['date'] = pd.to_datetime(holdings_df['date'])
    holdings_df = holdings_df.sort_values(['date', 'instrument'])
    unique_dates = sorted(holdings_df['date'].unique())
    if pd.to_datetime(target_date) not in unique_dates:
        logger.warning(f"指定日期 {target_date} 无持仓数据")
        return
    idx = unique_dates.index(pd.to_datetime(target_date))

    # === 新增：读取最新持仓csv作为昨日持仓 ===
    os.makedirs(holdings_dir, exist_ok=True)
    prev_holdings_file = None
    prev_holdings = set()
    csv_files = sorted(glob.glob(os.path.join(holdings_dir, "holdings_*.csv")))
    if csv_files:
        # 取最新日期的csv
        prev_holdings_file = csv_files[-1]
        prev_df = pd.read_csv(prev_holdings_file)
        if 'instrument' in prev_df.columns:
            prev_holdings = set(prev_df['instrument'])
    
    # === 今日目标持仓 ===
    today_holdings = holdings_df[holdings_df['date'] == pd.to_datetime(target_date)][['instrument', 'weight']]
    today_set = set(today_holdings['instrument'])

    # === 获取目标日期信号分布 ===
    day_df = holdings_df[holdings_df['date'] == pd.to_datetime(target_date)]
    if 'score_norm' in day_df.columns:
        signal_series = day_df['score_norm']
    elif 'score' in day_df.columns:
        signal_series = day_df['score']
    else:
        logger.warning('未找到score_norm或score，动态调仓失效，使用默认topk/n_drop')
        signal_series = None

    # === 动态topk/n_drop ===
    if signal_series is not None and len(signal_series) > 0:
        topk_dyn, n_drop_dyn = dynamic_topk_n_drop(signal_series, base_topk=topk, base_n_drop=n_drop)
        logger.info(f"{target_date} 动态topk: {topk_dyn}, 动态n_drop: {n_drop_dyn} (std={signal_series.std():.4f})")
        topk = topk_dyn
        n_drop = n_drop_dyn
    else:
        logger.info(f"{target_date} 使用默认topk: {topk}, n_drop: {n_drop}")

    # === 调仓明细 ===
    if prev_holdings_file is None:
        # 首日建仓
        buy_list = list(today_set)
        sell_list = []
        is_first_day = True
        logger.info(f"{target_date}（首次建仓）应买入股票: {buy_list}")
    else:
        buy_list = list(today_set - prev_holdings)
        sell_list = list(prev_holdings - today_set)
        is_first_day = False
        logger.info(f"{target_date} 应买入股票: {buy_list}")
        logger.info(f"{target_date} 应卖出股票: {sell_list}")

    logger.info(f"{target_date} 目标持仓:\n{today_holdings}")

    # === 保存今日持仓csv ===
    today_csv = os.path.join(holdings_dir, f"holdings_{target_date.replace('-', '')}.csv")
    today_holdings.to_csv(today_csv, index=False)

    return {
        "buy": buy_list,
        "sell": sell_list,
        "holdings": today_holdings,
        "positions": positions,
        "report": report,
        "is_first_day": is_first_day,
        "prev_holdings_file": prev_holdings_file,
        "today_holdings_file": today_csv
    }

if __name__ == "__main__":
    # 示例：指定日期和股票池
    result = run_daily_rebalance(
        instruments="csi800",           # 股票池
        target_date="2025-07-10",      # 指定调仓日期
        topk=5,
        n_drop=2,
        model_dir="./saved_models"
    )
    # 你可以在此处对 result 进一步处理，如自动下单等