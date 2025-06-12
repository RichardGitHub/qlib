import qlib
import numpy as np
import pandas as pd
from qlib.config import REG_CN
from qlib.contrib.model.double_ensemble import DEnsembleModel
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.model.pytorch_transformer import TransformerModel
from qlib.contrib.data.handler import DataHandlerLP
from qlib.data.dataset import DatasetH
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
from pathlib import Path
import pickle
import torch
from tqdm import tqdm
import model_utils
from multiprocessing import freeze_support
from qlib.contrib.data.handler import DynamicAlphaCustom
import itertools
import matplotlib.pyplot as plt
from qlib.data import D

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('progressive_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SafeTransformerModel(TransformerModel):
    """完全修正的Transformer包装器，包含完整错误处理"""
    def __init__(self, **kwargs):
        try:
            # 参数检查和默认值设置
            nhead = kwargs.get('nhead', 8)
            d_feat = kwargs.get('d_feat', 64)  # 默认特征维度
            
            # 计算合适的d_model（已修正括号问题）
            if 'd_model' not in kwargs:
                kwargs['d_model'] = ((d_feat + nhead - 1) // nhead) * nhead
            else:
                if kwargs['d_model'] % nhead != 0:
                    new_dim = ((kwargs['d_model'] + nhead - 1) // nhead) * nhead
                    logger.warning(f"自动调整d_model: {kwargs['d_model']} -> {new_dim}")
                    kwargs['d_model'] = new_dim
            
            # 最终验证
            if kwargs['d_model'] % nhead != 0:
                raise ValueError(f"d_model({kwargs['d_model']})必须能被nhead({nhead})整除")
            
            # 调试日志
            logger.debug(f"""
            Transformer配置:
            输入维度(d_feat): {d_feat}
            模型维度(d_model): {kwargs['d_model']}
            注意力头数(nhead): {nhead}
            是否整除: {kwargs['d_model'] % nhead == 0}
            """)
            
            super().__init__(**kwargs)
            
        except Exception as e:
            logger.error(f"Transformer初始化失败: {str(e)}")
            # 回退到安全配置
            safe_kwargs = {
                'd_feat': 64,
                'd_model': 64,
                'nhead': 8,
                'num_layers': 2,
                'dropout': 0.1
            }
            safe_kwargs.update(kwargs)  # 保留其他有效参数
            safe_kwargs['d_model'] = ((safe_kwargs['d_feat'] + 7) // 8) * 8  # 确保可被8整除
            super().__init__(**safe_kwargs)
            logger.info("已使用安全参数初始化Transformer")

class FeatureProjector:
    """处理不同股票池间的特征迁移"""
    @staticmethod
    def get_feature_mapping(old_features: List[str], new_features: List[str]) -> Tuple[np.ndarray, List[str]]:
        """生成特征投影矩阵和公共特征列表"""
        common_features = list(set(old_features) & set(new_features))
        mapping = np.zeros((len(new_features), len(old_features)))
        
        for i, new_feat in enumerate(new_features):
            if new_feat in common_features:
                j = old_features.index(new_feat)
                mapping[i, j] = 1
                
        return mapping, common_features

class ProgressiveModelManager:
    """管理渐进训练中的模型迁移"""
    def __init__(self, model_dir: str = "./saved_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
    
    def save_model(self, model: DEnsembleModel, instruments: str, feat_names: List[str]):
        """保存模型和特征信息"""
        # 保存模型
        model_path = self.model_dir / f"model_{instruments}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 保存特征
        feat_path = self.model_dir / f"features_{instruments}.pkl"
        with open(feat_path, 'wb') as f:
            pickle.dump(feat_names, f)
    
    def load_previous_model(self, current_instruments: str, instruments_order: List[str]) -> Tuple[Optional[DEnsembleModel], Optional[List[str]]]:
        """加载最近一个阶段的模型"""
        idx = instruments_order.index(current_instruments)
        if idx == 0:
            return None, None
        
        for prev_instruments in reversed(instruments_order[:idx]):
            model_path = self.model_dir / f"model_{prev_instruments}.pkl"
            feat_path = self.model_dir / f"features_{prev_instruments}.pkl"
            
            if model_path.exists() and feat_path.exists():
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(feat_path, 'rb') as f:
                    feat_names = pickle.load(f)
                return model, feat_names
        
        return None, None

def get_dynamic_model(instruments: str, feat_dim: int, device: str) -> DEnsembleModel:
    """动态调整模型结构（确保embed_dim可被nhead整除）"""    
    # 计算适配的d_model（大于等于feat_dim的最小可整除值）
    def calc_d_model(base_dim: int, nhead: int) -> int:
        return ((base_dim + nhead - 1) // nhead) * nhead
    
    model_config = {
        "csi300": {
            "lgb_params": {"num_leaves": 32, "learning_rate": 0.05},
            "trans_params": {
                "nhead": 4,
                "d_feat": feat_dim,
                "d_model": calc_d_model(feat_dim, 4),
            },
            "weights": [0.4, 0.6],
            "decay": 0.5
        },
        "csi500": {
            "lgb_params": {"num_leaves": 128, "learning_rate": 0.02},
            "trans_params": {
                "nhead": 12,
                "d_feat": feat_dim,
                "d_model": calc_d_model(feat_dim, 8),
                "dropout": 0.3
            },
            "weights": [0.4, 0.6],
            "decay": 0.7
        },
        "csi800": {
            "lgb_params": {"num_leaves": 64, "learning_rate": 0.02, "feature_fraction": 0.8, "bagging_fraction": 0.8},
            "trans_params": {
                "nhead": 8,
                "d_feat": feat_dim,
                "d_model": calc_d_model(feat_dim, 8),
                "dropout": 0.2
            },
            "weights": [0.5, 0.5],
            "decay": 0.7
        },
        "all": {
            "lgb_params": {"num_leaves": 64, "learning_rate": 0.01},
            "trans_params": {
                "nhead": 8,
                "d_model": 64,
                "batch_size": 256
            },
            "weights": [0.5, 0.5],
            "decay": 0.9
        },
        "default": {
            "lgb_params": {"num_leaves": 128, "learning_rate": 0.02},
            "trans_params": {
                "nhead": 12,
                "d_feat": feat_dim,
                "d_model": calc_d_model(feat_dim, 12),
                "batch_size": 128
            },
            "weights": [0.2, 0.8],
            "decay": 0.8
        }
    }
    
    config = model_config.get(instruments, model_config["default"])
    
    logger.info(f"Creating model for {instruments}: "
               f"feat_dim={feat_dim}, d_model={config['trans_params']['d_model']}, "
               f"nhead={config['trans_params']['nhead']}")
    
    return DEnsembleModel(
        models=[
            (LGBModel(**config["lgb_params"]), config["weights"][0]),
            (SafeTransformerModel(
                **config["trans_params"],
                GPU=0 if device.startswith("cuda") else -1
            ), config["weights"][1])
        ],
        method="average",
        decay=config["decay"]
    )

def transfer_model_weights(old_model: DEnsembleModel, new_model: DEnsembleModel, feat_mapping: np.ndarray):
    """适用于 pyqlib 0.9.6.99 的权重迁移方案"""
    try:
        # 调试日志：打印模型结构
        logger.debug(f"Old model type: {type(old_model)}")
        logger.debug(f"New model type: {type(new_model)}")
        
        # 方案1：检查是否是 DoubleEnsemble 结构
        if hasattr(old_model, 'ensemble') and hasattr(new_model, 'ensemble'):
            logger.info("Detected DoubleEnsemble structure")
            
            # 迁移 Transformer 部分
            if (len(old_model.ensemble) > 1 and len(new_model.ensemble) > 1 and
                isinstance(old_model.ensemble[1], TransformerModel) and
                isinstance(new_model.ensemble[1], TransformerModel)):
                
                old_trans = old_model.ensemble[1].model
                new_trans = new_model.ensemble[1].model
                
                with torch.no_grad():
                    # 输入层权重迁移
                    if hasattr(old_trans, 'input_proj') and hasattr(new_trans, 'input_proj'):
                        old_weight = old_trans.input_proj.weight
                        new_weight = torch.matmul(
                            torch.tensor(feat_mapping, dtype=torch.float32).to(old_weight.device),
                            old_weight
                        )
                        new_trans.input_proj.weight.copy_(new_weight)
                    
                    # 位置编码迁移
                    if hasattr(old_trans, 'pos_enc') and hasattr(new_trans, 'pos_enc'):
                        new_trans.pos_enc.pe.copy_(old_trans.pos_enc.pe)
            
            # 迁移 LGB 部分（通过预测值初始化）
            if len(old_model.ensemble) > 0 and len(new_model.ensemble) > 0:
                try:
                    if isinstance(old_model.ensemble[0], LGBModel):
                        dataset = new_model.ensemble[0].get_dataset()
                        if dataset is not None:
                            init_pred = old_model.ensemble[0].predict(dataset)
                            new_model.ensemble[0].set_init_score(init_pred)
                except Exception as e:
                    logger.warning(f"LGB transfer skipped: {str(e)}")
        
        # 方案2：检查旧版 models 属性
        elif hasattr(old_model, 'models') and hasattr(new_model, 'models'):
            logger.info("Detected legacy models structure")
            # Transformer部分权重迁移
            if (hasattr(old_model.models[1][0], 'model') and 
                hasattr(new_model.models[1][0], 'model')):
                
                old_transformer = old_model.models[1][0].model
                new_transformer = new_model.models[1][0].model
                
                # 输入层投影
                with torch.no_grad():
                    old_weight = old_transformer.input_proj.weight
                    new_transformer.input_proj.weight[:] = torch.matmul(
                        torch.tensor(feat_mapping, dtype=torch.float32).to(old_weight.device),
                        old_weight
                    )
                    
                    # 位置编码迁移
                    if hasattr(old_transformer, 'pos_enc'):
                        new_transformer.pos_enc.pe = old_transformer.pos_enc.pe
            
            # LightGBM模型无法直接迁移，但可以设置初始分数
            if hasattr(new_model.models[0][0], 'set_init_score'):
                new_model.models[0][0].set_init_score(
                    old_model.models[0][0].predict(new_model.models[0][0].get_dataset())
                )        
        else:
            logger.warning("Unrecognized model structure - using prediction-based transfer")
            # 回退方案：通过预测值迁移
            try:
                dataset = new_model.get_dataset()
                if dataset is not None:
                    init_pred = old_model.predict(dataset)
                    new_model.set_init_score(init_pred)
            except Exception as e:
                logger.error(f"Prediction-based transfer failed: {str(e)}")
        
        return new_model
    
    except Exception as e:
        logger.error(f"Weight transfer failed completely: {str(e)}")
        return new_model

def progressive_train_enhanced(
    instruments_order: List[str] = ["csi300", "csi500", "csi800", "all"],
    device: str = "cpu",
    start_date: str = "2015-01-01",
    end_date: str = "2025-12-31"
):
    """增强版渐进训练流程"""
    
    model_manager = ProgressiveModelManager()
    
    for instruments in tqdm(instruments_order, desc="Progressive Training"):
        logger.info(f"\n{'='*40}\nTraining: {instruments} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n{'='*40}")
        
        # 1. 数据准备
        handler = DynamicAlphaCustom(
            instruments=instruments,
            start_time=start_date,
            end_time=end_date,
            fit_start_time=start_date,
            fit_end_time=str(int(start_date[:4])+6) + "-12-31"  # 前6年训练
        )
        # 启用多标签
        if instruments == "csi500":
            label_expr, label_names = handler.get_label_config(
                return_days=5,      # 更短持有期
                quantile=0.8,       # 更高分位数
                multi_label=True,   # 多标签
                risk_label=True     # 风险标签
            )
        else:
            label_expr, label_names = handler.get_label_config(
                return_days=5,
                quantile=0.8,
                multi_label=True,
                risk_label=True
            )
        # 获取特征并保存
        features, feat_names = handler.get_feature_config()
        feat_dim = len(features)
        logger.info(f"Feature dimension: {feat_dim}")
        dataset = DatasetH(
            handler=handler,
            segments={
                "train": (start_date, str(int(start_date[:4])+7) + "-12-31"),
                "valid": (str(int(start_date[:4])+8) + "-01-01", str(int(start_date[:4])+8) + "-12-31"),
                "test": (str(int(start_date[:4])+9) + "-01-01", end_date)
            },
            label=label_expr,
            label_name=label_names
        )
      
        # 新增：输出特征和标签分布日志
        log_feature_and_label_distribution(handler, dataset, instruments, logger)
        # 2. 模型初始化
        model = get_dynamic_model(instruments, feat_dim, device)
        
        # 3. 迁移学习处理
        prev_model, prev_feat_names = model_manager.load_previous_model(instruments, instruments_order)
        if prev_model and prev_feat_names:
            logger.info("Applying transfer learning...")
            model = enhanced_feature_transfer(prev_model, prev_feat_names, model, feat_names, dataset)
        
        # 4. 训练
        model.fit(dataset)
        
        # 5. 保存模型和特征
        model_manager.save_model(model, instruments, feat_names)
        
        # 6. 验证和交易模拟
        run_daily_prediction_and_trading_enhanced(model, dataset, instruments)

def run_daily_prediction_and_trading_enhanced(model, dataset, instruments, norm_method="zscore"):
    """增强版每日预测和交易"""
    today = datetime.now().strftime("%Y-%m-%d")
    start_test = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    
    # 动态调整回测参数
    backtest_params = {
        "csi300": {"topk": 50, "ndrop": 5},
        "csi500": {"topk": 80, "ndrop": 8},
        "csi800": {"topk": 40, "ndrop": 8},
        "default": {"topk": 100, "ndrop": 10}
    }
    params = backtest_params.get(instruments, backtest_params["default"])
    
    # 更新数据集时间段
    dataset.config(segments={"test": (start_test, today)})
    
    # 预测
    pred = model.predict(dataset, segment="test")
    test_data = dataset.prepare("test")
    pred_df = pd.DataFrame(pred, index=test_data.index, columns=['score']).reset_index()
    pred_df = apply_filters(pred_df)
    # 改进的归一化（兼容旧版API）
    try:
        # 尝试新方法（如果industry列存在）
        if "industry" in test_data.columns:
            pred_df = pred_df.groupby(["datetime", "industry"]).apply(
                lambda x: model_utils.normalize_scores(x, method=norm_method)
            ).reset_index(drop=True)
        else:
            pred_df = model_utils.normalize_scores(pred_df, method=norm_method)
    except Exception as e:
        logger.warning(f"Normalization fallback: {str(e)}")
        # 回退到简单归一化
        pred_df['score_norm'] = (pred_df['score'] - pred_df['score'].mean()) / pred_df['score'].std()
    
    # 生成交易信号
    top_stocks = pred_df.sort_values('score_norm', ascending=False).head(params["topk"])
    trade_plan_path = f"trading_plans/{today}_{instruments}.csv"
    Path(trade_plan_path).parent.mkdir(exist_ok=True)
    top_stocks.to_csv(trade_plan_path, index=False)
    
    # 回测
    strategy = TopkDropoutStrategy(
        signal=pred,
        topk=params["topk"],
        n_drop=params["ndrop"]
    )
    
    report, _ = backtest_daily(
        start_time=start_test,
        end_time=today,
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
    
    # 改进的风险分析
    analysis = enhanced_risk_analysis(report['return'], instruments)
    print(f"\n[Performance Report - {instruments}]")
    logger.info(analysis)
    #print(analysis)

# 在生成策略信号前预处理
def apply_filters(df):
    # 过滤NaN
    df = df[df['score'].notna()]
    # 过滤极端值 (3个标准差之外)
    zscore = (df['score'] - df['score'].mean()) / df['score'].std()
    df = df[zscore.abs() < 3]
    return df

def enhanced_risk_analysis(returns: pd.Series, instruments: str) -> pd.DataFrame:
    """增强版风险分析"""
    returns = pd.to_numeric(returns, errors='coerce').dropna()
    
    # 动态基准
    if "csi300" in instruments:
        benchmark = "SH000300"
    elif "csi500" in instruments:
        benchmark = "SH000905"
    else:
        benchmark = "SH000985"  # 中证全指
    
    analysis = {
        "Annualized Return": returns.mean() * 252,
        "Volatility": returns.std() * np.sqrt(252),
        "Sharpe Ratio": returns.mean() / returns.std() * np.sqrt(252),
        "Information Ratio": returns.mean() / returns.std()  * np.sqrt(252) if returns.std()  != 0 else 0,
        "Max Drawdown": (returns.cumsum() - returns.cumsum().cummax()).min(),
        "Win Rate": (returns > 0).mean(),
        #"Benchmark Correlation": returns.corr(qlib.get_data(benchmark, fields="close")),
        "Tail Risk (VaR 5%)": returns.quantile(0.05)
    }
    
    return pd.DataFrame.from_dict(analysis, orient='index', columns=['Value'])

def enhanced_feature_transfer(prev_model, prev_feat_names, new_model, new_feat_names, dataset):
    """
    增强特征迁移：
    - 监控特征迁移覆盖率
    - 监控迁移前后信号相关性
    - 必要时采用模型融合平滑过渡
    """
    feat_mapping, common_feats = FeatureProjector.get_feature_mapping(prev_feat_names, new_feat_names)
    coverage = np.sum(feat_mapping) / feat_mapping.size
    logger.info(f"特征迁移覆盖率: {coverage:.2%}，公共特征数: {len(common_feats)}")
    # 权重迁移
    new_model = transfer_model_weights(prev_model, new_model, feat_mapping)
    # 信号相关性监控
    try:
        prev_pred = prev_model.predict(dataset, segment="valid")
        new_pred = new_model.predict(dataset, segment="valid")
        corr = np.corrcoef(prev_pred, new_pred)[0, 1]
        logger.info(f"迁移前后信号相关性: {corr:.4f}")
        if corr < 0.3:
            logger.warning("信号相关性过低，采用模型融合平滑过渡")
            new_model = ensemble_model_fusion(prev_model, new_model, alpha=0.5)
    except Exception as e:
        logger.warning(f"信号相关性监控失败: {e}")
    return new_model

def ensemble_model_fusion(prev_model, new_model, alpha=0.5):
    """
    模型融合：短期内用新旧模型加权，逐步提升新模型权重
    """
    class FusionModel:
        def __init__(self, m1, m2, alpha_schedule):
            self.m1 = m1
            self.m2 = m2
            self.alpha_schedule = alpha_schedule
            self.call_count = 0
        def predict(self, dataset, segment="test"):
            alpha = self.alpha_schedule[min(self.call_count, len(self.alpha_schedule)-1)]
            self.call_count += 1
            p1 = self.m1.predict(dataset, segment=segment)
            p2 = self.m2.predict(dataset, segment=segment)
            return alpha * p2 + (1 - alpha) * p1
        def fit(self, dataset):
            self.m2.fit(dataset)
    logger.info(f"融合模型：alpha={alpha}")
    return FusionModel(prev_model, new_model, alpha_schedule=[alpha])

def rolling_backtest(predictor, instruments, start_date, end_date, window='1M'):
    """
    多周期回测：分周期输出回测指标
    """
    import pandas as pd
    dates = pd.date_range(start=start_date, end=end_date, freq=window)
    results = []
    for i in range(len(dates)-1):
        s, e = dates[i].strftime('%Y-%m-%d'), dates[i+1].strftime('%Y-%m-%d')
        res = predictor.backtest_strategy(instruments, s, e)
        logger.info(f"{instruments} {s} ~ {e}: {res.get('risk_analysis', {})}")
        results.append({'start': s, 'end': e, 'result': res})
    return results

def rolling_train_and_backtest(instruments_order, window='3M', train_span='24M', start='2015-01-01', end='2025-12-31'):
    """
    滚动窗口训练：定期用最新数据训练，提升模型对突发市场变化的适应性
    """
    import pandas as pd
    all_dates = pd.date_range(start=start, end=end, freq=window)
    for i in range(len(all_dates)-1):
        train_start = (all_dates[i] - pd.DateOffset(months=int(train_span[:-1]))).strftime('%Y-%m-%d')
        train_end = all_dates[i].strftime('%Y-%m-%d')
        test_start = all_dates[i].strftime('%Y-%m-%d')
        test_end = all_dates[i+1].strftime('%Y-%m-%d')
        # 训练
        progressive_train_enhanced(
            instruments_order=instruments_order,
            start_date=train_start,
            end_date=train_end
        )
        # 回测
        for inst in instruments_order:
            predictor = StockPredictor()
            res = predictor.backtest_strategy(inst, test_start, test_end)
            logger.info(f"{inst} {test_start}~{test_end}: {res.get('risk_analysis', {})}")

def analyze_holdings(holdings_df, logger=None):
    freq = holdings_df['instrument'].value_counts()
    weight_sum = holdings_df.groupby('instrument')['weight'].sum()
    top_stocks = weight_sum.sort_values(ascending=False).head(10)
    if logger:
        logger.info(f"持仓出现频率Top10: {freq.head(10)}")
        logger.info(f"累计权重Top10: {top_stocks}")
    return freq, top_stocks

def analyze_signal_distribution(signal, logger=None):
    desc = signal.describe()
    if logger:
        logger.info(f"信号分布统计: {desc}")
    return desc

def dynamic_topk_n_drop(signal, base_topk=10, base_n_drop=2):
    std = signal.std()
    topk = int(base_topk * (1 + std))
    n_drop = int(base_n_drop * (1 + std))
    return max(1, topk), max(0, n_drop)

# 新增：批量参数实验与自动化胜率优化

def log_feature_and_label_distribution(handler, dataset, instruments_name, logger):
    # 特征分布
    try:
        df_feat = dataset.prepare("train", col_set="feature")
        # logger.info(f"[{instruments_name}] 过滤前特征分布:\n{df_feat.shape}")
        # # 自动过滤高缺失股票和高缺失日期，兼容MultiIndex
        # if isinstance(df_feat.index, pd.MultiIndex):
        #     # 过滤高缺失股票
        #     instruments = df_feat.index.get_level_values('instrument')
        #     stock_missing = instruments.value_counts()
        #     high_missing_stocks = stock_missing[stock_missing > 100].index
        #     mask_stock = ~instruments.isin(high_missing_stocks)
        #     df_feat = df_feat[mask_stock]
        #     # 过滤高缺失日期
        #     datetimes = df_feat.index.get_level_values('datetime')
        #     date_missing = datetimes.value_counts()
        #     high_missing_dates = date_missing[date_missing > 50].index
        #     mask_date = ~datetimes.isin(high_missing_dates)
        #     df_feat = df_feat[mask_date]
        # else:
        #     instruments = df_feat['instrument'] if 'instrument' in df_feat.columns else None
        #     datetimes = df_feat['datetime'] if 'datetime' in df_feat.columns else None
        #     if instruments is not None:
        #         stock_missing = instruments.value_counts()
        #         high_missing_stocks = stock_missing[stock_missing > 100].index
        #         mask_stock = ~instruments.isin(high_missing_stocks)
        #         df_feat = df_feat[mask_stock]
        #     if datetimes is not None:
        #         date_missing = datetimes.value_counts()
        #         high_missing_dates = date_missing[date_missing > 50].index
        #         mask_date = ~datetimes.isin(high_missing_dates)
        #         df_feat = df_feat[mask_date]
        # logger.info(f"[{instruments_name}] 过滤后特征分布:\n{df_feat.shape}")
        
        desc = df_feat.describe().T        
        logger.info(f"[{instruments_name}] 特征分布:\n{desc}")
        # 检查极端均值/方差/缺失
        missing = df_feat.isnull().sum()
        missing_rate = missing / len(df_feat)
        # 导出特征缺失率
        missing_rate_path = f"feature_missing_rate_{instruments_name}.csv"
        missing_rate.to_csv(missing_rate_path)
        logger.info(f"[{instruments_name}] 已导出特征缺失率明细: {missing_rate_path}")
        # 导出缺失样本明细
        missing_rows_path = f"feature_missing_rows_{instruments_name}.csv"
        df_feat[df_feat.isnull().any(axis=1)].to_csv(missing_rows_path)
        logger.info(f"[{instruments_name}] 已导出特征缺失样本明细: {missing_rows_path}")
        for name, row in desc.iterrows():
            if abs(row['mean']) > 10 or row['std'] > 10:
                logger.warning(f"[{instruments_name}] 特征异常: {name} mean={row['mean']:.2f}, std={row['std']:.2f}")
            if row['count'] < 0.9 * len(df_feat):
                logger.warning(f"[{instruments_name}] 特征缺失: {name} count={row['count']} / {len(df_feat)}")
    except Exception as e:
        logger.error(f"[{instruments_name}] 特征分布统计失败: {e}")
    # 标签分布
    try:
        df_label = dataset.prepare("train", col_set="label")
        label_imbalance = {}
        for col in df_label.columns:
            vc = df_label[col].value_counts(dropna=False)
            logger.info(f"[{instruments_name}] 标签[{col}]分布:\n{vc}")
            if vc.min() < 0.05 * vc.sum():
                logger.warning(f"[{instruments_name}] 标签[{col}]类别极端不均衡: {vc.to_dict()}")
                label_imbalance[col] = vc
        # 导出极端不均衡标签明细
        if label_imbalance:
            label_imbalance_path = f"label_imbalance_{instruments_name}.csv"
            pd.DataFrame(label_imbalance).to_csv(label_imbalance_path)
            logger.info(f"[{instruments_name}] 已导出标签极端不均衡明细: {label_imbalance_path}")
    except Exception as e:
        logger.error(f"[{instruments_name}] 标签分布统计失败: {e}")

def batch_experiment(
    instruments_list=["csi300", "csi500", "csi800", "all"],
    quantile_list=[0.7, 0.75, 0.8, 0.85],
    return_days_list=[2, 3, 5],
    multi_label=False,
    risk_label=False,
    device="cpu",
    start_date="2015-01-01",
    end_date="2025-12-31"
):
    results = []
    for instruments in instruments_list:
        for quantile, return_days in itertools.product(quantile_list, return_days_list):
            print(f"\n==== {instruments} | quantile={quantile} | return_days={return_days} ====")
            # 1. 数据准备
            handler = DynamicAlphaCustom(
                instruments=instruments,
                start_time=start_date,
                end_time=end_date,
                fit_start_time=start_date,
                fit_end_time=str(int(start_date[:4])+6) + "-12-31"
            )
            label_expr, label_names = handler.get_label_config(
                return_days=return_days,
                quantile=quantile,
                multi_label=multi_label,
                risk_label=risk_label
            )
            features, feat_names = handler.get_feature_config()
            feat_dim = len(features)
            dataset = DatasetH(
                handler=handler,
                segments={
                    "train": (start_date, str(int(start_date[:4])+7) + "-12-31"),
                    "valid": (str(int(start_date[:4])+8) + "-01-01", str(int(start_date[:4])+8) + "-12-31"),
                    "test": (str(int(start_date[:4])+9) + "-01-01", end_date)
                },
                label=label_expr,
                label_name=label_names
            )
            # 新增：输出特征和标签分布日志
            log_feature_and_label_distribution(handler, dataset, instruments, logger)
            model = get_dynamic_model(instruments, feat_dim, device)
            model.fit(dataset)
            # 生成test分段信号，并补齐index
            test_start = str(int(start_date[:4])+9) + "-01-01"
            test_end = end_date
            signal = model.predict(dataset, segment="test")
            # 若为多index（如多股票），需按日期补齐
            if hasattr(signal, 'index') and hasattr(signal.index, 'levels') and len(signal.index.levels) == 2:
                # 多股票信号，补齐所有日期-股票组合
                idx = pd.MultiIndex.from_product([D.calendar(start_time=test_start, end_time=test_end, freq='day'), signal.index.levels[1]], names=signal.index.names)
                signal = signal.reindex(idx).fillna(0)
            else:
                # 单一index，按日期补齐
                signal = signal.reindex(D.calendar(start_time=test_start, end_time=test_end, freq='day')).fillna(0)
            # 回测
            report, _ = backtest_daily(
                start_time=test_start,
                end_time=test_end,
                strategy=TopkDropoutStrategy(signal=signal, topk=10, n_drop=2),
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
            # 风险分析
            analysis = enhanced_risk_analysis(report['return'], instruments)
            # 记录结果
            result = {
                "instruments": instruments,
                "quantile": quantile,
                "return_days": return_days,
            }
            for k, v in analysis["Value"].items():
                result[k] = v
            results.append(result)
            logger.info(f"[{instruments}] 实验结果: {result}")
            print(analysis)
    # 输出csv
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("batch_experiment_results.csv", index=False)
    print("已保存批量实验结果到 batch_experiment_results.csv")
    # 可视化分析
    for instruments in instruments_list:
        sub = df[df["instruments"] == instruments]
        plt.figure(figsize=(10, 6))
        for return_days in return_days_list:
            sub2 = sub[sub["return_days"] == return_days]
            plt.plot(sub2["quantile"], sub2["Win Rate"], marker='o', label=f"return_days={return_days}")
        plt.title(f"{instruments} Win Rate vs Quantile")
        plt.xlabel("quantile")
        plt.ylabel("Win Rate")
        plt.legend()
        plt.savefig(f"{instruments}_winrate_vs_quantile.png")
        plt.close()
    print("已生成胜率对比可视化图片")

# 用法示例（可在main中调用）
if __name__ == "__main__":
    freeze_support()
    qlib.init(provider_uri="./qlib_data/cn_data", region=REG_CN)
    progressive_train_enhanced(
        instruments_order=["csi300", "csi500", "csi800", "all"],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    # 启动批量实验
    # batch_experiment(
    #     instruments_list=["csi500"],  # 可改为全部指数
    #     quantile_list=[0.7, 0.75, 0.8, 0.85],
    #     return_days_list=[2, 3, 5, 7],
    #     multi_label=False,
    #     risk_label=False,
    #     device="cpu"
    # )