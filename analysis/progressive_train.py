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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('progressive_training.log'),
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
                "d_model": calc_d_model(feat_dim, 4),  # 确保可被4整除
            },
            "weights": [0.4, 0.6],
            "decay": 0.5
        },
        "csi500": {
            "lgb_params": {"num_leaves": 64, "learning_rate": 0.03},
            "trans_params": {
                "nhead": 8,
                "d_feat": feat_dim,
                "d_model": calc_d_model(feat_dim, 8),  # 确保可被8整除
            },
            "weights": [0.3, 0.7],
            "decay": 0.7
        },
        "default": {
            "lgb_params": {"num_leaves": 128, "learning_rate": 0.02},
            "trans_params": {
                "nhead": 12,
                "d_feat": feat_dim,
                "d_model": calc_d_model(feat_dim, 12),  # 确保可被12整除
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
    qlib.init(provider_uri="./qlib_data/cn_data", region=REG_CN)
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
            }
        )
        
        # 2. 模型初始化
        model = get_dynamic_model(instruments, feat_dim, device)
        
        # 3. 迁移学习处理
        prev_model, prev_feat_names = model_manager.load_previous_model(instruments, instruments_order)
        if prev_model and prev_feat_names:
            logger.info("Applying transfer learning...")
            feat_mapping, common_feats = FeatureProjector.get_feature_mapping(prev_feat_names, feat_names)
            logger.info(f"Feature transfer: {len(common_feats)} common features")
            
            model = transfer_model_weights(prev_model, model, feat_mapping)
        
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
    print(analysis)
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
        "Max Drawdown": (returns.cumsum() - returns.cumsum().cummax()).min(),
        "Win Rate": (returns > 0).mean(),
        #"Benchmark Correlation": returns.corr(qlib.get_data(benchmark, fields="close")),
        "Tail Risk (VaR 5%)": returns.quantile(0.05)
    }
    
    return pd.DataFrame.from_dict(analysis, orient='index', columns=['Value'])

if __name__ == "__main__":
    freeze_support()
    progressive_train_enhanced(
        instruments_order=["csi300", "csi500", "csi800", "all"],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )