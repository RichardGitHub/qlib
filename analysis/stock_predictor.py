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
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_prediction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入 SafeTransformerModel
try:
    from progressive_train import SafeTransformerModel
except ImportError:
    logger.warning("SafeTransformerModel not available, using standard models only")
    SafeTransformerModel = None

class StockPredictor:
    def __init__(self, model_dir: str = "./saved_models"):
        """
        股票预测器初始化
        :param model_dir: 保存训练模型的目录
        """
        self.model_dir = Path(model_dir)
        self.current_model = None
        self.current_features = None
    
    def load_latest_model(self, instruments: str) -> bool:
        """
        加载指定股票池的最新模型
        :param instruments: 股票池名称 (e.g. "csi300")
        :return: 是否加载成功
        """
        model_path = self.model_dir / f"model_{instruments}.pkl"
        feat_path = self.model_dir / f"features_{instruments}.pkl"
        
        if not model_path.exists() or not feat_path.exists():
            logger.error(f"Model files not found for {instruments}")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                self.current_model = pickle.load(f)
            with open(feat_path, 'rb') as f:
                self.current_features = pickle.load(f)
            logger.info(f"Successfully loaded model for {instruments}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def prepare_prediction_data(
        self,
        instruments: str,
        start_date: str,
        end_date: str,
        lookback_days: int = 90
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
            logger.info(f"准备数据区间: {start_date} ~ {end_date}, 实际数据起始: {actual_start}, 特征最大窗口期: {lookback_days}天")
            
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
                    "test": (start_date, end_date)
                }
            )
            
            # 新增：数据完整性检查
            # try:
            #     test_data = dataset.prepare("test", col_set=["feature", "label"])
            #     if test_data.isnull().any().any():
            #         nan_info = test_data.isnull().sum()
            #         logger.warning(f"数据中存在NaN，各列NaN数量:\n{nan_info}")
            #         nan_rows = test_data[test_data.isnull().any(axis=1)]
            #         logger.warning(f"含NaN的样本（前10行）:\n{nan_rows.head(10)}")
            #         # 新增：打印含NaN样本的股票、日期及其前lookback_days天行情数据
            #         try:
            #             import qlib
            #             for idx in nan_rows.index[:3]:  # 只打印前3个
            #                 code = idx[1]
            #                 date = idx[0]
            #                 start_hist = (pd.to_datetime(date) - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            #                 end_hist = pd.to_datetime(date).strftime('%Y-%m-%d')
            #                 # 取close、volume等基础行情
            #                 raw = qlib.data.D.features([code], ["$close", "$volume"], start_hist, end_hist)
            #                 logger.info(f"{code} {date} 前{lookback_days}天行情数据:\n{raw}")
            #         except Exception as e:
            #             logger.warning(f"打印窗口期原始行情数据失败: {e}")
            #         before = len(test_data)
            #         test_data = test_data.dropna()
            #         after = len(test_data)
            #         logger.warning(f"已自动剔除含NaN样本，剩余样本数: {after}（剔除{before-after}）")
            #         if after == 0:
            #             logger.error("剔除NaN后无有效样本")
            #             return None
            # except Exception as e:
            #     logger.error(f"数据完整性检查未通过: {e}")
            #     if 'test_data' in locals():
            #         nan_info = test_data.isnull().sum()
            #         logger.error(f"各列NaN数量:\n{nan_info}")
            #         nan_rows = test_data[test_data.isnull().any(axis=1)]
            #         logger.error(f"含NaN的样本（前10行）:\n{nan_rows.head(10)}")
            #     return None
            
            # logger.info(f"Prepared prediction data for {instruments} from {start_date} to {end_date}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to prepare data: {str(e)}")
            return None
    
    def predict_stocks(
        self,
        instruments: str,
        prediction_date: str = None,
        top_k: int = 10,
        output_format: str = "raw"
    ) -> Dict:
        """
        执行股票预测
        :param instruments: 股票池名称
        :param prediction_date: 预测日期 (None则为最新日期)
        :param top_k: 返回前K个推荐股票
        :param output_format: 输出格式 ("raw"|"portfolio"|"signals")
        :return: 预测结果字典
        """
        if not self.current_model or self.current_features is None:
            if not self.load_latest_model(instruments):
                # 根据输出格式返回相应的错误结果
                if output_format == "signals":
                    return {
                        "date": prediction_date or datetime.now().strftime("%Y-%m-%d"),
                        "buy": [],
                        "sell": [],
                        "hold": [],
                        "error": "Model not available"
                    }
                elif output_format == "portfolio":
                    return {
                        "date": prediction_date or datetime.now().strftime("%Y-%m-%d"),
                        "stocks": [],
                        "summary": {"num_stocks": 0, "avg_score": 0, "score_std": 0},
                        "error": "Model not available"
                    }
                else:
                    return {"error": "Model not available"}
        
        # 设置预测日期 (默认为当前日期)
        end_date = prediction_date or datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - 
                     timedelta(days=7)).strftime("%Y-%m-%d")  # 一周数据
        
        # 准备数据
        dataset = self.prepare_prediction_data(instruments, start_date, end_date)
        if dataset is None:
            # 根据输出格式返回相应的错误结果
            if output_format == "signals":
                return {
                    "date": end_date,
                    "buy": [],
                    "sell": [],
                    "hold": [],
                    "error": "Data preparation failed"
                }
            elif output_format == "portfolio":
                return {
                    "date": end_date,
                    "stocks": [],
                    "summary": {"num_stocks": 0, "avg_score": 0, "score_std": 0},
                    "error": "Data preparation failed"
                }
            else:
                return {"error": "Data preparation failed"}
        
        try:
            # 获取预测结果
            pred = self.current_model.predict(dataset, segment="test")
            logger.info(f"模型预测分数样例: {pred[:10]}")
            # 检查测试集内容
            test_data = dataset.prepare("test", col_set=["feature", "label"])
            logger.info(f"测试集样例: {test_data.head() if hasattr(test_data, 'head') else str(test_data)[:200]}")

            # ========== 统一过滤和标准化分数 ========== 
            pred_df = pd.DataFrame({
                "instrument": test_data.index.get_level_values("instrument"),
                "date": test_data.index.get_level_values("datetime"),
                "score": pred
            })
            filtered_df = self._filter_predictions(pred_df)
            if len(filtered_df) == 0:
                logger.warning("过滤后无有效信号，回测和信号输出均为空")
                return {
                    "performance": None,
                    "risk_analysis": None,
                    "positions": None,
                    "positions_file": None,
                    "trades_file": None
                }
            # 用过滤后的score_norm作为信号，统一传递给回测和信号/报告
            logger.info("已统一回测和信号/报告的分数源（score_norm，含极端值过滤和标准化）")
            # 构建信号Series，索引为(date, instrument)
            signal = filtered_df.set_index(["date", "instrument"])["score_norm"].sort_index()
            # ========== 新增：信号index检查和修正 ==========
            logger.info(f"signal index type: {type(signal.index)}, names: {signal.index.names}")
            # 强制修正index类型和顺序
            try:
                signal.index = pd.MultiIndex.from_tuples([
                    (pd.to_datetime(d), str(i)) for d, i in signal.index
                ], names=["date", "instrument"])
                logger.info(f"修正后signal index type: {type(signal.index)}, names: {signal.index.names}")
            except Exception as e:
                logger.warning(f"信号index修正失败: {e}")
            # 输出每日可选股票数样例
            logger.info(f'每日可选股票数样例: {filtered_df.groupby("date").size().head()}')

            # 输出每日score_norm分布样例
            try:
                logger.info(f'每日score_norm分布样例: {filtered_df.groupby("date")["score_norm"].describe().head()}')
            except Exception as e:
                logger.warning(f'输出score_norm分布失败: {e}')

            # 在信号生成后分析分布
            analyze_signal_distribution(filtered_df['score_norm'], logger)

            # 在策略生成前动态调整风控参数
            topk, n_drop = dynamic_topk_n_drop(filtered_df['score_norm'], top_k, 2)

            # 生成信号输出
            if output_format == "signals":
                signals = self._generate_trading_signals(filtered_df, mode="topk", top_k=topk)
                return signals
            elif output_format == "portfolio":
                return self._generate_portfolio(filtered_df, top_k)
            else:
                return {"predictions": filtered_df}
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            return {"error": str(e)}
    
    def _filter_predictions(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤预测结果，并增强日志和数据完整性检查
        :param pred_df: 原始预测DataFrame
        :return: 过滤后的DataFrame
        """
        logger.info(f"原始预测样本数: {len(pred_df)}")
        # 1. 补全/剔除NaN值
        pred_df = pred_df.copy()
        nan_count = pred_df["score"].isna().sum()
        if nan_count > 0:
            logger.warning(f"score列存在NaN数量: {nan_count}，将用前向/后向填充后再剔除剩余NaN")
            pred_df["score"] = pred_df["score"].fillna(method="ffill").fillna(method="bfill")
        filtered = pred_df[pred_df["score"].notna()]
        logger.info(f"去除NaN后样本数: {len(filtered)}")
        if len(filtered) == 0:
            logger.warning("所有预测分数均为NaN，建议检查模型输出和特征工程")
            return filtered

        # 2. 更严格的极端值处理（按分位数截断）
        lower = filtered["score"].quantile(0.01)
        upper = filtered["score"].quantile(0.99)
        before_clip = len(filtered)
        filtered = filtered[(filtered["score"] >= lower) & (filtered["score"] <= upper)]
        logger.info(f"极端值分位数截断后样本数: {len(filtered)} (去除{before_clip - len(filtered)})")
        if len(filtered) == 0:
            logger.warning("所有分数被极端值过滤，建议放宽过滤条件或检查模型输出")
            return filtered

        # 3. 标准化
        if "industry" in filtered.columns:
            filtered["score_norm"] = filtered.groupby(["date", "industry"])["score"].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
        else:
            std = filtered["score"].std()
            filtered["score_norm"] = (filtered["score"] - filtered["score"].mean()) / std if std > 0 else 0

        logger.info(f"过滤后样本数: {len(filtered)}")
        if len(filtered) == 0:
            logger.warning("过滤后无有效股票，建议检查模型和特征")
        else:
            logger.info(f"过滤后分数样例: {len(filtered)}")
        # 新增：输出score_norm分布统计
        logger.info(f"信号分布统计: {filtered['score_norm'].describe()}")
        return filtered
    
    def _generate_portfolio(self, pred_df: pd.DataFrame, top_k: int) -> Dict:
        """
        生成投资组合建议
        :param pred_df: 预测DataFrame
        :param top_k: 组合股票数量
        :return: 组合字典
        """
        top_stocks = pred_df.sort_values("score_norm", ascending=False).head(top_k)
        
        # 计算权重 (softmax归一化)
        scores = top_stocks["score_norm"].values
        weights = np.exp(scores) / np.sum(np.exp(scores))
        
        portfolio = {
            "date": pred_df["date"].max().strftime("%Y-%m-%d"),
            "stocks": [],
            "summary": {
                "num_stocks": len(top_stocks),
                "avg_score": top_stocks["score_norm"].mean(),
                "score_std": top_stocks["score_norm"].std()
            }
        }
        
        for i, (_, row) in enumerate(top_stocks.iterrows()):
            portfolio["stocks"].append({
                "symbol": row["instrument"],
                "score": float(row["score_norm"]),
                "weight": float(weights[i])
            })
        
        return portfolio
    
    def _generate_trading_signals(self, pred_df: pd.DataFrame, mode: str = "topk", top_k: int = 10, buy_quantile: float = 0.8, sell_quantile: float = 0.2) -> Dict:
        """
        生成交易信号
        :param pred_df: 预测DataFrame
        :param mode: 'topk' 或 'quantile'，控制信号生成方式
        :param top_k: TopK方式下买入/卖出股票数量
        :param buy_quantile: 分位数方式下买入阈值
        :param sell_quantile: 分位数方式下卖出阈值
        :return: 交易信号字典
        """
        day = pred_df["date"].max()
        day_df = pred_df[pred_df["date"] == day]
        if mode == "topk":
            top_buy = day_df.sort_values("score_norm", ascending=False).head(top_k)
            top_sell = day_df.sort_values("score_norm", ascending=True).head(top_k)
            signals = {
                "date": str(day),
                "buy": top_buy[["instrument", "score_norm"]].values.tolist(),
                "sell": top_sell[["instrument", "score_norm"]].values.tolist(),
                "hold": []
            }
        else:
            buy_thresh = day_df["score_norm"].quantile(buy_quantile)
            sell_thresh = day_df["score_norm"].quantile(sell_quantile)
            signals = {
                "date": str(day),
                "buy": day_df[day_df["score_norm"] > buy_thresh][["instrument", "score_norm"]].values.tolist(),
                "sell": day_df[day_df["score_norm"] < sell_thresh][["instrument", "score_norm"]].values.tolist(),
                "hold": day_df[(day_df["score_norm"] >= sell_thresh) & (day_df["score_norm"] <= buy_thresh)][["instrument", "score_norm"]].values.tolist()
            }
        return signals
    
    def backtest_strategy(
        self,
        instruments: str,
        start_date: str,
        end_date: str,
        topk: int = 10,
        n_drop: int = 2
    ) -> Dict:
        """
        回测策略表现，并生成每日持仓和买卖明细报告文件
        :param instruments: 股票池名称
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param topk: 每日选取的股票数量
        :param n_drop: 每日淘汰的股票数量
        :return: 回测结果字典，包含报告文件路径
        """
        if not self.load_latest_model(instruments):
            return {"error": "Model not available"}
        
        dataset = self.prepare_prediction_data(instruments, start_date, end_date)
        if dataset is None:
            return {"error": "Data preparation failed"}
        
        try:
            # 获取预测结果
            pred = self.current_model.predict(dataset, segment="test")
            logger.info(f"模型预测分数样例: {pred[:10]}")
            # 检查测试集内容
            test_data = dataset.prepare("test", col_set=["feature", "label"])
            logger.info(f"测试集样例: {test_data.head() if hasattr(test_data, 'head') else str(test_data)[:200]}")

            # ========== 统一过滤和标准化分数 ========== 
            pred_df = pd.DataFrame({
                "instrument": test_data.index.get_level_values("instrument"),
                "date": test_data.index.get_level_values("datetime"),
                "score": pred
            })
            filtered_df = self._filter_predictions(pred_df)
            if len(filtered_df) == 0:
                logger.warning("过滤后无有效信号，回测和信号输出均为空")
                return {
                    "performance": None,
                    "risk_analysis": None,
                    "positions": None,
                    "positions_file": None,
                    "trades_file": None
                }
            # 用过滤后的score_norm作为信号，统一传递给回测和信号/报告
            logger.info("已统一回测和信号/报告的分数源（score_norm，含极端值过滤和标准化）")
            # 构建信号Series，索引为(date, instrument)
            signal = filtered_df.set_index(["date", "instrument"])["score_norm"].sort_index()
            # ========== 新增：信号index检查和修正 ==========
            logger.info(f"signal index type: {type(signal.index)}, names: {signal.index.names}, sample: {list(signal.index)[:5]}")
            # 强制修正index类型和顺序
            try:
                signal.index = pd.MultiIndex.from_tuples([
                    (pd.to_datetime(d), str(i)) for d, i in signal.index
                ], names=["date", "instrument"])
                logger.info(f"修正后signal index type: {type(signal.index)}, names: {signal.index.names}, sample: {list(signal.index)[:5]}")
            except Exception as e:
                logger.warning(f"信号index修正失败: {e}")
            # 输出每日可选股票数样例
            logger.info(f'每日可选股票数样例: {filtered_df.groupby("date").size().head()}')

            # 输出每日score_norm分布样例
            try:
                logger.info(f'每日score_norm分布样例: {filtered_df.groupby("date")["score_norm"].describe().head()}')
            except Exception as e:
                logger.warning(f'输出score_norm分布失败: {e}')

            # 在信号生成后分析分布
            analyze_signal_distribution(filtered_df['score_norm'], logger)

            # 在策略生成前动态调整风控参数
            topk, n_drop = dynamic_topk_n_drop(filtered_df['score_norm'], topk, n_drop)

            # 创建策略（优先用TopkDropoutStrategy）
            strategy = TopkDropoutStrategy(
                signal=signal,
                topk=topk,
                n_drop=n_drop
            )
            
            # 执行回测
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
            # 输出positions内容样例
            try:
                if isinstance(positions, dict):
                    logger.info(f'positions dict样例: {list(positions.items())[:2]}')
                elif hasattr(positions, "head"):
                    logger.info(f'positions head样例: {positions.head()}')
                else:
                    logger.info(f'positions样例: {str(positions)[:200]}')
            except Exception as e:
                logger.warning(f'输出positions样例失败: {e}')
            # 如果positions依然为空或内容异常，尝试用TopkDropoutStrategy(n_drop=0)再回测
            if (isinstance(positions, dict) and all((not v or (isinstance(v, dict) and len(v) == 0)) for v in positions.values())) or (hasattr(positions, 'empty') and positions.empty):
                logger.warning('TopkDropoutStrategy未分配有效持仓，尝试用TopkDropoutStrategy(n_drop=0)...')
                strategy = TopkDropoutStrategy(signal=signal, topk=topk, n_drop=0)
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
                try:
                    if isinstance(positions, dict):
                        #logger.info(f'[TopkDropoutStrategy(n_drop=0)] positions dict样例: {list(positions.items())[:2]}')
                        logger.info(f'[TopkDropoutStrategy(n_drop=0)] positions dict')
                    elif hasattr(positions, "head"):
                        logger.info(f'[TopkDropoutStrategy(n_drop=0)] positions head样例: {positions.head()}')
                    else:
                        logger.info(f'[TopkDropoutStrategy(n_drop=0)] positions样例: {str(positions)[:200]}')
                except Exception as e:
                    logger.warning(f'[TopkDropoutStrategy(n_drop=0)] 输出positions样例失败: {e}')

            # 风险分析
            analysis = self._analyze_backtest(report, instruments)

            # ========== 新增：保存每日持仓明细（修正版） ==========
            reports_dir = self.model_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            pos_file = None
            holdings = []
            if isinstance(positions, dict) and len(positions) > 0:
                for date, v in positions.items():
                    #logger.info(f'[{date}] v type: {type(v)}, v keys: {list(v.keys()) if hasattr(v, "keys") else v}')
                   
                    if isinstance(v, dict):
                        # 如果v是字典，直接处理
                        logger.info(f"[{date}] v 已是字典，keys: {list(v.keys())}")
                    elif hasattr(v, 'to_dict'):
                        # 如果v有to_dict方法，尝试转换为字典
                        v_dict = v.to_dict()
                        if not v_dict:
                            v_dict = v.__dict__  # 如果to_dict返回None，尝试使用__dict__
                            logger.info(f"[{date}] v.to_dict()为None，已用__dict__，keys: {list(v_dict.keys())}")
                        else:
                            logger.info(f"[{date}] v 已转为 dict, keys: {list(v_dict.keys())}")
                        v = v_dict
                    elif hasattr(v, '__dict__'):
                        # 如果v有__dict__属性，直接使用
                        v = v.__dict__
                        #logger.info(f"[{date}] v 使用__dict__转换，keys: {list(v.keys())}")
                    else:
                        # 如果无法转换，记录详细信息并跳过
                        logger.warning(f"[{date}] v 无法转换为字典，类型: {type(v)}, 内容: {str(v)[:200]}")
                        continue

                    # 检查是否包含position字段
                    if 'position' in v and isinstance(v['position'], dict):
                        pos_dict = v['position']
                        for inst, detail in pos_dict.items():
                            if str(inst).lower() in ['cash', 'now_account_value']:
                                continue
                            if not isinstance(detail, dict):
                                continue
                            record = {
                                'date': date,
                                'instrument': inst,
                                'weight': detail.get('weight', None),
                                'amount': detail.get('amount', None),
                                'price': detail.get('price', None),
                                'count_day': detail.get('count_day', None)
                            }
                            holdings.append(record)
                    else:
                        logger.warning(f"[{date}] 未找到position字段，跳过")
                logger.info(f"holdings 列表长度: {len(holdings)}")
                if len(holdings) > 0:
                    holdings_df = pd.DataFrame(holdings)
                    logger.info(f"持仓明细DataFrame样例:\n{holdings_df.head()}")
                    logger.info(f"持仓明细DataFrame总行数: {len(holdings_df)}")
                    pos_file = reports_dir / f"positions_{instruments}_{start_date}_{end_date}.csv"
                    logger.info(f"写入路径存在: {reports_dir.exists()}")
                    try:
                        holdings_df.to_csv(pos_file, index=False)
                        logger.info(f"已生成标准化持仓明细文件: {pos_file}")
                    except Exception as e:
                        logger.error(f"写入持仓明细CSV失败: {e}")
                else:
                    logger.warning("未提取到有效持仓明细，未生成持仓报告文件")
            else:
                logger.warning("positions为空或格式异常，未生成持仓报告文件")
                pos_file = None

            # ========== 新增：生成每日买卖明细（修正版） ==========
            trades_file = None
            trades = []
            if len(holdings) > 0:
                holdings_df = pd.DataFrame(holdings)
                holdings_df['date'] = pd.to_datetime(holdings_df['date'])
                holdings_df = holdings_df.sort_values(['date', 'instrument'])
                grouped = holdings_df.groupby('date')['instrument'].apply(set)
                prev_set = set()
                for d, cur_set in grouped.items():
                    buy = list(cur_set - prev_set)
                    sell = list(prev_set - cur_set)
                    trades.append({
                        'date': d,
                        'buy': ','.join(buy),
                        'sell': ','.join(sell),
                        'hold': ','.join(cur_set)
                    })
                    prev_set = cur_set
                if len(trades) > 0:
                    trades_df = pd.DataFrame(trades)
                    trades_file = reports_dir / f"trades_{instruments}_{start_date}_{end_date}.csv"
                    trades_df.to_csv(trades_file, index=False)
                    logger.info(f"已生成买卖明细文件: {trades_file}")
                else:
                    logger.warning("买卖明细为空，未生成买卖报告文件")
            else:
                logger.warning("无有效持仓，未生成买卖报告文件")
                trades_file = None

            # 在持仓明细生成后分析持仓结构
            if len(holdings) > 0:
                holdings_df = pd.DataFrame(holdings)
                analyze_holdings(holdings_df, logger)

            return {
                "performance": report,
                "risk_analysis": analysis,
                "positions": positions,
                "positions_file": str(pos_file) if pos_file else None,
                "trades_file": str(trades_file) if trades_file else None
            }
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_backtest(self, report: pd.DataFrame, instruments: str) -> Dict:
        """
        分析回测结果
        :param report: 回测报告DataFrame
        :param instruments: 股票池名称
        :return: 风险分析字典
        """
        returns = pd.to_numeric(report['return'], errors='coerce').dropna()
        
        # 动态选择基准
        benchmark_map = {
            "csi300": "SH000300",
            "csi500": "SH000905",
            "csi800": "SH000906",
            "all": "SH000985"
        }
        benchmark = benchmark_map.get(instruments, "SH000985")
        
        analysis = {
            "Annualized Return": float(returns.mean() * 252),
            "Annualized Volatility": float(returns.std() * np.sqrt(252)),
            "Sharpe Ratio": float(returns.mean() / returns.std() * np.sqrt(252)),
            #"Information Ratio": returns.mean() / returns.std()  * np.sqrt(252) if returns.std()  != 0 else 0,
            "Max Drawdown": float((returns.cumsum() - returns.cumsum().cummax()).min()),
            "Win Rate": float((returns > 0).mean()),
            "Tail Risk (VaR 5%)": float(returns.quantile(0.05)),
            "Benchmark": benchmark
        }
        
        return analysis

# ========== 辅助分析函数 ==========
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

# 使用示例
if __name__ == "__main__":
    # 初始化QLib (必须在使用前调用)
    qlib.init(provider_uri="./qlib_data/cn_data", region=REG_CN)
    
    # 创建预测器实例
    predictor = StockPredictor()
    
    # 示例1: 获取CSI300的最新预测
    # csi300_pred = predictor.predict_stocks(
    #     instruments="csi800",
    #     prediction_date="2025-05-29",
    #     output_format="portfolio"
    # )
    # print("CSI300 Portfolio Recommendation:")
    # print("\nCSI300 predict Results:")
    # if "error" in csi300_pred:
    #     print(f"Backtest failed: {csi300_pred['error']}")
    # elif "risk_analysis" in csi300_pred:
    #     print(csi300_pred["risk_analysis"])
    # else:
    #     print("Risk analysis not available")
    #     print(f"Available keys: {list(csi300_pred.keys())}")

    
    # 示例2: 回测CSI500策略
    today = datetime.now().strftime("%Y-%m-%d")
    start_test = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    backtest_start_date = start_test
    backtest_end_date = today
    # all_backtest = predictor.backtest_strategy(
    #     instruments="all",
    #     start_date=backtest_start_date,
    #     end_date=backtest_end_date
    # )
    csi800_backtest = predictor.backtest_strategy(
        instruments="csi800",
        start_date=backtest_start_date,
        end_date=backtest_end_date,
        topk=5
    )
    # csi500_backtest = predictor.backtest_strategy(
    #     instruments="csi500",
    #     start_date=backtest_start_date,
    #     end_date=backtest_end_date
    # )
    # csi300_backtest = predictor.backtest_strategy(
    #     instruments="csi300",
    #     start_date=backtest_start_date,
    #     end_date=backtest_end_date
    # )
    # print("\nall Backtest Results:")
    # if "error" in all_backtest:
    #     print(f"Backtest failed: {all_backtest['error']}")
    # elif "risk_analysis" in all_backtest:
    #     print(all_backtest["risk_analysis"])
    # else:
    #     print("Risk analysis not available")
    #     print(f"Available keys: {list(all_backtest.keys())}")
    
    
    print("\nCSI800 Backtest Results:")
    if "error" in csi800_backtest:
        print(f"Backtest failed: {csi800_backtest['error']}")
    elif "risk_analysis" in csi800_backtest:
        print(csi800_backtest["risk_analysis"])
    else:
        print("Risk analysis not available")
        print(f"Available keys: {list(csi800_backtest.keys())}")
    
    
    # print("\nCSI500 Backtest Results:")
    # if "error" in csi500_backtest:
    #     print(f"Backtest failed: {csi500_backtest['error']}")
    # elif "risk_analysis" in csi500_backtest:
    #     print(csi500_backtest["risk_analysis"])
    # else:
    #     print("Risk analysis not available")
    #     print(f"Available keys: {list(csi500_backtest.keys())}")
    
   
    # print("\nCSI300 Backtest Results:")
    # if "error" in csi300_backtest:
    #     print(f"Backtest failed: {csi300_backtest['error']}")
    # elif "risk_analysis" in csi300_backtest:
    #     print(csi300_backtest["risk_analysis"])
    # else:
    #     print("Risk analysis not available")
    #     print(f"Available keys: {list(csi300_backtest.keys())}")
    
    #示例3: 生成交易信号
    # signals = predictor.predict_stocks(
    #     instruments="csi800",
    #     output_format="signals",
    #     prediction_date="2025-06-10",
    #     top_k=5
    # )
    # try:
    #     logger.info(f"\nTrading Signals:\n{json.dumps(signals, indent=2, ensure_ascii=False)}")
    # except TypeError as e:
    #     logger.warning(f"信号序列化为JSON失败: {e}")
    #     logger.info(f"signals内容: {signals}")

    # if "error" in signals:
    #     print(f"Signal generation failed: {signals['error']}")
    #     print(f"Buy List: {len(signals.get('buy', []))} stocks")
    #     print(f"Sell List: {len(signals.get('sell', []))} stocks")
    # else:
    #     print(f"Buy List: {len(signals.get('buy', []))} stocks")
    #     print(f"Sell List: {len(signals.get('sell', []))} stocks")