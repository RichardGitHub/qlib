import time
import psutil
import functools
from typing import Callable, Any

def monitor_performance(func: Callable) -> Callable:
    """性能监控装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            logger.info(f"{func.__name__} 执行时间: {end_time - start_time:.2f}秒")
            logger.info(f"{func.__name__} 内存使用: {end_memory - start_memory:.2f}MB")
            
            return result
        except Exception as e:
            logger.error(f"{func.__name__} 执行失败: {e}")
            raise
    
    return wrapper

@monitor_performance
def train_model_with_monitoring(model, dataset):
    """带性能监控的模型训练"""
    return model.fit(dataset)