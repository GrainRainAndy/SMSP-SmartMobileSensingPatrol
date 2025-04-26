from datetime import datetime

def get_timestamp():
    """返回当前时间字符串，格式：YYYYMMDD_HHMMSS_mmm"""
    now = datetime.now()
    return now.strftime('%Y%m%d_%H%M%S_') + f"{int(now.microsecond / 1000):03d}"