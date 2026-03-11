"""全局配置 - 从 .env 文件或环境变量读取"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(Path(__file__).parent / ".env")

# DeepSeek API 配置
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn/v1")
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V3")

# 路径配置
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
PROMPTS_DIR = BASE_DIR / "prompts"

# 确保输出目录存在
OUTPUTS_DIR.mkdir(exist_ok=True)

# 分析配置
MAX_REPORTS_KEEP = 20          # 最多保留的报告数量
RF_N_ESTIMATORS = 100          # 随机森林树的数量
RF_RANDOM_STATE = 42
MIN_DATA_ROWS = 10             # 最少有效行数
MISSING_RATE_THRESHOLD = 0.5   # 缺失率阈值
COLLINEARITY_THRESHOLD = 0.8   # 多重共线性警告阈值
