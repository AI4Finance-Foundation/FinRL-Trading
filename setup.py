"""
FinRL Trading Platform Setup
===========================

Setup script for the FinRL quantitative trading platform.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read core requirements
def read_requirements():
    requirements = []
    with open("requirements.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("tensorflow") and not line.startswith("torch"):
                requirements.append(line)
    return requirements

setup(
    name="finrl_trading",
    version="2.0.2",
    author="FinRL LLC",
    author_email="contact@finrl.ai",
    description="A modern quantitative trading platform with ML strategies and live trading capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AI4Finance-Foundation/FinRL-Trading",
    project_urls={
        "Bug Reports": "https://github.com/AI4Finance-Foundation/FinRL-Trading/issues",
        "Source": "https://github.com/AI4Finance-Foundation/FinRL-Trading",
        "Documentation": "https://finrl.readthedocs.io/",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Framework :: AsyncIO",
        "Typing :: Typed",
    ],
    keywords="quantitative trading machine learning reinforcement learning alpaca finance",
    python_requires=">=3.11",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "web": [
            "streamlit>=1.28.0",
            "plotly>=5.15.0",
            "seaborn>=0.12.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "gymnasium>=0.29.0",
            "stable-baselines3>=2.1.0",
            "xgboost>=2.0.0",
            "lightgbm>=4.1.0",
        ],
        "database": [
            "psycopg2-binary>=2.9.0",
            "sqlalchemy>=2.0.0",
        ],
        "all": [
            "torch>=2.0.0",
            "gymnasium>=0.29.0",
            "stable-baselines3>=2.1.0",
            "xgboost>=2.0.0",
            "lightgbm>=4.1.0",
            "psycopg2-binary>=2.9.0",
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "finrl=main:main",
            "finrl-dashboard=main:main_dashboard",
            "finrl-backtest=main:main_backtest",
            "finrl-trade=main:main_trade",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    data_files=[
        ("config", ["requirements.txt"]),
    ],
)
