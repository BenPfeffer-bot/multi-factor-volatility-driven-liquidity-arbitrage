from setuptools import setup, find_packages

setup(
    name="Novel Arbitrage Strategy: Multi-Factor Volatility-Driven Liquidity Arbitrage (MVLA)",
    version="0.1.0",
    description="Algotrading volatility strategy with regime shift detection on DOWJONES TITANS 50 + options data",
    author="Ben Pfeffer",
    author_email="benpfefferpro@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "torch>=1.9.0",
        "yfinance>=0.1.63",
        "websocket-client>=1.2.1",
        "requests>=2.26.0",
        "python-dotenv>=0.19.0",
        "xgboost>=1.4.2",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "jupyter>=1.0.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
