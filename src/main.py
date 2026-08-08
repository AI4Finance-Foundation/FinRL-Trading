#!/usr/bin/env python3
"""
FinRL Trading Platform - Main Entry Point
========================================

Command-line interface for the FinRL trading platform.
Supports MT5 Enterprise System for full lifecycle management.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_config
from utils.logging_utils import setup_logging


def setup_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="FinRL Trading Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py dashboard    # Start web dashboard
  python src/main.py backtest     # Run backtest
  python src/main.py trade        # Execute live trading
  python src/main.py trade --enterprise --strategy equal-weight --mode dry-run --symbols EURUSD GBPUSD
  python src/main.py trade --all-in-one --strategy trend --mode dry-run --symbols EURUSD GBPUSD
  python src/main.py trade --all-in-one --strategy pairs --mode dry-run
  python src/main.py data         # Manage data
        """
    )

    parser.add_argument(
        'command',
        choices=['dashboard', 'backtest', 'trade', 'data', 'config'],
        help='Command to execute'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    # Enterprise / MT5 options
    parser.add_argument(
        '--enterprise', '-e',
        action='store_true',
        help='Use MT5EnterpriseSystem for full lifecycle management'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='equal-weight',
        choices=['ml', 'rl', 'adaptive', 'signal', 'equal-weight', 'momentum', 'bucket', 'trend', 'reversion', 'pairs'],
        help='Trading strategy (default: equal-weight, requires --enterprise)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='dry-run',
        choices=['live', 'paper', 'dry-run'],
        help='Execution mode (default: dry-run, requires --enterprise)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=None,
        help='Symbols to trade (space-separated, requires --enterprise)'
    )
    parser.add_argument(
        '--account',
        type=str,
        default=None,
        help='Account name for multi-account MT5 (requires --enterprise)'
    )
    parser.add_argument(
        '--schedule',
        type=str,
        default=None,
        choices=['daily', 'hourly', 'once'],
        help='Run on a schedule (requires --enterprise)'
    )
    parser.add_argument(
        '--all-in-one',
        action='store_true',
        help='Use all_in_one pipeline that wires all 10 strategies to MT5'
    )

    return parser


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    # Setup configuration
    config = get_config()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else getattr(logging, config.logging.level)
    setup_logging(level=log_level)

    logger = logging.getLogger(__name__)
    logger.info(f"FinRL Trading Platform v{config.version}")
    logger.info(f"Environment: {config.environment}")

    try:
        if args.command == 'dashboard':
            from web.app import main as dashboard_main
            dashboard_main()

        elif args.command == 'backtest':
            from backtest.backtest_engine import main as backtest_main
            backtest_main()

        elif args.command == 'trade':
            if args.all_in_one:
                # Run with all_in_one pipeline
                from trading.all_in_one import run_mt5_all_in_one
                run_mt5_all_in_one(
                    strategy_name=args.strategy,
                    mode=args.mode,
                    symbols=args.symbols,
                    account_name=args.account,
                    verbose=args.verbose,
                    schedule=args.schedule,
                )
            elif args.enterprise:
                # Run with MT5EnterpriseSystem
                from run_trading import run_enterprise
                run_enterprise(
                    strategy_name=args.strategy,
                    mode=args.mode,
                    symbols=args.symbols,
                    account_name=args.account,
                    verbose=args.verbose,
                    schedule=args.schedule,
                )
            else:
                from trading.trade_executor import main as trade_main
                trade_main()

        elif args.command == 'data':
            from data.data_processor import main as data_main
            data_main()

        elif args.command == 'config':
            print("Current Configuration:")
            print(f"  Environment: {config.environment}")
            print(f"  Alpaca Configured: {bool(config.alpaca.api_key)}")
            print(f"  WRDS Configured: {bool(config.wrds.username)}")
            print(f"  Data Directory: {config.get_data_dir()}")
            print(f"  Log Level: {config.logging.level}")

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()