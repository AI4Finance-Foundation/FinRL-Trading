#!/usr/bin/env python3
"""
FinRL Trading Platform - Main Entry Point
========================================

Command-line interface for the FinRL trading platform.
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
            try:
                from backtest.backtest_engine import main as backtest_main
            except ImportError:
                logger.error(
                    "backtest_engine has no CLI main(); use "
                    "'./deploy.sh --mode backtest' or "
                    "'python src/strategies/run_adaptive_rotation_strategy.py --backtest'."
                )
                sys.exit(2)
            backtest_main()

        elif args.command == 'trade':
            try:
                from trading.trade_executor import main as trade_main
            except ImportError:
                logger.error(
                    "trade_executor has no CLI main(); see the live-trading "
                    "examples in the repository instead."
                )
                sys.exit(2)
            trade_main()

        elif args.command == 'data':
            try:
                from data.data_processor import main as data_main
            except ImportError:
                logger.error(
                    "data_processor has no CLI main(); use the scripts under "
                    "'src/data/' or 'deploy.sh download-data' instead."
                )
                sys.exit(2)
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


def _run_command(command: str):
    """Invoke a subcommand from a console-script entry point."""
    sys.argv = [sys.argv[0], command] + sys.argv[1:]
    return main()


def main_dashboard():
    """Console entry point for the Streamlit dashboard."""
    return _run_command("dashboard")


def main_backtest():
    """Console entry point for the backtest CLI."""
    return _run_command("backtest")


def main_trade():
    """Console entry point for trade execution."""
    return _run_command("trade")


if __name__ == "__main__":
    main()
