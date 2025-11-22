#!/usr/bin/env python3
"""
Main entry point for the Cluster Voice Assistant application.

This launcher supports two modes:
1. Event-driven orchestrator (voice_assistant) - coordinates STT, LLM, TTS services
2. Cluster mode - the full voice assistant with display, animation, etc.

Usage:
    # Run the event-driven voice assistant
    python main.py --mode orchestrator

    # Run the full Cluster application
    python main.py --mode cluster

    # Run orchestrator by default
    python main.py
"""
import os
import sys
import logging
import argparse
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)


def run_orchestrator():
    """Run the event-driven voice assistant orchestrator."""
    from src.orchestrator.voice_assistant import main as orchestrator_main
    orchestrator_main()


def run_cluster():
    """Run the full Cluster application."""
    from src.main import main as cluster_main
    asyncio.run(cluster_main())


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Cluster Voice Assistant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  orchestrator  - Event-driven voice assistant with STT, LLM, TTS services
  cluster       - Full Cluster application with display and animations

Examples:
  python main.py --mode orchestrator
  python main.py --mode cluster
  python main.py  # defaults to orchestrator
"""
    )
    parser.add_argument(
        '--mode',
        choices=['orchestrator', 'cluster'],
        default='orchestrator',
        help='Application mode (default: orchestrator)'
    )

    args = parser.parse_args()

    logger.info(f"Starting Cluster Voice Assistant in {args.mode} mode...")

    try:
        if args.mode == 'orchestrator':
            run_orchestrator()
        else:
            run_cluster()

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
