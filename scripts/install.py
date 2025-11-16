#!/usr/bin/env python3
"""
Installation script for Voice Assistant.
Handles post-dependency setup including model downloads and configuration.
"""

import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai.model_manager import get_model_manager, is_default_model_available

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Installer:
    """Main installer class for Voice Assistant setup."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.models_dir = Path(self.config.get('models_dir', 'models'))
        self.config_dir = Path(self.config.get('config_dir', 'config'))
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.voices_dir = Path(self.config.get('voices_dir', 'voices'))
        self.logs_dir = Path(self.config.get('logs_dir', 'logs'))
        
        # Installation steps
        self.steps = [
            ('create_directories', 'Creating necessary directories'),
            ('download_default_model', 'Downloading default AI model'),
            ('setup_configuration', 'Setting up configuration files'),
            ('verify_installation', 'Verifying installation')
        ]
    
    async def install(self, skip_steps: Optional[List[str]] = None) -> bool:
        """Run the complete installation process."""
        skip_steps = skip_steps or []
        
        logger.info("Starting Voice Assistant installation...")
        logger.info(f"Installation directory: {Path.cwd()}")
        
        success_count = 0
        total_steps = len(self.steps)
        
        for step_id, step_name in self.steps:
            if step_id in skip_steps:
                logger.info(f"Skipping step: {step_name}")
                continue
                
            logger.info(f"Step {success_count + 1}/{total_steps}: {step_name}")
            
            try:
                success = await getattr(self, step_id)()
                if success:
                    logger.info(f"✓ {step_name} completed successfully")
                    success_count += 1
                else:
                    logger.error(f"✗ {step_name} failed")
                    return False
                    
            except Exception as e:
                logger.error(f"✗ {step_name} failed with error: {e}")
                return False
        
        logger.info(f"Installation completed successfully! ({success_count}/{total_steps} steps)")
        return True
    
    async def create_directories(self) -> bool:
        """Create necessary directories."""
        try:
            directories = [
                self.models_dir,
                self.config_dir,
                self.data_dir,
                self.voices_dir,
                self.logs_dir
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False
    
    async def download_default_model(self) -> bool:
        """Download the default AI model from models.json."""
        try:
            # Initialize model manager
            manager = get_model_manager()
            
            # Get default model info
            default_model = manager.get_default_model()
            if not default_model:
                logger.error("✗ No default model configured in models.json")
                return False
            
            logger.info(f"Setting up default model: {default_model.name}")
            logger.info(f"  Model ID: {default_model.id}")
            logger.info(f"  Repository: {default_model.repo_id}")
            logger.info(f"  Quantization: {default_model.quantization}")
            logger.info(f"  Parameters: {default_model.parameters}")
            logger.info(f"  File patterns: {default_model.file_patterns}")
            
            # Check if default model is already available
            if is_default_model_available():
                logger.info("✓ Default model is already downloaded")
                
                # Print model stats
                stats = manager.get_model_stats()
                logger.info(f"Model statistics:")
                logger.info(f"  Total models: {stats['total_models']}")
                logger.info(f"  Downloaded models: {stats['downloaded_models']}")
                logger.info(f"  Total size: {stats['total_size_gb']:.2f} GB")
                
                return True
            
            # Download the default model
            logger.info(f"Starting download of {default_model.name}...")
            logger.info(f"This may take several minutes depending on your connection speed...")
            
            success = await manager.download_default_model()
            
            if success:
                logger.info(f"✓ Default model downloaded successfully: {default_model.name}")
                
                # Print model stats
                stats = manager.get_model_stats()
                logger.info(f"Model statistics:")
                logger.info(f"  Total models: {stats['total_models']}")
                logger.info(f"  Downloaded models: {stats['downloaded_models']}")
                logger.info(f"  Total size: {stats['total_size_gb']:.2f} GB")
                logger.info(f"  Default model: {stats['default_model']}")
                
                return True
            else:
                logger.error(f"✗ Failed to download default model: {default_model.name}")
                logger.warning("You can manually download the model later using:")
                logger.warning(f"  python -c \"from ai.model_manager import get_model_manager; import asyncio; asyncio.run(get_model_manager().download_model('{default_model.id}'))\"")
                return False
                
        except Exception as e:
            logger.error(f"Failed to download default model: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    async def setup_configuration(self) -> bool:
        """Set up configuration files."""
        try:
            import shutil
            
            # Setup .env file from env.example if it doesn't exist
            env_file = Path('.env')
            env_example = Path('env.example')
            
            if not env_file.exists() and env_example.exists():
                logger.info("Creating .env file from env.example...")
                shutil.copy(env_example, env_file)
                logger.info(f"Created .env file: {env_file}")
            elif env_file.exists():
                logger.info(".env file already exists, skipping copy from env.example")
            elif not env_example.exists():
                logger.warning("env.example not found, skipping .env setup")
            
            # Create default configuration if it doesn't exist
            config_file = self.config_dir / "assistant_config.yaml"
            
            if not config_file.exists():
                logger.info("Creating default configuration file...")
                
                default_config = {
                    'name': 'Voice Assistant',
                    'version': '1.0.0',
                    'debug': False,
                    'log_level': 'INFO',
                    'llm': {
                        'provider_type': 'local',
                        'model': 'llama32-1b-q6k',
                        'model_id': 'llama32-1b-q6k',
                        'context_window': 8192,
                        'temperature': 0.7,
                        'max_tokens': 512,
                        'n_gpu_layers': 0,
                        'n_threads': 4,
                        'use_mmap': True,
                        'use_mlock': False,
                        'lazy_loading': False,
                        'cache_size': 1
                    },
                    'audio': {
                        'sample_rate': 16000,
                        'channels': 1,
                        'wake_word_threshold': 0.5,
                        'vad_threshold': 0.3
                    },
                    'display': {
                        'eyes_display': True,
                        'mouth_display': True,
                        'resolution': [800, 600],
                        'fps': 30,
                        'static_mode': False
                    },
                    'camera': {
                        'enabled': False,
                        'resolution': [640, 480],
                        'fps': 15
                    }
                }
                
                import yaml
                with open(config_file, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
                
                logger.info(f"Created configuration file: {config_file}")
            else:
                logger.info(f"Configuration file already exists: {config_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup configuration: {e}")
            return False
    
    async def verify_installation(self) -> bool:
        """Verify that the installation was successful."""
        try:
            logger.info("Verifying installation...")
            
            # Check directories
            required_dirs = [self.models_dir, self.config_dir, self.data_dir, self.voices_dir, self.logs_dir]
            for directory in required_dirs:
                if not directory.exists():
                    logger.error(f"Required directory missing: {directory}")
                    return False
            
            # Check model availability
            manager = get_model_manager()
            if not is_default_model_available():
                logger.error("Default model not available")
                return False
            
            # Check configuration
            config_file = self.config_dir / "assistant_config.yaml"
            if not config_file.exists():
                logger.error("Configuration file missing")
                return False
            
            logger.info("✓ Installation verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Installation verification failed: {e}")
            return False
    
    def print_installation_summary(self):
        """Print a summary of the installation."""
        print("\n" + "="*60)
        print("VOICE ASSISTANT INSTALLATION SUMMARY")
        print("="*60)
        
        # Print directory structure
        print("\nDirectory Structure:")
        directories = [
            ("Models", self.models_dir),
            ("Config", self.config_dir),
            ("Data", self.data_dir),
            ("Voices", self.voices_dir),
            ("Logs", self.logs_dir)
        ]
        
        for name, path in directories:
            status = "✓" if path.exists() else "✗"
            print(f"  {status} {name}: {path}")
        
        # Print model information
        try:
            manager = get_model_manager()
            stats = manager.get_model_stats()
            print(f"\nModel Information:")
            print(f"  Total models: {stats['total_models']}")
            print(f"  Downloaded models: {stats['downloaded_models']}")
            print(f"  Total size: {stats['total_size_gb']:.2f} GB")
            print(f"  Default model: {stats['default_model']}")
        except Exception as e:
            print(f"  Error getting model info: {e}")
        
        print("\n" + "="*60)
        print("INSTALLATION COMPLETE")
        print("="*60)
        print("The Voice Assistant is now ready to use!")
        print("\nTo start the application:")
        print("1. Run: python src/main.py")
        print("2. Or use Docker: docker-compose up")
        print("\nFor more information, see the documentation files.")


async def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="Voice Assistant Installation Script")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--voices-dir", default="voices", help="Voices directory")
    parser.add_argument("--logs-dir", default="logs", help="Logs directory")
    parser.add_argument("--skip", nargs="+", help="Skip installation steps", 
                       choices=["create_directories", "download_default_model", "setup_configuration", "verify_installation"])
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create installer configuration
    config = {
        'models_dir': args.models_dir,
        'config_dir': args.config_dir,
        'data_dir': args.data_dir,
        'voices_dir': args.voices_dir,
        'logs_dir': args.logs_dir
    }
    
    # Run installation
    installer = Installer(config)
    success = await installer.install(skip_steps=args.skip)
    
    if success:
        installer.print_installation_summary()
        return 0
    else:
        logger.error("Installation failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
