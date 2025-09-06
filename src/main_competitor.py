# src/main_competitor.py
"""
Command-line interface for the competitor analysis system
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from competitor.analyzer import CompetitorAnalyzer, run_competitor_analysis
from competitor.config import CompetitorConfig

def setup_logging(verbose: bool = False, debug: bool = False):
    """Setup logging configuration"""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Competitor Analysis System for ecommerce search companies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all configured competitors
  python src/main_competitor.py

  # Analyze specific competitors
  python src/main_competitor.py --competitors "Algolia" "Constructor.io"

  # Generate PDF report with comprehensive analysis
  python src/main_competitor.py --format pdf --depth comprehensive

  # Quick analysis with JSON output
  python src/main_competitor.py --format json --depth basic

  # Validate configuration
  python src/main_competitor.py --validate-config

  # List configured competitors
  python src/main_competitor.py --list-competitors
        """
    )
    
    # Main analysis options
    parser.add_argument(
        '--competitors',
        nargs='+',
        help='Specific competitors to analyze (default: all configured)'
    )
    
    parser.add_argument(
        '--format',
        choices=['pdf', 'docx', 'json'],
        default='pdf',
        help='Output format for reports (default: pdf)'
    )
    
    parser.add_argument(
        '--depth',
        choices=['basic', 'standard', 'comprehensive'],
        default='standard',
        help='Analysis depth level (default: standard)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for reports (default: from config)'
    )
    
    # Configuration options
    parser.add_argument(
        '--config',
        default='competitor_config.yaml',
        help='Path to configuration file (default: competitor_config.yaml)'
    )
    
    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration file and exit'
    )
    
    parser.add_argument(
        '--update-config',
        action='store_true',
        help='Update configuration file with latest schema'
    )
    
    parser.add_argument(
        '--list-competitors',
        action='store_true',
        help='List configured competitors and exit'
    )
    
    # Execution options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be analyzed without running analysis'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser

def validate_configuration(config_path: str) -> bool:
    """Validate configuration file"""
    try:
        config = CompetitorConfig(config_path)
        
        print("✓ Configuration file loaded successfully")
        print(f"✓ Found {len(config.competitors)} configured competitors")
        
        # Check required environment variables
        import os
        required_apis = []
        optional_apis = []
        
        # Check LLM APIs
        if not any(os.getenv(key) for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'PERPLEXITY_API_KEY']):
            required_apis.append('At least one LLM API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or PERPLEXITY_API_KEY)')
        
        # Check optional APIs
        optional_keys = [
            'CRUNCHBASE_API_KEY', 'GITHUB_TOKEN', 'NEWS_API_KEY',
            'TWITTER_BEARER_TOKEN', 'LINKEDIN_ACCESS_TOKEN'
        ]
        
        for key in optional_keys:
            if not os.getenv(key):
                optional_apis.append(key)
        
        # Report findings
        if required_apis:
            print("\n❌ Missing required environment variables:")
            for api in required_apis:
                print(f"  - {api}")
            return False
        else:
            print("✓ Required environment variables configured")
        
        if optional_apis:
            print(f"\n⚠️  Optional APIs not configured ({len(optional_apis)} of {len(optional_keys)}):")
            for api in optional_apis[:3]:  # Show first 3
                print(f"  - {api}")
            if len(optional_apis) > 3:
                print(f"  - ... and {len(optional_apis) - 3} more")
            print("  (Enhanced features will be limited)")
        else:
            print("✓ All optional APIs configured")
        
        # Validate output directory
        output_dir = Path(config.output_config['output_dir'])
        if output_dir.exists():
            print(f"✓ Output directory exists: {output_dir}")
        else:
            print(f"ℹ️  Output directory will be created: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def list_competitors(config_path: str) -> None:
    """List configured competitors"""
    try:
        config = CompetitorConfig(config_path)
        
        print(f"Configured Competitors ({len(config.competitors)}):")
        print("=" * 50)
        
        for i, competitor in enumerate(config.competitors, 1):
            name = competitor['name']
            website = competitor['website']
            priority = competitor.get('priority', 'medium')
            threat = competitor.get('competitive_threat', 'medium')
            last_analyzed = competitor.get('last_analyzed', 'Never')
            
            print(f"{i}. {name}")
            print(f"   Website: {website}")
            print(f"   Priority: {priority.title()}")
            print(f"   Threat Level: {threat.title()}")
            print(f"   Last Analyzed: {last_analyzed}")
            
            if 'market_segment' in competitor:
                segments = ', '.join(competitor['market_segment'])
                print(f"   Market Segments: {segments}")
            
            print()
        
        # Show summary by priority
        priority_counts = {}
        threat_counts = {}
        
        for competitor in config.competitors:
            priority = competitor.get('priority', 'medium')
            threat = competitor.get('competitive_threat', 'medium')
            
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
            threat_counts[threat] = threat_counts.get(threat, 0) + 1
        
        print("Summary:")
        print(f"  Priority: {dict(priority_counts)}")
        print(f"  Threat Levels: {dict(threat_counts)}")
        
    except Exception as e:
        print(f"❌ Failed to list competitors: {e}")

async def run_analysis(args) -> None:
    """Run the competitor analysis"""
    try:
        # Setup analyzer
        analyzer = CompetitorAnalyzer(args.config)
        
        # Override config with CLI args
        analyzer.config.override_from_cli_args(args)
        
        if args.dry_run:
            # Show what would be analyzed
            competitors_to_analyze = analyzer.config.competitors
            if args.competitors:
                competitors_to_analyze = [
                    comp for comp in analyzer.config.competitors
                    if comp['name'].lower() in [c.lower() for c in args.competitors]
                ]
            
            print("Dry Run - Would analyze:")
            for comp in competitors_to_analyze:
                print(f"  - {comp['name']} ({comp['website']})")
            
            print(f"\nOutput format: {args.format}")
            print(f"Analysis depth: {args.depth}")
            print(f"Output directory: {analyzer.config.output_config['output_dir']}")
            return
        
        # Run actual analysis
        print("🔍 Starting competitor analysis...")
        
        intelligence = await analyzer.analyze_all_competitors(args.competitors)
        
        if not intelligence.profiles:
            print("❌ No competitor data collected. Check configuration and API keys.")
            return
        
        print(f"✓ Analyzed {len(intelligence.profiles)} competitors")
        
        # Update threat levels
        await analyzer.update_competitor_threat_levels(intelligence)
        
        # Generate reports
        print(f"📊 Generating {args.format.upper()} report...")
        
        if args.format == 'pdf':
            report_files = [await analyzer.report_generator.generate_pdf_report(intelligence)]
        elif args.format == 'docx':
            report_files = [await analyzer.report_generator.generate_docx_report(intelligence)]
        elif args.format == 'json':
            report_files = [await analyzer.report_generator.generate_json_report(intelligence)]
        else:
            report_files = await analyzer.generate_reports(intelligence)
        
        # Show results
        print("\n🎉 Analysis completed successfully!")
        print(f"📁 Generated {len(report_files)} report file(s):")
        
        for file_path in report_files:
            file_size = Path(file_path).stat().st_size / 1024  # KB
            print(f"  - {file_path} ({file_size:.1f} KB)")
        
        # Show executive summary
        exec_summary = await analyzer.generate_executive_summary(intelligence)
        print("\n📋 Executive Summary:")
        print("-" * 50)
        print(exec_summary[:500] + "..." if len(exec_summary) > 500 else exec_summary)
        
        # Show threat summary
        threats = await analyzer.get_competitive_threats_by_level(intelligence)
        high_threats = threats.get('high', []) + threats.get('critical', [])
        
        if high_threats:
            print(f"\n⚠️  High Priority Threats ({len(high_threats)}):")
            for threat in high_threats:
                print(f"  - {threat}")
        
        # Cleanup
        await analyzer.cleanup()
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.debug)
    
    # Handle configuration commands
    if args.validate_config:
        success = validate_configuration(args.config)
        sys.exit(0 if success else 1)
    
    if args.list_competitors:
        list_competitors(args.config)
        sys.exit(0)
    
    if args.update_config:
        try:
            config = CompetitorConfig(args.config)
            config.save_config()
            print(f"✓ Configuration updated: {args.config}")
        except Exception as e:
            print(f"❌ Failed to update configuration: {e}")
            sys.exit(1)
        sys.exit(0)
    
    # Validate configuration before analysis
    if not validate_configuration(args.config):
        print("\n❌ Configuration validation failed. Fix issues before running analysis.")
        sys.exit(1)
    
    # Run analysis
    try:
        asyncio.run(run_analysis(args))
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()