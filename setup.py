#!/usr/bin/env python3
"""
Setup script for Competitor Analysis System
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the README file for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Comprehensive competitor analysis system for ecommerce search companies"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "aiohttp>=3.9.0",
        "beautifulsoup4>=4.12.0",
        "pyyaml>=6.0",
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        "python-dateutil>=2.8.0",
        "reportlab>=4.0.0",
        "python-docx>=1.1.0"
    ]

setup(
    name="competitor-analysis",
    version="1.0.0",
    author="Your Organization",
    author_email="your-email@company.com",
    description="Comprehensive competitor analysis system for ecommerce search companies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/competitor-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0"
        ],
        "enhanced": [
            "crunchbase-api>=0.1.0",
            "PyGithub>=1.59.0",
            "python-twitter>=3.5"
        ]
    },
    entry_points={
        "console_scripts": [
            "competitor-analysis=main_competitor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "competitor": ["*.yaml", "*.yml"],
    },
)