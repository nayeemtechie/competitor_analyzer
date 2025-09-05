# src/competitor/reports/__init__.py
"""
Report generation modules for competitor analysis
"""

from .pdf_reports import PDFReportGenerator
from .docx_reports import DOCXReportGenerator
from .json_reports import JSONReportGenerator

class ReportGenerator:
    """Main report generator that coordinates all report formats"""
    
    def __init__(self, config, llm_provider):
        self.config = config
        self.llm_provider = llm_provider
        
        # Initialize report generators
        self.pdf_generator = PDFReportGenerator(config, llm_provider)
        self.docx_generator = DOCXReportGenerator(config, llm_provider)
        self.json_generator = JSONReportGenerator(config, llm_provider)
    
    async def generate_pdf_report(self, intelligence):
        """Generate PDF report"""
        return await self.pdf_generator.generate_report(intelligence)
    
    async def generate_docx_report(self, intelligence):
        """Generate Word document report"""
        return await self.docx_generator.generate_report(intelligence)
    
    async def generate_json_report(self, intelligence):
        """Generate JSON data export"""
        return await self.json_generator.generate_report(intelligence)

__all__ = ['ReportGenerator', 'PDFReportGenerator', 'DOCXReportGenerator', 'JSONReportGenerator']

# src/competitor/reports/pdf_reports.py
"""
PDF report generator for competitor analysis
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PDFReportGenerator:
    """Generates comprehensive PDF reports"""
    
    def __init__(self, config, llm_provider):
        self.config = config
        self.llm_provider = llm_provider
        self.output_dir = Path(config.output_config.get('output_dir', 'competitor_reports'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_report(self, intelligence) -> str:
        """Generate comprehensive PDF report"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        except ImportError:
            logger.error("ReportLab not installed. Run: pip install reportlab")
            return await self._generate_simple_text_report(intelligence)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"competitor_analysis_{timestamp}.pdf"
        filepath = self.output_dir / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2C3E50')
        )

            heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#34495E'),
            borderWidth=1,
            borderColor=colors.HexColor('#BDC3C7'),
            borderPadding=8
        )
        
        # Title page
        story.append(Paragraph("Competitor Analysis Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"Competitors Analyzed: {len(intelligence.profiles)}", styles['Normal']))
        story.append(PageBreak())
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        executive_summary = await self._generate_executive_summary(intelligence)
        story.append(Paragraph(executive_summary, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Threat Level Matrix
        story.append(Paragraph("Competitive Threat Matrix", heading_style))
        threat_table = self._create_threat_matrix_table(intelligence)
        if threat_table:
            story.append(threat_table)
        story.append(PageBreak())
        
        # Individual Competitor Profiles
        for profile in intelligence.profiles:
            story.extend(await self._create_competitor_profile_section(profile, styles))
            story.append(PageBreak())
        
        # Market Analysis
        story.append(Paragraph("Market Landscape Analysis", heading_style))
        market_analysis = await self._generate_market_analysis(intelligence)
        story.append(Paragraph(market_analysis, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Strategic Recommendations
        story.append(Paragraph("Strategic Recommendations", heading_style))
        recommendations = await self._generate_recommendations(intelligence)
        story.append(Paragraph(recommendations, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PDF report generated: {filepath}")
        return str(filepath)
    
    async def _generate_executive_summary(self, intelligence) -> str:
        """Generate executive summary using LLM"""
        try:
            competitor_names = [p.name for p in intelligence.profiles]
            high_threat_count = len([p for p in intelligence.profiles 
                                   if p.threat_level and p.threat_level.value in ['high', 'critical']])
            
            prompt = f"""
            Create an executive summary for a competitive analysis of {len(intelligence.profiles)} ecommerce search competitors: {', '.join(competitor_names)}.
            
            Key findings:
            - {high_threat_count} competitors pose high/critical threat
            - Market analysis covers funding, features, and market positioning
            - Analysis includes technology trends and innovation indicators
            
            Structure the summary with:
            1. Market Overview (2-3 sentences)
            2. Primary Competitive Threats (3 bullet points)
            3. Strategic Opportunities (2-3 bullet points)
            4. Immediate Action Items (3 bullet points with timeframes)
            
            Keep under 400 words, executive-friendly, and actionable.
            """
            
            model = self.config.get_llm_model('summary')
            summary = self.llm_provider.chat(
                system="You are a senior business analyst creating executive summaries for C-level executives.",
                user=prompt,
                model=model
            )
            
            return summary
            
        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}")
            return self._create_fallback_summary(intelligence)
    
    def _create_fallback_summary(self, intelligence) -> str:
        """Create fallback summary without LLM"""
        competitor_count = len(intelligence.profiles)
        high_threat_count = len([p for p in intelligence.profiles 
                               if p.threat_level and p.threat_level.value in ['high', 'critical']])
        
        return f"""
        EXECUTIVE SUMMARY
        
        This analysis examines {competitor_count} key competitors in the ecommerce search market, providing strategic insights for competitive positioning and market response.
        
        KEY FINDINGS:
        • {high_threat_count} competitors pose high competitive threat requiring immediate attention
        • Market shows continued consolidation with active funding and feature development
        • AI/ML capabilities becoming standard across leading vendors
        
        STRATEGIC IMPLICATIONS:
        • Accelerated product development needed to maintain competitive parity
        • Pricing pressure likely to increase as market matures
        • Customer retention becomes critical as competitors strengthen offerings
        
        RECOMMENDED ACTIONS:
        • Conduct detailed feature gap analysis within 90 days
        • Evaluate pricing strategy competitiveness by end of quarter
        • Strengthen customer success and retention programs immediately
        """
    
    def _create_threat_matrix_table(self, intelligence):
        """Create threat level matrix table"""
        try:
            from reportlab.platypus import Table, TableStyle
            from reportlab.lib import colors
            
            # Prepare data
            data = [['Competitor', 'Threat Level', 'Funding', 'Key Strengths', 'Primary Concerns']]
            
            for profile in intelligence.profiles:
                threat_level = profile.threat_level.value if profile.threat_level else 'medium'
                funding = profile.funding_info.total_funding if profile.funding_info else 'Unknown'
                
                # Key strengths (simplified)
                strengths = []
                if profile.key_features and len(profile.key_features) > 15:
                    strengths.append("Comprehensive features")
                if profile.funding_info and profile.funding_info.total_funding and 'B' in profile.funding_info.total_funding:
                    strengths.append("Well-funded")
                if profile.case_studies and len(profile.case_studies) > 5:
                    strengths.append("Strong customer base")
                
                # Primary concerns
                concerns = []
                if threat_level in ['high', 'critical']:
                    concerns.append("Direct competitive threat")
                if profile.job_postings and len(profile.job_postings) > 15:
                    concerns.append("Aggressive hiring")
                if profile.recent_news and len(profile.recent_news) > 5:
                    concerns.append("High market activity")
                
                data.append([
                    profile.name,
                    threat_level.title(),
                    funding,
                    '; '.join(strengths[:2]) if strengths else 'Standard offering',
                    '; '.join(concerns[:2]) if concerns else 'Moderate activity'
                ])
            
            # Create table
            table = Table(data, colWidths=[1.5*inch, 1*inch, 1*inch, 2*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            
            return table
            
        except Exception as e:
            logger.warning(f"Failed to create threat matrix table: {e}")
            return None
    
    async def _create_competitor_profile_section(self, profile, styles):
        """Create detailed competitor profile section"""
        section = []
        
        # Competitor name as section header
        section.append(Paragraph(f"Competitor Profile: {profile.name}", styles['Heading2']))
        
        # Basic information
        basic_info = f"""
        <b>Website:</b> {profile.website}<br/>
        <b>Threat Level:</b> {profile.threat_level.value.title() if profile.threat_level else 'Medium'}<br/>
        <b>Target Markets:</b> {', '.join(profile.target_markets) if profile.target_markets else 'Unknown'}<br/>
        <b>Last Analyzed:</b> {profile.last_analyzed or 'N/A'}
        """
        section.append(Paragraph(basic_info, styles['Normal']))
        section.append(Spacer(1, 0.2*inch))
        
        # Funding information
        if profile.funding_info:
            funding_text = f"""
            <b>Funding Overview:</b><br/>
            Total Funding: {profile.funding_info.total_funding or 'Unknown'}<br/>
            Last Round: {profile.funding_info.last_round_amount or 'Unknown'} 
            ({profile.funding_info.last_round_type or 'Unknown'}) in {profile.funding_info.last_round_date or 'Unknown'}
            """
            section.append(Paragraph(funding_text, styles['Normal']))
            section.append(Spacer(1, 0.1*inch))
        
        # Key features
        if profile.key_features:
            section.append(Paragraph("<b>Key Features & Capabilities:</b>", styles['Normal']))
            features_text = "• " + "<br/>• ".join(profile.key_features[:10])  # Limit to top 10
            section.append(Paragraph(features_text, styles['Normal']))
            section.append(Spacer(1, 0.1*inch))
        
        # Technology stack
        if profile.technology_stack:
            tech_text = f"<b>Technology Stack:</b> {', '.join(profile.technology_stack)}"
            section.append(Paragraph(tech_text, styles['Normal']))
            section.append(Spacer(1, 0.1*inch))
        
        # Recent activity
        recent_activity = []
        if profile.job_postings:
            recent_activity.append(f"{len(profile.job_postings)} active job postings")
        if profile.recent_news:
            recent_activity.append(f"{len(profile.recent_news)} recent news mentions")
        if profile.github_activity and profile.github_activity.public_repos:
            recent_activity.append(f"{profile.github_activity.public_repos} public repositories")
        
        if recent_activity:
            activity_text = f"<b>Recent Activity:</b> {', '.join(recent_activity)}"
            section.append(Paragraph(activity_text, styles['Normal']))
        
        section.append(Spacer(1, 0.2*inch))
        
        return section
    
    async def _generate_market_analysis(self, intelligence) -> str:
        """Generate market landscape analysis"""
        try:
            market_data = {
                'total_competitors': len(intelligence.profiles),
                'well_funded_count': len([p for p in intelligence.profiles 
                                        if p.funding_info and p.funding_info.total_funding and 'B' in p.funding_info.total_funding]),
                'high_threat_count': len([p for p in intelligence.profiles 
                                        if p.threat_level and p.threat_level.value in ['high', 'critical']]),
                'avg_features': sum(len(p.key_features) if p.key_features else 0 for p in intelligence.profiles) / len(intelligence.profiles)
            }
            
            prompt = f"""
            Analyze the competitive landscape for ecommerce search based on these market indicators:
            
            Market Structure:
            - {market_data['total_competitors']} major competitors analyzed
            - {market_data['well_funded_count']} billion-dollar funded companies
            - {market_data['high_threat_count']} pose high competitive threat
            - Average of {market_data['avg_features']:.1f} features per competitor
            
            Provide analysis covering:
            1. Market maturity and competitive intensity
            2. Key competitive dynamics and trends
            3. Market consolidation vs fragmentation
            4. Innovation trends across competitors
            
            Keep analysis strategic and under 300 words.
            """
            
            model = self.config.get_llm_model('analysis')
            analysis = self.llm_provider.chat(
                system="You are a market research analyst specializing in B2B SaaS competitive landscapes.",
                user=prompt,
                model=model
            )
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Market analysis generation failed: {e}")
            return self._create_fallback_market_analysis(intelligence)
    
    def _create_fallback_market_analysis(self, intelligence) -> str:
        """Create fallback market analysis"""
        competitor_count = len(intelligence.profiles)
        well_funded = len([p for p in intelligence.profiles 
                          if p.funding_info and p.funding_info.total_funding])
        
        return f"""
        MARKET LANDSCAPE ANALYSIS
        
        The ecommerce search market shows high competitive intensity with {competitor_count} major players analyzed. 
        
        MARKET MATURITY: The presence of {well_funded} well-funded competitors indicates a maturing market with established players and significant barriers to entry.
        
        COMPETITIVE DYNAMICS: Competition is primarily feature-driven, with most vendors offering comprehensive search capabilities. Differentiation focuses on AI/ML capabilities, ease of integration, and vertical specialization.
        
        INNOVATION TRENDS: Artificial intelligence and machine learning are becoming table stakes, with advanced personalization and real-time capabilities emerging as key differentiators.
        
        STRATEGIC IMPLICATIONS: The market is consolidating around well-funded leaders while maintaining room for specialized players with unique value propositions.
        """
    
    async def _generate_recommendations(self, intelligence) -> str:
        """Generate strategic recommendations"""
        try:
            high_threat_competitors = [p.name for p in intelligence.profiles 
                                     if p.threat_level and p.threat_level.value in ['high', 'critical']]
            
            prompt = f"""
            Based on competitive analysis of ecommerce search competitors, provide strategic recommendations.
            
            Context:
            - {len(intelligence.profiles)} competitors analyzed
            - High threat competitors: {', '.join(high_threat_competitors) if high_threat_competitors else 'None identified'}
            - Market shows active funding and feature development
            
            Provide 5-7 specific, actionable recommendations with:
            1. Clear action items
            2. Suggested timeframes (90 days, 6 months, 1 year)
            3. Business rationale
            
            Focus on product strategy, competitive positioning, and market response.
            """
            
            model = self.config.get_llm_model('analysis')
            recommendations = self.llm_provider.chat(
                system="You are a strategic business consultant providing competitive response recommendations.",
                user=prompt,
                model=model
            )
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Recommendations generation failed: {e}")
            return self._create_fallback_recommendations(intelligence)
    
    def _create_fallback_recommendations(self, intelligence) -> str:
        """Create fallback recommendations"""
        return """
        STRATEGIC RECOMMENDATIONS
        
        1. IMMEDIATE ACTIONS (0-90 days):
           • Conduct detailed feature gap analysis against top 3 competitors
           • Strengthen customer retention programs and success metrics
           • Evaluate pricing strategy competitiveness
        
        2. SHORT-TERM INITIATIVES (3-6 months):
           • Accelerate AI/ML capability development to match market standards
           • Enhance integration ecosystem and developer experience
           • Implement competitive monitoring and intelligence system
        
        3. LONG-TERM STRATEGY (6-12 months):
           • Consider strategic partnerships or acquisitions for capability gaps
           • Develop vertical-specific solutions for market differentiation
           • Invest in next-generation search technologies (voice, visual, semantic)
        
        4. CONTINUOUS ACTIVITIES:
           • Monitor competitor funding rounds and strategic moves
           • Track feature releases and market positioning changes
           • Maintain competitive battle cards for sales team
        """
    
    async def _generate_simple_text_report(self, intelligence) -> str:
        """Generate simple text report when PDF libraries unavailable"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"competitor_analysis_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("COMPETITOR ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Competitors Analyzed: {len(intelligence.profiles)}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            summary = await self._generate_executive_summary(intelligence)
            f.write(summary + "\n\n")
            
            # Competitor Profiles
            f.write("COMPETITOR PROFILES\n")
            f.write("-" * 20 + "\n")
            for profile in intelligence.profiles:
                f.write(f"\n{profile.name}\n")
                f.write("=" * len(profile.name) + "\n")
                f.write(f"Website: {profile.website}\n")
                f.write(f"Threat Level: {profile.threat_level.value if profile.threat_level else 'Medium'}\n")
                
                if profile.funding_info and profile.funding_info.total_funding:
                    f.write(f"Funding: {profile.funding_info.total_funding}\n")
                
                if profile.key_features:
                    f.write(f"Key Features: {len(profile.key_features)} features identified\n")
                
                if profile.job_postings:
                    f.write(f"Active Hiring: {len(profile.job_postings)} job postings\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("STRATEGIC RECOMMENDATIONS\n")
            f.write("-" * 25 + "\n")
            recommendations = await self._generate_recommendations(intelligence)
            f.write(recommendations + "\n")
        
        logger.info(f"Text report generated: {filepath}")
        return str(filepath)

# src/competitor/reports/docx_reports.py
"""
Word document report generator for competitor analysis
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DOCXReportGenerator:
    """Generates Word document reports"""
    
    def __init__(self, config, llm_provider):
        self.config = config
        self.llm_provider = llm_provider
        self.output_dir = Path(config.output_config.get('output_dir', 'competitor_reports'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_report(self, intelligence) -> str:
        """Generate Word document report"""
        try:
            from docx import Document
            from docx.shared import Inches
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            from docx.enum.style import WD_STYLE_TYPE
        except ImportError:
            logger.error("python-docx not installed. Run: pip install python-docx")
            return await self._generate_simple_text_report(intelligence)
        
        # Create document
        doc = Document()
        
        # Title
        title = doc.add_heading('Competitor Analysis Report', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Metadata
        doc.add_paragraph(f'Generated: {datetime.now().strftime("%B %d, %Y")}')
        doc.add_paragraph(f'Competitors Analyzed: {len(intelligence.profiles)}')
        doc.add_page_break()
        
        # Table of Contents placeholder
        doc.add_heading('Table of Contents', level=1)
        toc_items = [
            '1. Executive Summary',
            '2. Competitive Threat Matrix',
            '3. Individual Competitor Profiles',
            '4. Market Landscape Analysis',
            '5. Strategic Recommendations'
        ]
        for item in toc_items:
            doc.add_paragraph(item, style='List Number')
        doc.add_page_break()
        
        # Executive Summary
        doc.add_heading('Executive Summary', level=1)
        executive_summary = await self._generate_executive_summary(intelligence)
        doc.add_paragraph(executive_summary)
        doc.add_page_break()
        
        # Competitive Threat Matrix
        doc.add_heading('Competitive Threat Matrix', level=1)
        self._add_threat_matrix_table(doc, intelligence)
        doc.add_page_break()
        
        # Individual Competitor Profiles
        doc.add_heading('Individual Competitor Profiles', level=1)
        for profile in intelligence.profiles:
            self._add_competitor_profile(doc, profile)
            doc.add_page_break()
        
        # Market Analysis
        doc.add_heading('Market Landscape Analysis', level=1)
        market_analysis = await self._generate_market_analysis(intelligence)
        doc.add_paragraph(market_analysis)
        doc.add_page_break()
        
        # Strategic Recommendations
        doc.add_heading('Strategic Recommendations', level=1)
        recommendations = await self._generate_recommendations(intelligence)
        doc.add_paragraph(recommendations)
        
        # Save document
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"competitor_analysis_{timestamp}.docx"
        filepath = self.output_dir / filename
        doc.save(str(filepath))
        
        logger.info(f"DOCX report generated: {filepath}")
        return str(filepath)
    
    def _add_threat_matrix_table(self, doc, intelligence):
        """Add threat matrix table to document"""
        try:
            # Create table
            table = doc.add_table(rows=1, cols=5)
            table.style = 'Light Grid Accent 1'
            
            # Header row
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'Competitor'
            hdr_cells[1].text = 'Threat Level'
            hdr_cells[2].text = 'Funding'
            hdr_cells[3].text = 'Key Strengths'
            hdr_cells[4].text = 'Primary Concerns'
            
            # Data rows
            for profile in intelligence.profiles:
                row_cells = table.add_row().cells
                row_cells[0].text = profile.name
                row_cells[1].text = profile.threat_level.value.title() if profile.threat_level else 'Medium'
                row_cells[2].text = profile.funding_info.total_funding if profile.funding_info else 'Unknown'
                
                # Key strengths
                strengths = []
                if profile.key_features and len(profile.key_features) > 15:
                    strengths.append("Comprehensive features")
                if profile.case_studies and len(profile.case_studies) > 5:
                    strengths.append("Strong customer base")
                row_cells[3].text = '; '.join(strengths[:2]) if strengths else 'Standard offering'
                
                # Primary concerns
                concerns = []
                if profile.threat_level and profile.threat_level.value in ['high', 'critical']:
                    concerns.append("Direct competitive threat")
                if profile.job_postings and len(profile.job_postings) > 15:
                    concerns.append("Aggressive hiring")
                row_cells[4].text = '; '.join(concerns[:2]) if concerns else 'Moderate activity'
            
        except Exception as e:
            logger.warning(f"Failed to create threat matrix table: {e}")
            doc.add_paragraph("Threat matrix table could not be generated.")
    
    def _add_competitor_profile(self, doc, profile):
        """Add individual competitor profile section"""
        # Competitor name
        doc.add_heading(f'{profile.name}', level=2)
        
        # Basic information
        doc.add_paragraph(f'Website: {profile.website}')
        doc.add_paragraph(f'Threat Level: {profile.threat_level.value.title() if profile.threat_level else "Medium"}')
        if profile.target_markets:
            doc.add_paragraph(f'Target Markets: {", ".join(profile.target_markets)}')
        
        # Funding information
        if profile.funding_info:
            doc.add_heading('Funding Overview', level=3)
            funding_para = doc.add_paragraph()
            funding_para.add_run('Total Funding: ').bold = True
            funding_para.add_run(profile.funding_info.total_funding or 'Unknown')
            
            if profile.funding_info.last_round_amount:
                funding_para.add_run(f'\nLast Round: {profile.funding_info.last_round_amount}')
                if profile.funding_info.last_round_type:
                    funding_para.add_run(f' ({profile.funding_info.last_round_type})')
        
        # Key features
        if profile.key_features:
            doc.add_heading('Key Features & Capabilities', level=3)
            for feature in profile.key_features[:10]:  # Limit to top 10
                doc.add_paragraph(feature, style='List Bullet')
        
        # Technology stack
        if profile.technology_stack:
            doc.add_heading('Technology Stack', level=3)
            doc.add_paragraph(', '.join(profile.technology_stack))
        
        # Recent activity
        if profile.job_postings or profile.recent_news or profile.github_activity:
            doc.add_heading('Recent Activity', level=3)
            activity_items = []
            
            if profile.job_postings:
                activity_items.append(f'{len(profile.job_postings)} active job postings')
            if profile.recent_news:
                activity_items.append(f'{len(profile.recent_news)} recent news mentions')
            if profile.github_activity and profile.github_activity.public_repos:
                activity_items.append(f'{profile.github_activity.public_repos} public repositories')
            
            for item in activity_items:
                doc.add_paragraph(item, style='List Bullet')
    
    async def _generate_executive_summary(self, intelligence) -> str:
        """Generate executive summary (reuse PDF logic)"""
        # Reuse the same logic as PDF generator
        pdf_generator = PDFReportGenerator(self.config, self.llm_provider)
        return await pdf_generator._generate_executive_summary(intelligence)
    
    async def _generate_market_analysis(self, intelligence) -> str:
        """Generate market analysis (reuse PDF logic)"""
        pdf_generator = PDFReportGenerator(self.config, self.llm_provider)
        return await pdf_generator._generate_market_analysis(intelligence)
    
    async def _generate_recommendations(self, intelligence) -> str:
        """Generate recommendations (reuse PDF logic)"""
        pdf_generator = PDFReportGenerator(self.config, self.llm_provider)
        return await pdf_generator._generate_recommendations(intelligence)
    
    async def _generate_simple_text_report(self, intelligence) -> str:
        """Fallback to text report if DOCX unavailable"""
        pdf_generator = PDFReportGenerator(self.config, self.llm_provider)
        return await pdf_generator._generate_simple_text_report(intelligence)

# src/competitor/reports/json_reports.py
"""
JSON report generator for competitor analysis data export
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class JSONReportGenerator:
    """Generates JSON data exports"""
    
    def __init__(self, config, llm_provider):
        self.config = config
        self.llm_provider = llm_provider
        self.output_dir = Path(config.output_config.get('output_dir', 'competitor_reports'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON export settings
        self.json_config = config.output_config.get('json', {})
        self.pretty_print = self.json_config.get('pretty_print', True)
        self.include_raw_data = self.json_config.get('include_raw_data', False)
    
    async def generate_report(self, intelligence) -> str:
        """Generate comprehensive JSON data export"""
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"competitor_data_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Prepare export data
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator_version': '1.0',
                'total_competitors': len(intelligence.profiles),
                'analysis_config': self._serialize_config(intelligence.config) if intelligence.config else {},
                'data_sources_used': intelligence.metadata.get('data_sources_used', []),
                'llm_models_used': intelligence.metadata.get('llm_models_used', {})
            },
            'executive_summary': await self._generate_executive_insights(intelligence),
            'competitors': [],
            'market_analysis': await self._generate_market_insights(intelligence),
            'threat_matrix': self._generate_threat_matrix(intelligence),
            'strategic_recommendations': await self._generate_strategic_insights(intelligence)
        }
        
        # Add competitor data
        for profile in intelligence.profiles:
            competitor_data = await self._serialize_competitor_profile(profile)
            export_data['competitors'].append(competitor_data)
        
        # Write JSON file
        indent = 2 if self.pretty_print else None
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=indent, default=self._json_serializer, ensure_ascii=False)
        
        logger.info(f"JSON report generated: {filepath}")
        return str(filepath)
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for complex objects"""
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # dataclass or custom objects
            return asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.__dict__
        elif hasattr(obj, 'value'):  # Enum objects
            return obj.value
        else:
            return str(obj)
    
    async def _serialize_competitor_profile(self, profile) -> Dict[str, Any]:
        """Serialize competitor profile to JSON-friendly format"""
        # Convert profile to dictionary
        profile_dict = asdict(profile)
        
        # Add computed metrics
        profile_dict['computed_metrics'] = {
            'feature_count': len(profile.key_features) if profile.key_features else 0,
            'case_study_count': len(profile.case_studies) if profile.case_studies else 0,
            'job_posting_count': len(profile.job_postings) if profile.job_postings else 0,
            'news_mention_count': len(profile.recent_news) if profile.recent_news else 0,
            'social_platform_count': len(profile.social_presence) if profile.social_presence else 0,
            'technology_count': len(profile.technology_stack) if profile.technology_stack else 0,
            'total_patents': profile.patent_data.total_patents if profile.patent_data else 0,
            'github_repos': profile.github_activity.public_repos if profile.github_activity else 0
        }
        
        # Add analysis scores if available
        if hasattr(profile, 'feature_score') and profile.feature_score:
            profile_dict['analysis_scores'] = {
                'feature_score': profile.feature_score,
                'market_position_score': getattr(profile, 'market_position_score', None),
                'innovation_score': getattr(profile, 'innovation_score', None),
                'overall_threat_score': getattr(profile, 'overall_threat_score', None)
            }
        
        # Clean up None values if not including raw data
        if not self.include_raw_data:
            profile_dict = self._clean_none_values(profile_dict)
        
        return profile_dict
    
    def _clean_none_values(self, data):
        """Remove None values from nested dictionaries"""
        if isinstance(data, dict):
            return {k: self._clean_none_values(v) for k, v in data.items() if v is not None}
        elif isinstance(data, list):
            return [self._clean_none_values(item) for item in data if item is not None]
        else:
            return data
    
    def _serialize_config(self, config) -> Dict[str, Any]:
        """Serialize analysis configuration"""
        if not config:
            return {}
        
        try:
            config_dict = asdict(config)
            return config_dict
        except:
            # Fallback for non-dataclass config
            return {
                'depth_level': getattr(config, 'depth_level', 'standard'),
                'competitors': getattr(config, 'competitors', []),
                'output_formats': getattr(config, 'output_formats', []),
                'data_sources': {
                    'analyze_website': getattr(config, 'analyze_website', True),
                    'analyze_funding': getattr(config, 'analyze_funding', True),
                    'analyze_jobs': getattr(config, 'analyze_jobs', True),
                    'analyze_news': getattr(config, 'analyze_news', True),
                    'analyze_social': getattr(config, 'analyze_social', True),
                    'analyze_github': getattr(config, 'analyze_github', True)
                }
            }
    
    async def _generate_executive_insights(self, intelligence) -> Dict[str, Any]:
        """Generate executive-level insights"""
        insights = {
            'market_overview': await self._analyze_market_overview(intelligence),
            'competitive_landscape': self._analyze_competitive_landscape(intelligence),
            'threat_assessment': self._analyze_threat_assessment(intelligence),
            'key_findings': await self._extract_key_findings(intelligence)
        }
        
        return insights
    
    async def _analyze_market_overview(self, intelligence) -> Dict[str, Any]:
        """Analyze market overview metrics"""
        overview = {
            'total_competitors_analyzed': len(intelligence.profiles),
            'market_segments_covered': set(),
            'funding_landscape': {
                'total_funding_tracked': 0,
                'well_funded_competitors': 0,
                'recent_funding_activity': 0
            },
            'innovation_indicators': {
                'ai_adoption_rate': 0,
                'patent_activity': 0,
                'github_activity_score': 0
            }
        }
        
        # Collect market segments
        for profile in intelligence.profiles:
            if profile.target_markets:
                overview['market_segments_covered'].update(profile.target_markets)
        
        overview['market_segments_covered'] = list(overview['market_segments_covered'])
        
        # Analyze funding landscape
        well_funded_count = 0
        recent_funding_count = 0
        
        for profile in intelligence.profiles:
            if profile.funding_info and profile.funding_info.total_funding:
                if 'B' in profile.funding_info.total_funding or \
                   ('M' in profile.funding_info.total_funding and '500' in profile.funding_info.total_funding):
                    well_funded_count += 1
                
                # Check for recent funding (within 2 years)
                if profile.funding_info.last_round_date:
                    try:
                        from datetime import datetime
                        last_round = datetime.fromisoformat(profile.funding_info.last_round_date.replace('Z', '+00:00'))
                        if (datetime.now() - last_round).days < 730:
                            recent_funding_count += 1
                    except:
                        pass
        
        overview['funding_landscape']['well_funded_competitors'] = well_funded_count
        overview['funding_landscape']['recent_funding_activity'] = recent_funding_count
        
        # Analyze innovation indicators
        ai_adoption_count = 0
        total_patents = 0
        github_scores = []
        
        for profile in intelligence.profiles:
            # AI adoption
            if profile.key_features:
                ai_features = [f for f in profile.key_features 
                              if any(word in f.lower() for word in ['ai', 'ml', 'machine learning'])]
                if ai_features:
                    ai_adoption_count += 1
            
            # Patent activity
            if profile.patent_data:
                total_patents += profile.patent_data.total_patents
            
            # GitHub activity
            if profile.github_activity and profile.github_activity.activity_score:
                github_scores.append(profile.github_activity.activity_score)
        
        overview['innovation_indicators']['ai_adoption_rate'] = ai_adoption_count / len(intelligence.profiles)
        overview['innovation_indicators']['patent_activity'] = total_patents
        overview['innovation_indicators']['github_activity_score'] = sum(github_scores) / len(github_scores) if github_scores else 0
        
        return overview
    
    def _analyze_competitive_landscape(self, intelligence) -> Dict[str, Any]:
        """Analyze competitive landscape structure"""
        landscape = {
            'threat_distribution': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'market_positioning': {
                'enterprise_focused': 0,
                'mid_market_focused': 0,
                'broad_market': 0
            },
            'technology_adoption': {
                'modern_stack_adoption': 0,
                'ai_capabilities': 0,
                'open_source_activity': 0
            }
        }
        
        # Threat distribution
        for profile in intelligence.profiles:
            threat_level = profile.threat_level.value if profile.threat_level else 'medium'
            landscape['threat_distribution'][threat_level] += 1
        
        # Market positioning
        for profile in intelligence.profiles:
            if profile.target_markets:
                if 'enterprise' in profile.target_markets:
                    landscape['market_positioning']['enterprise_focused'] += 1
                elif 'mid-market' in profile.target_markets:
                    landscape['market_positioning']['mid_market_focused'] += 1
                elif len(profile.target_markets) > 2:
                    landscape['market_positioning']['broad_market'] += 1
        
        # Technology adoption
        for profile in intelligence.profiles:
            # Modern stack
            if profile.technology_stack:
                modern_tech = ['Kubernetes', 'GraphQL', 'Microservices', 'React']
                if any(tech in profile.technology_stack for tech in modern_tech):
                    landscape['technology_adoption']['modern_stack_adoption'] += 1
            
            # AI capabilities
            if profile.key_features:
                ai_features = [f for f in profile.key_features 
                              if 'ai' in f.lower() or 'ml' in f.lower()]
                if ai_features:
                    landscape['technology_adoption']['ai_capabilities'] += 1
            
            # Open source activity
            if profile.github_activity and profile.github_activity.public_repos > 10:
                landscape['technology_adoption']['open_source_activity'] += 1
        
        return landscape
    
    def _analyze_threat_assessment(self, intelligence) -> Dict[str, Any]:
        """Analyze threat assessment summary"""
        assessment = {
            'highest_threat_competitors': [],
            'emerging_threats': [],
            'threat_categories': {
                'funding_threats': [],
                'technology_threats': [],
                'market_threats': []
            }
        }
        
        # Sort by threat level
        sorted_profiles = sorted(intelligence.profiles, 
                               key=lambda p: self._get_threat_score(p), 
                               reverse=True)
        
        # Highest threats
        assessment['highest_threat_competitors'] = [
            {
                'name': p.name,
                'threat_level': p.threat_level.value if p.threat_level else 'medium',
                'key_concerns': self._identify_key_concerns(p)
            }
            for p in sorted_profiles[:3]
        ]
        
        # Categorize threats
        for profile in intelligence.profiles:
            # Funding threats
            if profile.funding_info and profile.funding_info.total_funding:
                if 'B' in profile.funding_info.total_funding:
                    assessment['threat_categories']['funding_threats'].append(profile.name)
            
            # Technology threats
            if profile.technology_stack and len(profile.technology_stack) > 5:
                assessment['threat_categories']['technology_threats'].append(profile.name)
            
            # Market threats
            if profile.case_studies and len(profile.case_studies) > 8:
                assessment['threat_categories']['market_threats'].append(profile.name)
        
        return assessment
    
    def _get_threat_score(self, profile) -> float:
        """Get numerical threat score for sorting"""
        threat_mapping = {
            'critical': 4.0,
            'high': 3.0,
            'medium': 2.0,
            'low': 1.0
        }
        
        threat_level = profile.threat_level.value if profile.threat_level else 'medium'
        return threat_mapping.get(threat_level, 2.0)
    
    def _identify_key_concerns(self, profile) -> List[str]:
        """Identify key concerns for a competitor"""
        concerns = []
        
        # Funding concerns
        if profile.funding_info and profile.funding_info.total_funding:
            if 'B' in profile.funding_info.total_funding:
                concerns.append("Well-funded with significant resources")
        
        # Market concerns
        if profile.case_studies and len(profile.case_studies) > 10:
            concerns.append("Strong customer portfolio")
        
        # Innovation concerns
        if profile.job_postings:
            eng_jobs = [job for job in profile.job_postings 
                       if 'engineering' in job.department.lower()]
            if len(eng_jobs) > 10:
                concerns.append("Aggressive engineering hiring")
        
        # Technology concerns
        if profile.technology_stack:
            modern_tech_count = len([tech for tech in profile.technology_stack 
                                   if tech in ['AI', 'ML', 'Kubernetes', 'GraphQL']])
            if modern_tech_count > 2:
                concerns.append("Advanced technology stack")
        
        return concerns[:3]  # Limit to top 3 concerns
    
    async def _extract_key_findings(self, intelligence) -> List[str]:
        """Extract key findings from the analysis"""
        findings = []
        
        # Market structure finding
        total_competitors = len(intelligence.profiles)
        findings.append(f"Analyzed {total_competitors} major competitors in ecommerce search market")
        
        # Threat finding
        high_threat_count = len([p for p in intelligence.profiles 
                               if p.threat_level and p.threat_level.value in ['high', 'critical']])
        if high_threat_count > 0:
            findings.append(f"{high_threat_count} competitors pose high/critical competitive threat")
        
        # Funding finding
        well_funded_count = len([p for p in intelligence.profiles 
                               if p.funding_info and p.funding_info.total_funding and 'B' in p.funding_info.total_funding])
        if well_funded_count > 0:
            findings.append(f"{well_funded_count} competitors have billion-dollar funding levels")
        
        # Innovation finding
        ai_adoption_count = 0
        for profile in intelligence.profiles:
            if profile.key_features and any('ai' in f.lower() or 'ml' in f.lower() for f in profile.key_features):
                ai_adoption_count += 1
        
        if ai_adoption_count > total_competitors * 0.5:
            findings.append("AI/ML capabilities becoming standard across majority of competitors")
        
        # Hiring activity finding
        active_hiring_count = len([p for p in intelligence.profiles 
                                 if p.job_postings and len(p.job_postings) > 10])
        if active_hiring_count > 0:
            findings.append(f"{active_hiring_count} competitors showing aggressive hiring patterns")
        
        return findings
    
    async def _generate_market_insights(self, intelligence) -> Dict[str, Any]:
        """Generate market analysis insights"""
        insights = {
            'market_maturity': await self._assess_market_maturity(intelligence),
            'competitive_intensity': self._calculate_competitive_intensity(intelligence),
            'innovation_trends': self._analyze_innovation_trends(intelligence),
            'market_gaps': self._identify_market_gaps(intelligence)
        }
        
        return insights
    
    async def _assess_market_maturity(self, intelligence) -> Dict[str, Any]:
        """Assess market maturity indicators"""
        maturity = {
            'maturity_score': 0.0,
            'indicators': [],
            'assessment': 'developing'
        }
        
        total_competitors = len(intelligence.profiles)
        
        # Calculate maturity indicators
        well_funded_ratio = len([p for p in intelligence.profiles 
                               if p.funding_info and p.funding_info.total_funding]) / total_competitors
        
        feature_rich_ratio = len([p for p in intelligence.profiles 
                                if p.key_features and len(p.key_features) > 15]) / total_competitors
        
        enterprise_focused_ratio = len([p for p in intelligence.profiles 
                                      if p.target_markets and 'enterprise' in p.target_markets]) / total_competitors
        
        # Calculate overall maturity score
        maturity_score = (well_funded_ratio * 0.4 + feature_rich_ratio * 0.3 + enterprise_focused_ratio * 0.3)
        maturity['maturity_score'] = round(maturity_score, 2)
        
        # Determine assessment
        if maturity_score > 0.7:
            maturity['assessment'] = 'mature'
            maturity['indicators'].append('High proportion of well-funded competitors')
            maturity['indicators'].append('Comprehensive feature sets across vendors')
        elif maturity_score > 0.4:
            maturity['assessment'] = 'developing'
            maturity['indicators'].append('Mix of established and emerging players')
        else:
            maturity['assessment'] = 'emerging'
            maturity['indicators'].append('Early-stage market with growth opportunities')
        
        return maturity
    
    def _calculate_competitive_intensity(self, intelligence) -> Dict[str, Any]:
        """Calculate competitive intensity metrics"""
        intensity = {
            'intensity_score': 0.0,
            'factors': [],
            'level': 'moderate'
        }
        
        # Factor calculations
        competitor_density = min(len(intelligence.profiles) / 10, 1.0)  # Normalize to max 10 competitors
        
        funding_intensity = len([p for p in intelligence.profiles 
                               if p.funding_info and p.funding_info.total_funding]) / len(intelligence.profiles)
        
        innovation_intensity = len([p for p in intelligence.profiles 
                                  if p.patent_data and p.patent_data.total_patents > 5]) / len(intelligence.profiles)
        
        # Calculate overall intensity
        intensity_score = (competitor_density * 0.3 + funding_intensity * 0.4 + innovation_intensity * 0.3)
        intensity['intensity_score'] = round(intensity_score, 2)
        
        # Determine level
        if intensity_score > 0.7:
            intensity['level'] = 'high'
            intensity['factors'].append('High competitor density')
            intensity['factors'].append('Strong funding across market')
        elif intensity_score > 0.4:
            intensity['level'] = 'moderate'
            intensity['factors'].append('Balanced competitive environment')
        else:
            intensity['level'] = 'low'
            intensity['factors'].append('Emerging competitive landscape')
        
        return intensity
    
    def _analyze_innovation_trends(self, intelligence) -> Dict[str, Any]:
        """Analyze innovation trends across competitors"""
        trends = {
            'ai_ml_adoption': 0,
            'modern_architecture': 0,
            'open_source_activity': 0,
            'patent_filing_activity': 0,
            'trending_technologies': []
        }
        
        total_competitors = len(intelligence.profiles)
        
        # AI/ML adoption
        ai_adopters = len([p for p in intelligence.profiles 
                          if p.key_features and any('ai' in f.lower() or 'ml' in f.lower() for f in p.key_features)])
        trends['ai_ml_adoption'] = round(ai_adopters / total_competitors, 2)
        
        # Modern architecture
        modern_arch_adopters = len([p for p in intelligence.profiles 
                                  if p.technology_stack and any(tech in p.technology_stack 
                                  for tech in ['Kubernetes', 'Microservices', 'GraphQL'])])
        trends['modern_architecture'] = round(modern_arch_adopters / total_competitors, 2)
        
        # Open source activity
        open_source_active = len([p for p in intelligence.profiles 
                                if p.github_activity and p.github_activity.public_repos > 10])
        trends['open_source_activity'] = round(open_source_active / total_competitors, 2)
        
        # Patent activity
        patent_active = len([p for p in intelligence.profiles 
                           if p.patent_data and p.patent_data.total_patents > 5])
        trends['patent_filing_activity'] = round(patent_active / total_competitors, 2)
        
        # Trending technologies
        tech_frequency = {}
        for profile in intelligence.profiles:
            if profile.technology_stack:
                for tech in profile.technology_stack:
                    tech_frequency[tech] = tech_frequency.get(tech, 0) + 1
        
        # Sort by frequency and take top technologies
        sorted_tech = sorted(tech_frequency.items(), key=lambda x: x[1], reverse=True)
        trends['trending_technologies'] = [{'technology': tech, 'adoption_count': count} 
                                         for tech, count in sorted_tech[:10]]
        
        return trends
    
    def _identify_market_gaps(self, intelligence) -> List[str]:
        """Identify potential market gaps"""
        gaps = []
        
        # Analyze target market coverage
        covered_segments = set()
        for profile in intelligence.profiles:
            if profile.target_markets:
                covered_segments.update(profile.target_markets)
        
        # Common segments not well covered
        expected_segments = {'enterprise', 'mid-market', 'smb', 'developer'}
        missing_segments = expected_segments - covered_segments
        
        for segment in missing_segments:
            gaps.append(f"Underserved market segment: {segment}")
        
        # Feature gaps
        all_features = []
        for profile in intelligence.profiles:
            if profile.key_features:
                all_features.extend([f.lower() for f in profile.key_features])
        
        feature_text = ' '.join(all_features)
        
        # Check for missing capabilities
        if 'voice search' not in feature_text:
            gaps.append("Voice search capabilities appear underrepresented")
        
        if 'visual search' not in feature_text:
            gaps.append("Visual search functionality gap identified")
        
        if 'federated search' not in feature_text:
            gaps.append("Federated search across sources not widely offered")
        
        return gaps[:5]  # Limit to top 5 gaps
    
    def _generate_threat_matrix(self, intelligence) -> Dict[str, Any]:
        """Generate threat assessment matrix"""
        matrix = {
            'threat_rankings': [],
            'threat_summary': {
                'critical_threats': 0,
                'high_threats': 0,
                'medium_threats': 0,
                'low_threats': 0
            },
            'threat_factors': {
                'funding_based': [],
                'technology_based': [],
                'market_based': []
            }
        }
        
        # Create threat rankings
        for profile in intelligence.profiles:
            threat_level = profile.threat_level.value if profile.threat_level else 'medium'
            
            threat_info = {
                'competitor': profile.name,
                'threat_level': threat_level,
                'threat_score': self._get_threat_score(profile),
                'key_factors': self._identify_key_concerns(profile)
            }
            
            matrix['threat_rankings'].append(threat_info)
            matrix['threat_summary'][f'{threat_level}_threats'] += 1
        
        # Sort by threat score
        matrix['threat_rankings'].sort(key=lambda x: x['threat_score'], reverse=True)
        
        # Categorize threat factors
        for profile in intelligence.profiles:
            # Funding-based threats
            if profile.funding_info and profile.funding_info.total_funding:
                if 'B' in profile.funding_info.total_funding:
                    matrix['threat_factors']['funding_based'].append(profile.name)
            
            # Technology-based threats
            if profile.technology_stack and len(profile.technology_stack) > 5:
                matrix['threat_factors']['technology_based'].append(profile.name)
            
            # Market-based threats
            if profile.case_studies and len(profile.case_studies) > 8:
                matrix['threat_factors']['market_based'].append(profile.name)
        
        return matrix
    
    async def _generate_strategic_insights(self, intelligence) -> Dict[str, Any]:
        """Generate strategic recommendations and insights"""
        insights = {
            'immediate_actions': [],
            'short_term_initiatives': [],
            'long_term_strategy': [],
            'monitoring_priorities': [],
            'investment_recommendations': []
        }
        
        # Analyze threat levels for recommendations
        high_threat_count = len([p for p in intelligence.profiles 
                               if p.threat_level and p.threat_level.value in ['high', 'critical']])
        
        # Immediate actions (0-90 days)
        if high_threat_count > 0:
            insights['immediate_actions'].append("Conduct detailed competitive feature analysis")
            insights['immediate_actions'].append("Strengthen customer retention programs")
        
        insights['immediate_actions'].append("Evaluate pricing strategy competitiveness")
        insights['immediate_actions'].append("Implement competitive monitoring system")
        
        # Short-term initiatives (3-6 months)
        ai_adoption_rate = len([p for p in intelligence.profiles 
                              if p.key_features and any('ai' in f.lower() for f in p.key_features)]) / len(intelligence.profiles)
        
        if ai_adoption_rate > 0.5:
            insights['short_term_initiatives'].append("Accelerate AI/ML capability development")
        
        insights['short_term_initiatives'].append("Enhance integration ecosystem")
        insights['short_term_initiatives'].append("Develop competitive battle cards")
        
        # Long-term strategy (6-12 months)
        well_funded_count = len([p for p in intelligence.profiles 
                               if p.funding_info and p.funding_info.total_funding])
        
        if well_funded_count > len(intelligence.profiles) * 0.6:
            insights['long_term_strategy'].append("Consider strategic partnerships or acquisitions")
        
        insights['long_term_strategy'].append("Invest in next-generation search technologies")
        insights['long_term_strategy'].append("Develop vertical-specific solutions")
        
        # Monitoring priorities
        for profile in intelligence.profiles:
            if profile.threat_level and profile.threat_level.value in ['high', 'critical']:
                insights['monitoring_priorities'].append({
                    'competitor': profile.name,
                    'focus_areas': ['funding rounds', 'product launches', 'customer wins']
                })
        
        # Investment recommendations
        if ai_adoption_rate > 0.7:
            insights['investment_recommendations'].append("High priority: AI/ML research and development")
        
        if high_threat_count > 2:
            insights['investment_recommendations'].append("Medium priority: Competitive intelligence platform")
        
        insights['investment_recommendations'].append("Ongoing: Customer success and retention initiatives")
        
        return insights(profile.key_features) if profile.key_features else 0,
            'case_study_count': len# src/competitor/reports/__init__.py
"""
Report generation modules for competitor analysis
"""

from .pdf_reports import PDFReportGenerator
from .docx_reports import DOCXReportGenerator
from .json_reports import JSONReportGenerator

class ReportGenerator:
    """Main report generator that coordinates all report formats"""
    
    def __init__(self, config, llm_provider):
        self.config = config
        self.llm_provider = llm_provider
        
        # Initialize report generators
        self.pdf_generator = PDFReportGenerator(config, llm_provider)
        self.docx_generator = DOCXReportGenerator(config, llm_provider)
        self.json_generator = JSONReportGenerator(config, llm_provider)
    
    async def generate_pdf_report(self, intelligence):
        """Generate PDF report"""
        return await self.pdf_generator.generate_report(intelligence)
    
    async def generate_docx_report(self, intelligence):
        """Generate Word document report"""
        return await self.docx_generator.generate_report(intelligence)
    
    async def generate_json_report(self, intelligence):
        """Generate JSON data export"""
        return await self.json_generator.generate_report(intelligence)

__all__ = ['ReportGenerator', 'PDFReportGenerator', 'DOCXReportGenerator', 'JSONReportGenerator']

# src/competitor/reports/pdf_reports.py
"""
PDF report generator for competitor analysis
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PDFReportGenerator:
    """Generates comprehensive PDF reports"""
    
    def __init__(self, config, llm_provider):
        self.config = config
        self.llm_provider = llm_provider
        self.output_dir = Path(config.output_config.get('output_dir', 'competitor_reports'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_report(self, intelligence) -> str:
        """Generate comprehensive PDF report"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        except ImportError:
            logger.error("ReportLab not installed. Run: pip install reportlab")
            return await self._generate_simple_text_report(intelligence)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"competitor_analysis_{timestamp}.pdf"
        filepath = self.output_dir / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2C3E50')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',