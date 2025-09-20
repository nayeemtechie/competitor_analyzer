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
        
        # PDF configuration
        self.pdf_config = config.output_config.get('pdf', {})
        self.brand_color = self.pdf_config.get('brand_color', [52, 152, 219])
        self.include_executive_summary = self.pdf_config.get('include_executive_summary', True)
        self.include_detailed_profiles = self.pdf_config.get('include_detailed_profiles', True)
        self.include_threat_matrix = self.pdf_config.get('include_threat_matrix', True)
        self.include_appendix = self.pdf_config.get('include_appendix', True)
        self.font_size = self.pdf_config.get('font_size', 11)
    
    async def generate_report(self, intelligence) -> str:
        """Generate comprehensive PDF report"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
            from reportlab.graphics.shapes import Drawing
            from reportlab.graphics.charts.barcharts import VerticalBarChart
            from reportlab.graphics.charts.piecharts import Pie
        except ImportError:
            logger.error("ReportLab not installed. Run: pip install reportlab")
            return await self._generate_simple_text_report(intelligence)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"competitor_analysis_{timestamp}.pdf"
        filepath = self.output_dir / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(filepath), pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        brand_color_hex = colors.Color(
            self.brand_color[0]/255, 
            self.brand_color[1]/255, 
            self.brand_color[2]/255
        )
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=brand_color_hex,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=brand_color_hex,
            borderWidth=1,
            borderColor=colors.HexColor('#BDC3C7'),
            borderPadding=8,
            fontName='Helvetica-Bold'
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=brand_color_hex,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=self.font_size,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        )
        
        # Title page
        story.extend(self._create_title_page(intelligence, title_style, normal_style))
        story.append(PageBreak())
        
        # Table of contents
        story.extend(self._create_table_of_contents(intelligence, heading_style, normal_style))
        story.append(PageBreak())
        
        # Executive Summary
        if self.include_executive_summary:
            story.extend(await self._create_executive_summary(intelligence, heading_style, normal_style))
            story.append(PageBreak())
        
        # Competitive Threat Matrix
        if self.include_threat_matrix:
            story.extend(self._create_threat_matrix_section(intelligence, heading_style, subheading_style, normal_style))
            story.append(PageBreak())
        
        # Market Analysis
        story.extend(await self._create_market_analysis_section(intelligence, heading_style, subheading_style, normal_style))
        story.append(PageBreak())
        
        # Individual Competitor Profiles
        if self.include_detailed_profiles:
            for profile in intelligence.profiles:
                story.extend(await self._create_competitor_profile_section(profile, heading_style, subheading_style, normal_style))
                story.append(PageBreak())
        
        # Strategic Recommendations
        story.extend(await self._create_recommendations_section(intelligence, heading_style, subheading_style, normal_style))
        
        # Appendix
        if self.include_appendix:
            story.append(PageBreak())
            story.extend(self._create_appendix(intelligence, heading_style, subheading_style, normal_style))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PDF report generated: {filepath}")
        return str(filepath)
    
    def _create_title_page(self, intelligence, title_style, normal_style) -> List:
        """Create title page content"""
        from reportlab.platypus import Paragraph, Spacer
        from reportlab.lib.units import inch
        
        content = []
        
        # Main title
        content.append(Paragraph("Competitive Intelligence Report", title_style))
        content.append(Spacer(1, 0.3*inch))
        
        # Subtitle
        content.append(Paragraph("Ecommerce Search Market Analysis", normal_style))
        content.append(Spacer(1, 0.2*inch))
        
        # Analysis details
        analysis_date = datetime.now().strftime('%B %d, %Y')
        content.append(Paragraph(f"<b>Generated:</b> {analysis_date}", normal_style))
        content.append(Paragraph(f"<b>Competitors Analyzed:</b> {len(intelligence.profiles)}", normal_style))
        
        if intelligence.analysis_config:
            content.append(Paragraph(
                f"<b>Analysis Depth:</b> {intelligence.analysis_config.depth_level.value.title()}",
                normal_style
            ))
        
        content.append(Spacer(1, 0.3*inch))
        
        # Competitor list
        content.append(Paragraph("<b>Companies Included:</b>", normal_style))
        for profile in intelligence.profiles:
            content.append(Paragraph(f"• {profile.name}", normal_style))
        
        content.append(Spacer(1, 0.5*inch))
        
        # Disclaimer
        disclaimer = """
        <i>This report contains confidential competitive intelligence. 
        The information herein is intended for internal strategic planning purposes only. 
        Distribution should be limited to authorized personnel.</i>
        """
        content.append(Paragraph(disclaimer, normal_style))
        
        return content
    
    def _create_table_of_contents(self, intelligence, heading_style, normal_style) -> List:
        """Create table of contents"""
        from reportlab.platypus import Paragraph
        
        content = []
        content.append(Paragraph("Table of Contents", heading_style))
        
        toc_items = [
            "1. Executive Summary",
            "2. Competitive Threat Matrix", 
            "3. Market Landscape Analysis",
            "4. Individual Competitor Profiles"
        ]
        
        # Add each competitor to TOC
        for i, profile in enumerate(intelligence.profiles, 1):
            toc_items.append(f"   4.{i} {profile.name}")
        
        toc_items.extend([
            "5. Strategic Recommendations",
            "6. Appendix"
        ])
        
        for item in toc_items:
            content.append(Paragraph(item, normal_style))
        
        return content
    
    async def _create_executive_summary(self, intelligence, heading_style, normal_style) -> List:
        """Create executive summary section"""
        from reportlab.platypus import Paragraph, Spacer
        from reportlab.lib.units import inch
        
        content = []
        content.append(Paragraph("Executive Summary", heading_style))
        
        # Generate executive summary
        executive_summary = await self._generate_executive_summary_content(intelligence)
        
        # Split into sections for better formatting
        sections = executive_summary.split('\n\n')
        for section in sections:
            if section.strip():
                content.append(Paragraph(section.strip(), normal_style))
                content.append(Spacer(1, 0.1*inch))
        
        # Key metrics summary
        content.append(Paragraph("<b>Key Market Metrics:</b>", normal_style))
        
        metrics = [
            f"• Total competitors analyzed: {len(intelligence.profiles)}",
            f"• High-threat competitors: {len([p for p in intelligence.profiles if p.threat_level and p.threat_level.value in ['high', 'critical']])}",
            f"• Well-funded competitors: {len([p for p in intelligence.profiles if p.funding_info and p.funding_info.total_funding and 'B' in p.funding_info.total_funding])}",
            f"• Active hiring competitors: {len([p for p in intelligence.profiles if p.job_postings and len(p.job_postings) > 10])}"
        ]
        
        for metric in metrics:
            content.append(Paragraph(metric, normal_style))
        
        return content
    
    def _create_threat_matrix_section(self, intelligence, heading_style, subheading_style, normal_style) -> List:
        """Create threat matrix section"""
        from reportlab.platypus import Paragraph, Table, TableStyle, Spacer
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        content = []
        content.append(Paragraph("Competitive Threat Matrix", heading_style))
        
        # Create threat level table
        table_data = [['Competitor', 'Threat Level', 'Funding Status', 'Key Strengths', 'Primary Concerns']]
        
        for profile in intelligence.profiles:
            threat_level = profile.threat_level.value.title() if profile.threat_level else 'Medium'
            funding = self._get_funding_status(profile)
            strengths = self._get_key_strengths(profile)
            concerns = self._get_primary_concerns(profile)
            
            table_data.append([
                profile.name,
                threat_level,
                funding,
                strengths,
                concerns
            ])
        
        # Create and style table
        table = Table(table_data, colWidths=[1.2*inch, 0.8*inch, 1*inch, 1.8*inch, 1.8*inch])
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # Data styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        content.append(table)
        content.append(Spacer(1, 0.2*inch))
        
        # Threat level distribution
        content.append(Paragraph("Threat Level Distribution", subheading_style))
        
        threat_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for profile in intelligence.profiles:
            level = profile.threat_level.value if profile.threat_level else 'medium'
            threat_counts[level] += 1
        
        distribution_text = []
        for level, count in threat_counts.items():
            if count > 0:
                percentage = (count / len(intelligence.profiles)) * 100
                distribution_text.append(f"• {level.title()}: {count} competitors ({percentage:.1f}%)")
        
        for text in distribution_text:
            content.append(Paragraph(text, normal_style))
        
        return content
    
    async def _create_market_analysis_section(self, intelligence, heading_style, subheading_style, normal_style) -> List:
        """Create market analysis section"""
        from reportlab.platypus import Paragraph, Spacer
        from reportlab.lib.units import inch
        
        content = []
        content.append(Paragraph("Market Landscape Analysis", heading_style))
        
        # Market overview
        content.append(Paragraph("Market Overview", subheading_style))
        market_analysis = await self._generate_market_analysis_content(intelligence)
        
        sections = market_analysis.split('\n\n')
        for section in sections:
            if section.strip():
                content.append(Paragraph(section.strip(), normal_style))
                content.append(Spacer(1, 0.1*inch))
        
        # Competitive dynamics
        content.append(Paragraph("Competitive Dynamics", subheading_style))
        
        dynamics = [
            f"Market consolidation: {len([p for p in intelligence.profiles if p.funding_info and p.funding_info.total_funding])} of {len(intelligence.profiles)} competitors are well-funded",
            f"Innovation focus: {len([p for p in intelligence.profiles if p.key_features and any('ai' in f.lower() for f in p.key_features)])} competitors emphasize AI/ML capabilities",
            f"Geographic expansion: {len([p for p in intelligence.profiles if p.job_postings and any('international' in j.location.lower() if j.location else False for j in p.job_postings)])} competitors show international hiring",
            f"Technology adoption: Average of {sum(len(p.technology_stack) if p.technology_stack else 0 for p in intelligence.profiles) / len(intelligence.profiles):.1f} technologies per competitor"
        ]
        
        for dynamic in dynamics:
            content.append(Paragraph(f"• {dynamic}", normal_style))
        
        return content
    
    async def _create_competitor_profile_section(self, profile, heading_style, subheading_style, normal_style) -> List:
        """Create individual competitor profile section"""
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        content = []
        content.append(Paragraph(f"Competitor Profile: {profile.name}", heading_style))
        
        # Basic information table
        basic_info = [
            ['Attribute', 'Value'],
            ['Website', profile.website],
            ['Threat Level', profile.threat_level.value.title() if profile.threat_level else 'Medium'],
            ['Target Markets', ', '.join(profile.target_markets) if profile.target_markets else 'Unknown'],
            ['Last Analyzed', profile.last_analyzed or 'N/A']
        ]
        
        info_table = Table(basic_info, colWidths=[1.5*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        content.append(info_table)
        content.append(Spacer(1, 0.2*inch))
        
        # Funding information
        if profile.funding_info:
            content.append(Paragraph("Funding Overview", subheading_style))
            funding_text = f"""
            <b>Total Funding:</b> {profile.funding_info.total_funding or 'Unknown'}<br/>
            <b>Last Round:</b> {profile.funding_info.last_round_amount or 'Unknown'} 
            ({profile.funding_info.last_round_type or 'Unknown'}) in {profile.funding_info.last_round_date or 'Unknown'}<br/>
            <b>Valuation:</b> {profile.funding_info.valuation or 'Unknown'}
            """
            content.append(Paragraph(funding_text, normal_style))
            content.append(Spacer(1, 0.1*inch))
        
        # Key features
        if profile.key_features:
            content.append(Paragraph("Key Features & Capabilities", subheading_style))
            for feature in profile.key_features[:10]:  # Limit to top 10
                content.append(Paragraph(f"• {feature}", normal_style))
            content.append(Spacer(1, 0.1*inch))
        
        # Technology stack
        if profile.technology_stack:
            content.append(Paragraph("Technology Stack", subheading_style))
            tech_text = ', '.join(profile.technology_stack)
            content.append(Paragraph(tech_text, normal_style))
            content.append(Spacer(1, 0.1*inch))
        
        # Recent activity summary
        content.append(Paragraph("Recent Activity", subheading_style))
        activity_items = []
        
        if profile.job_postings:
            activity_items.append(f"{len(profile.job_postings)} active job postings")
        if profile.recent_news:
            activity_items.append(f"{len(profile.recent_news)} recent news mentions")
        if profile.github_activity and profile.github_activity.public_repos:
            activity_items.append(f"{profile.github_activity.public_repos} public repositories")
        if profile.case_studies:
            activity_items.append(f"{len(profile.case_studies)} customer case studies")
        
        if activity_items:
            for item in activity_items:
                content.append(Paragraph(f"• {item}", normal_style))
        else:
            content.append(Paragraph("Limited recent activity data available", normal_style))
        
        return content
    
    async def _create_recommendations_section(self, intelligence, heading_style, subheading_style, normal_style) -> List:
        """Create strategic recommendations section"""
        from reportlab.platypus import Paragraph, Spacer
        from reportlab.lib.units import inch
        
        content = []
        content.append(Paragraph("Strategic Recommendations", heading_style))
        
        # Generate recommendations
        recommendations = await self._generate_recommendations_content(intelligence)
        
        # Parse and format recommendations
        if 'IMMEDIATE ACTIONS' in recommendations:
            sections = recommendations.split('IMMEDIATE ACTIONS')[1].split('SHORT-TERM')[0] if 'SHORT-TERM' in recommendations else recommendations.split('IMMEDIATE ACTIONS')[1]
            content.append(Paragraph("Immediate Actions (0-90 days)", subheading_style))
            for line in sections.split('\n'):
                line = line.strip()
                if line and line.startswith('•'):
                    content.append(Paragraph(line, normal_style))
            content.append(Spacer(1, 0.1*inch))
        
        if 'SHORT-TERM' in recommendations:
            sections = recommendations.split('SHORT-TERM')[1].split('LONG-TERM')[0] if 'LONG-TERM' in recommendations else recommendations.split('SHORT-TERM')[1]
            content.append(Paragraph("Short-term Initiatives (3-6 months)", subheading_style))
            for line in sections.split('\n'):
                line = line.strip()
                if line and line.startswith('•'):
                    content.append(Paragraph(line, normal_style))
            content.append(Spacer(1, 0.1*inch))
        
        if 'LONG-TERM' in recommendations:
            sections = recommendations.split('LONG-TERM')[1].split('CONTINUOUS')[0] if 'CONTINUOUS' in recommendations else recommendations.split('LONG-TERM')[1]
            content.append(Paragraph("Long-term Strategy (6-12 months)", subheading_style))
            for line in sections.split('\n'):
                line = line.strip()
                if line and line.startswith('•'):
                    content.append(Paragraph(line, normal_style))
        
        return content
    
    def _create_appendix(self, intelligence, heading_style, subheading_style, normal_style) -> List:
        """Create appendix section"""
        from reportlab.platypus import Paragraph, Table, TableStyle
        from reportlab.lib import colors
        
        content = []
        content.append(Paragraph("Appendix", heading_style))
        
        # Data sources used
        content.append(Paragraph("Data Sources", subheading_style))
        if intelligence.metadata and 'data_sources_used' in intelligence.metadata:
            sources = intelligence.metadata['data_sources_used']
            for source in sources:
                content.append(Paragraph(f"• {source.title()}", normal_style))
        
        # Analysis configuration
        content.append(Paragraph("Analysis Configuration", subheading_style))
        if intelligence.analysis_config:
            config_data = [
                ['Parameter', 'Value'],
                ['Analysis Depth', intelligence.analysis_config.depth_level.value.title()],
                ['Competitors Analyzed', ', '.join(intelligence.analysis_config.competitors)],
                ['Output Formats', ', '.join(intelligence.analysis_config.output_formats)],
                ['Analysis Date', intelligence.analysis_date]
            ]
            
            config_table = Table(config_data)
            config_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            content.append(config_table)
        
        return content
    
    # Helper methods
    def _get_funding_status(self, profile) -> str:
        """Get funding status for threat matrix"""
        if not profile.funding_info or not profile.funding_info.total_funding:
            return 'Unknown'
        
        funding = profile.funding_info.total_funding
        if 'B' in funding:
            return 'Well-funded (Billion+)'
        elif 'M' in funding:
            try:
                amount = float(funding.replace('$', '').replace('M', '').replace(',', ''))
                if amount > 200:
                    return 'Well-funded (200M+)'
                elif amount > 50:
                    return 'Funded (50M+)'
                else:
                    return 'Early-stage'
            except:
                return 'Funded'
        return 'Early-stage'
    
    def _get_key_strengths(self, profile) -> str:
        """Get key strengths for threat matrix"""
        strengths = []
        
        if profile.key_features and len(profile.key_features) > 15:
            strengths.append("Comprehensive features")
        
        if profile.case_studies and len(profile.case_studies) > 8:
            strengths.append("Strong customer base")
        
        if profile.funding_info and profile.funding_info.total_funding and 'B' in profile.funding_info.total_funding:
            strengths.append("Well-funded")
        
        if profile.technology_stack:
            ai_tech = [t for t in profile.technology_stack if t in ['AI', 'Machine Learning']]
            if ai_tech:
                strengths.append("AI capabilities")
        
        return '; '.join(strengths[:2]) if strengths else 'Standard offering'
    
    def _get_primary_concerns(self, profile) -> str:
        """Get primary concerns for threat matrix"""
        concerns = []
        
        if profile.threat_level and profile.threat_level.value in ['high', 'critical']:
            concerns.append("Direct competitive threat")
        
        if profile.job_postings and len(profile.job_postings) > 15:
            concerns.append("Aggressive hiring")
        
        if profile.recent_news and len(profile.recent_news) > 5:
            concerns.append("High market activity")
        
        if profile.funding_info and profile.funding_info.last_round_date:
            from datetime import datetime
            try:
                last_round = datetime.fromisoformat(profile.funding_info.last_round_date.replace('Z', '+00:00'))
                if (datetime.now() - last_round).days < 365:
                    concerns.append("Recent funding")
            except:
                pass
        
        return '; '.join(concerns[:2]) if concerns else 'Moderate activity'
    
    async def _generate_executive_summary_content(self, intelligence) -> str:
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
    
    async def _generate_recommendations_content(self, intelligence) -> str:
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
            summary = await self._generate_executive_summary_content(intelligence)
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
            recommendations = await self._generate_recommendations_content(intelligence)
            f.write(recommendations + "\n")
        
        logger.info(f"Text report generated: {filepath}")
        return str(filepath):
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
    
    async def _generate_market_analysis_content(self, intelligence) -> str:
        """Generate market analysis content"""
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
            
        except Exception as e