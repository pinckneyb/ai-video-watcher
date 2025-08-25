"""
PDF Report Generator for Surgical VOP Assessments
Creates professional assessment reports for surgical residents.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

class SurgicalVOPReportGenerator:
    """Generates professional PDF reports for surgical VOP assessments."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom styles for the report."""
        
        # Title style
        self.styles.add(ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.darkblue
        ))
        
        # Assessment point style
        self.styles.add(ParagraphStyle(
            'AssessmentPoint',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceBefore=8,
            spaceAfter=8,
            leftIndent=20
        ))
        
        # Score style
        self.styles.add(ParagraphStyle(
            'ScoreStyle',
            parent=self.styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            textColor=colors.darkred
        ))
    
    def generate_vop_report(
        self, 
        assessment_data: Dict[str, Any], 
        rubric_scores: Dict[int, int],
        overall_result: Dict[str, Any],
        output_filename: str
    ) -> str:
        """
        Generate a comprehensive VOP assessment report.
        
        Args:
            assessment_data: Analysis results from the video assessment
            rubric_scores: Manual scores for each rubric point
            overall_result: Overall pass/fail result
            output_filename: Path for the output PDF file
            
        Returns:
            str: Path to the generated PDF file
        """
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_filename,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build the report content
        story = []
        
        # Header
        story.extend(self._create_header(assessment_data))
        
        # Video information
        story.extend(self._create_video_info_section(assessment_data))
        
        # Overall result
        story.extend(self._create_overall_result_section(overall_result))
        
        # Detailed rubric assessment
        story.extend(self._create_rubric_assessment_section(rubric_scores, assessment_data))
        
        # Technical analysis
        story.extend(self._create_technical_analysis_section(assessment_data))
        
        # Recommendations
        story.extend(self._create_recommendations_section(assessment_data, overall_result))
        
        # Footer
        story.extend(self._create_footer())
        
        # Build PDF
        doc.build(story)
        
        return output_filename
    
    def _create_header(self, assessment_data: Dict[str, Any]) -> List:
        """Create the report header."""
        story = []
        
        # Institution header
        story.append(Paragraph("SURGICAL VERIFICATION OF PROFICIENCY", self.styles['CustomTitle']))
        story.append(Paragraph("Suturing Technique Assessment Report", self.styles['Heading2']))
        story.append(Spacer(1, 20))
        
        # Assessment info table
        pattern_name = assessment_data['video_info']['pattern'].replace('_', ' ').title()
        assessment_date = datetime.now().strftime("%B %d, %Y")
        
        header_data = [
            ['Suture Pattern:', pattern_name],
            ['Assessment Date:', assessment_date],
            ['Video File:', assessment_data['video_info']['filename']],
            ['Analysis FPS:', f"{assessment_data['video_info']['fps']} frames/second"]
        ]
        
        header_table = Table(header_data, colWidths=[2*inch, 4*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(header_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_video_info_section(self, assessment_data: Dict[str, Any]) -> List:
        """Create video information section."""
        story = []
        
        story.append(Paragraph("Video Analysis Summary", self.styles['CustomHeading']))
        
        total_frames = assessment_data['video_info']['total_frames']
        fps = assessment_data['video_info']['fps']
        duration = total_frames / fps if fps > 0 else 0
        
        info_text = f"""
        This assessment analyzed {total_frames} frames extracted at {fps} FPS, 
        covering approximately {duration:.1f} seconds of surgical technique. 
        The video was processed using advanced AI analysis to evaluate specific 
        technical competencies according to institutional VOP standards.
        """
        
        story.append(Paragraph(info_text, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        return story
    
    def _create_overall_result_section(self, overall_result: Dict[str, Any]) -> List:
        """Create overall assessment result section."""
        story = []
        
        story.append(Paragraph("Assessment Result", self.styles['CustomHeading']))
        
        # Result box
        if overall_result['pass']:
            result_text = f"<b>PASS</b> - Average Score: {overall_result['average_score']:.1f}/5.0"
            result_color = colors.darkgreen
        else:
            result_text = f"<b>FAIL</b> - {overall_result['reason']}"
            result_color = colors.darkred
        
        result_data = [[Paragraph(result_text, self.styles['ScoreStyle'])]]
        result_table = Table(result_data, colWidths=[6*inch])
        result_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('BOX', (0, 0), (-1, -1), 2, result_color),
            ('FONTSIZE', (0, 0), (-1, -1), 14),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        story.append(result_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_rubric_assessment_section(self, rubric_scores: Dict[int, int], assessment_data: Dict[str, Any]) -> List:
        """Create detailed rubric assessment section."""
        story = []
        
        story.append(Paragraph("Detailed Rubric Assessment", self.styles['CustomHeading']))
        
        # Load rubric data to get point descriptions
        try:
            with open("unified_rubric.JSON", 'r') as f:
                rubric_data = json.load(f)
            
            pattern_id = assessment_data['video_info']['pattern']
            pattern_data = None
            for pattern in rubric_data.get("patterns", []):
                if pattern["id"] == pattern_id:
                    pattern_data = pattern
                    break
            
            if pattern_data:
                # Create rubric table
                table_data = [['Point', 'Criterion', 'Critical', 'Score', 'Assessment']]
                
                for point in pattern_data["points"]:
                    pid = point["pid"]
                    score = rubric_scores.get(pid, 3)
                    critical = "Yes" if point.get("critical", False) else "No"
                    
                    # Score color coding
                    if score >= 4:
                        score_color = colors.darkgreen
                    elif score >= 3:
                        score_color = colors.orange
                    else:
                        score_color = colors.darkred
                    
                    table_data.append([
                        str(pid),
                        point['title'],
                        critical,
                        f"{score}/5",
                        self._get_score_interpretation(score)
                    ])
                
                rubric_table = Table(table_data, colWidths=[0.5*inch, 2.5*inch, 0.8*inch, 0.6*inch, 1.6*inch])
                rubric_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(rubric_table)
        
        except Exception as e:
            story.append(Paragraph(f"Error loading rubric details: {e}", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_technical_analysis_section(self, assessment_data: Dict[str, Any]) -> List:
        """Create technical analysis section with timestamps."""
        story = []
        
        story.append(Paragraph("Technical Analysis with Timestamps", self.styles['CustomHeading']))
        
        # Extract key observations from analysis
        for batch_result in assessment_data.get('analysis', []):
            timestamp_range = batch_result.get('timestamp_range', 'Unknown')
            narrative = batch_result.get('narrative', 'No analysis available')
            
            story.append(Paragraph(f"<b>Time Range: {timestamp_range}</b>", self.styles['AssessmentPoint']))
            story.append(Paragraph(narrative, self.styles['Normal']))
            story.append(Spacer(1, 10))
        
        return story
    
    def _create_recommendations_section(self, assessment_data: Dict[str, Any], overall_result: Dict[str, Any]) -> List:
        """Create recommendations section."""
        story = []
        
        story.append(Paragraph("Recommendations for Improvement", self.styles['CustomHeading']))
        
        # Generate recommendations based on scores and analysis
        if overall_result['pass']:
            recommendations = [
                "Continue practicing to maintain proficiency",
                "Consider advancing to more complex suturing techniques",
                "Review any points scored below 4 for optimization opportunities"
            ]
        else:
            recommendations = [
                "Additional practice required before attempting VOP reassessment",
                "Focus on critical technique points that scored below 3",
                "Consider supervised practice sessions with attending physician",
                "Review instructional materials for proper technique"
            ]
        
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_footer(self) -> List:
        """Create report footer."""
        story = []
        
        footer_text = f"""
        <i>This assessment was generated using AI-assisted video analysis technology. 
        Results should be reviewed by qualified surgical faculty before final determination. 
        Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}.</i>
        """
        
        story.append(Paragraph(footer_text, self.styles['Normal']))
        
        return story
    
    def _get_score_interpretation(self, score: int) -> str:
        """Get text interpretation of numeric score."""
        interpretations = {
            1: "Unacceptable",
            2: "Poor",
            3: "Adequate",
            4: "Good", 
            5: "Excellent"
        }
        return interpretations.get(score, "Unknown")

# Usage example
def generate_sample_report():
    """Generate a sample report for testing."""
    
    sample_data = {
        'video_info': {
            'filename': 'sample_suturing_video.mp4',
            'pattern': 'simple_interrupted',
            'fps': 2.0,
            'total_frames': 120
        },
        'analysis': [
            {
                'timestamp_range': '00:00:00 - 00:00:30',
                'narrative': 'Initial needle entry demonstrates proper angle and tissue handling...'
            }
        ]
    }
    
    sample_scores = {1: 4, 2: 3, 3: 4, 4: 3, 5: 4, 6: 3, 7: 4}
    sample_result = {'pass': True, 'average_score': 3.6}
    
    generator = SurgicalVOPReportGenerator()
    return generator.generate_vop_report(
        sample_data, 
        sample_scores, 
        sample_result, 
        'sample_vop_report.pdf'
    )
