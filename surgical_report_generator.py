"""
PDF Report Generator for Surgical VOP Assessments
Creates professional assessment reports for surgical residents.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import os
from PIL import Image as PILImage

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
        
        # Overall assessment and summative feedback
        story.extend(self._create_summative_assessment_section(overall_result, assessment_data))
        
        # Page break before image comparison
        story.append(PageBreak())
        
        # Image comparison section
        story.extend(self._create_image_comparison_section(assessment_data))
        
        # New page for detailed narrative analysis
        story.append(PageBreak())
        
        # Detailed narrative analysis section
        story.extend(self._create_detailed_narrative_section(assessment_data))
        
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
        
        # Minimal assessment info
        video_info = assessment_data['video_info']
        pattern_name = video_info['pattern'].replace('_', ' ').title()
        assessment_date = datetime.now().strftime("%B %d, %Y")
        
        header_data = [
            ['Suture Pattern:', pattern_name],
            ['Assessment Date:', assessment_date]
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
        """Remove verbose video info section."""
        return []
    
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
        
        story.append(Paragraph("Assessment Results", self.styles['CustomHeading']))
        story.append(Spacer(1, 10))
        
        # Load rubric data and enhanced narrative for detailed assessments
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
                enhanced_narrative = assessment_data.get('enhanced_narrative', '')
                
                # Extract assessment paragraphs from enhanced narrative for each rubric point
                for point in pattern_data["points"]:
                    pid = point["pid"]
                    score = rubric_scores.get(pid, 3)
                    score_text = self._get_score_interpretation(score)
                    
                    # Create professional rubric point assessment
                    story.append(Paragraph(f"<b>{pid}. {point['title']}</b>", self.styles['Normal']))
                    story.append(Spacer(1, 6))
                    
                    # Score with interpretation
                    score_para = f"<b>Score: {score}/5 ({score_text})</b>"
                    story.append(Paragraph(score_para, self.styles['Normal']))
                    story.append(Spacer(1, 6))
                    
                    # Create cogent assessment with actionable advice
                    # This will be populated from the enhanced narrative parsing
                    feedback_text = self._extract_rubric_feedback(enhanced_narrative, pid, point['title'], score)
                    
                    story.append(Paragraph(feedback_text, self.styles['Normal']))
                    story.append(Spacer(1, 12))
        
        except Exception as e:
            story.append(Paragraph(f"Error loading rubric details: {e}", self.styles['Normal']))
        
        return story
    
    def _extract_rubric_feedback(self, enhanced_narrative: str, pid: int, title: str, score: int) -> str:
        """Extract specific rubric point assessment from GPT-5 enhanced narrative."""
        if not enhanced_narrative:
            return f"Summary: Detailed assessment requires enhanced narrative from GPT-5 analysis of complete video."
        
        # The GPT-5 enhanced narrative should contain assessments for each rubric point
        # Look for content structured by rubric point numbers or titles
        lines = enhanced_narrative.split('\n')
        
        # Strategy 1: Look for explicit rubric point sections
        rubric_content = []
        collecting = False
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Check if this line starts a rubric point section
            if (line_clean.startswith(f"{pid}.") or 
                line_clean.startswith(f"Point {pid}") or
                line_clean.startswith(f"Rubric {pid}") or
                (str(pid) in line_clean and title.lower() in line_clean.lower())):
                collecting = True
                rubric_content = [line_clean]
                continue
            
            # If we're collecting and hit another rubric point, stop
            elif collecting and (any(line_clean.startswith(f"{j}.") for j in range(1, 8) if j != pid) or
                                line_clean.startswith("Point ") or
                                line_clean.startswith("Rubric ") or
                                line_clean.startswith("RUBRIC_SCORES")):
                break
            
            # If collecting, add substantial content
            elif collecting and len(line_clean) > 10:
                rubric_content.append(line_clean)
        
        # If we found specific rubric content, use it
        if rubric_content and len(rubric_content) > 1:
            # Combine the content, excluding the header line
            content = " ".join(rubric_content[1:])
            if len(content) > 300:
                content = content[:300] + "..."
            return f"Summary: {content}"
        
        # Strategy 2: Look for content related to this rubric point by keywords
        keywords = {
            1: ['needle', 'perpendicular', 'angle', 'entry', 'passes', '90'],
            2: ['tissue', 'handling', 'gentle', 'forceps', 'grasp', 'manipulation'],
            3: ['knot', 'square', 'secure', 'tie', 'throw', 'tension'],
            4: ['approximation', 'tension', 'edge', 'contact', 'gap', 'alignment'],
            5: ['spacing', 'even', 'uniform', 'distance', 'interval', 'cm'],
            6: ['eversion', 'edge', 'inversion', 'position', 'roll', 'flat'],
            7: ['motion', 'efficiency', 'movement', 'economy', 'smooth', 'hands']
        }.get(pid, [title.lower()])
        
        # Find paragraphs with relevant keywords
        relevant_paras = []
        current_para = []
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                if current_para and any(any(keyword in sentence.lower() for keyword in keywords) 
                                      for sentence in current_para):
                    relevant_paras.append(" ".join(current_para))
                current_para = []
            else:
                current_para.append(line_clean)
        
        # Check last paragraph
        if current_para and any(any(keyword in sentence.lower() for keyword in keywords) 
                              for sentence in current_para):
            relevant_paras.append(" ".join(current_para))
        
        # Return the most relevant paragraph
        if relevant_paras:
            # Pick the longest/most substantial paragraph
            best_para = max(relevant_paras, key=len)
            if len(best_para) > 300:
                best_para = best_para[:300] + "..."
            return f"Summary: {best_para}"
        
        # Strategy 3: Extract general procedural content
        substantial_paras = []
        current_para = []
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                if current_para and len(" ".join(current_para)) > 50:
                    para_text = " ".join(current_para)
                    if not any(skip in para_text.lower() for skip in ['rubric_scores', 'score:', 'overall']):
                        substantial_paras.append(para_text)
                current_para = []
            else:
                if not line_clean.startswith(('RUBRIC', 'Score:')):
                    current_para.append(line_clean)
        
        # Check last paragraph
        if current_para and len(" ".join(current_para)) > 50:
            para_text = " ".join(current_para)
            if not any(skip in para_text.lower() for skip in ['rubric_scores', 'score:', 'overall']):
                substantial_paras.append(para_text)
        
        if substantial_paras:
            # Take the first substantial paragraph as general assessment
            para = substantial_paras[0]
            if len(para) > 300:
                para = para[:300] + "..."
            return f"Summary: {para}"
        
        return f"Summary: Assessment of {title.lower()} based on video analysis shows {self._get_score_interpretation(score).lower()} performance level."
    
    def _create_summative_assessment_section(self, overall_result: Dict[str, Any], assessment_data: Dict[str, Any]) -> List:
        """Create summative assessment with final score and holistic feedback."""
        story = []
        
        story.append(Paragraph("Final Assessment", self.styles['CustomHeading']))
        story.append(Spacer(1, 10))
        
        # VOP-aligned final score with Likert scale + adjective
        avg_score = overall_result.get('average_score', 3.0)
        score_text = self._get_score_interpretation(round(avg_score))
        
        story.append(Paragraph(f"<b>Final Score: {round(avg_score)} - {score_text}</b>", self.styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Create longer, actionable summative feedback
        scores = assessment_data.get('extracted_scores', {})
        avg_score = overall_result.get('average_score', 3.0)
        
        # Generate summative feedback based on performance patterns
        summative_feedback = self._generate_summative_feedback(scores, avg_score, assessment_data)
        
        story.append(Paragraph("<b>Summative Comment:</b>", self.styles['Normal']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(summative_feedback.strip(), self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_technical_analysis_section(self, assessment_data: Dict[str, Any]) -> List:
        """Removed - keeping reports concise."""
        return []
    
    def _create_recommendations_section(self, assessment_data: Dict[str, Any], overall_result: Dict[str, Any]) -> List:
        """Removed - recommendations included in summative feedback."""
        return []
    
    def _create_footer(self) -> List:
        """Create report footer."""
        story = []
        
        footer_text = f"""
        <i>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}.</i>
        """
        
        story.append(Paragraph(footer_text, self.styles['Normal']))
        
        return story
    
    def _get_score_interpretation(self, score: int) -> str:
        """Get VOP-aligned text interpretation of numeric score."""
        interpretations = {
            1: "Remediation / Unsafe",
            2: "Minimal Pass / Basic Competent",
            3: "Developing Pass / Generally Reliable",
            4: "Proficient", 
            5: "Exemplary / Model"
        }
        return interpretations.get(score, "Unknown")
    
    def _create_image_comparison_section(self, assessment_data: Dict[str, Any]) -> List:
        """Create side-by-side comparison with gold standard image."""
        story = []
        
        try:
            story.append(Paragraph("Visual Comparison", self.styles['CustomHeading']))
            
            pattern_id = assessment_data['video_info']['pattern']
            
            # Map pattern IDs to gold standard images
            gold_standard_images = {
                'simple_interrupted': 'Simple_Interrupted_Suture_example.png',
                'vertical_mattress': 'Vertical_Mattress_Suture_example.png',
                'subcuticular': 'subcuticular_example.png'
            }
            
            gold_standard_path = gold_standard_images.get(pattern_id)
            
            if gold_standard_path and os.path.exists(gold_standard_path):
                # Create comparison table
                story.append(Paragraph("Side-by-Side Comparison: Gold Standard vs. Learner Performance", self.styles['Normal']))
                story.append(Spacer(1, 10))
                
                # Calculate image dimensions to make them equal height
                target_height = 2.5 * inch  # Target height for both images
                
                # Add gold standard image
                try:
                    gold_img = Image(gold_standard_path)
                    # Get original dimensions
                    pil_img = PILImage.open(gold_standard_path)
                    original_width, original_height = pil_img.size
                    
                    # Calculate width to maintain aspect ratio
                    aspect_ratio = original_width / original_height
                    gold_width = target_height * aspect_ratio
                    
                    gold_img.drawHeight = target_height
                    gold_img.drawWidth = gold_width
                    
                    # Create table with images side by side
                    image_data = [
                        ['Gold Standard', 'Learner Performance'],
                        [gold_img, Paragraph("Final frame from analyzed video would appear here in actual implementation", self.styles['Normal'])]
                    ]
                    
                    image_table = Table(image_data, colWidths=[3*inch, 3*inch])
                    image_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ]))
                    
                    story.append(image_table)
                    story.append(Spacer(1, 10))
                    
                    # Add explanation
                    explanation = f"""
                    The gold standard image above represents the ideal final result for {pattern_id.replace('_', ' ').title()} 
                    suturing technique. Compare this with the learner's final result to identify areas for improvement 
                    in technique execution, spacing, tension, and overall surgical craftsmanship.
                    """
                    story.append(Paragraph(explanation, self.styles['Normal']))
                    
                except Exception as img_error:
                    story.append(Paragraph(f"Error loading gold standard image: {img_error}", self.styles['Normal']))
            else:
                story.append(Paragraph(f"Gold standard image not available for {pattern_id}", self.styles['Normal']))
            
        except Exception as e:
            story.append(Paragraph(f"Error creating image comparison: {e}", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _generate_summative_feedback(self, scores: Dict[int, int], avg_score: float, assessment_data: Dict[str, Any]) -> str:
        """Extract summative feedback from AI-generated enhanced narrative."""
        enhanced_narrative = assessment_data.get('enhanced_narrative', '')
        
        if not enhanced_narrative:
            return "Comprehensive summative feedback requires enhanced narrative generation from AI analysis."
        
        # Look for conclusion, summary, or assessment sections in the narrative
        lines = enhanced_narrative.split('\n')
        summative_content = []
        
        # Find paragraphs that seem like conclusions or overall assessments
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Look for conclusion-type content
            if any(keyword in line_lower for keyword in ['overall', 'summary', 'conclusion', 'assessment', 'competency', 'performance']):
                # Take this line and a few following lines
                for j in range(i, min(i + 3, len(lines))):
                    if lines[j].strip() and len(lines[j].strip()) > 20:
                        summative_content.append(lines[j].strip())
                break
        
        # If no specific conclusion found, take the last substantial paragraphs
        if not summative_content:
            substantial_lines = [line.strip() for line in lines 
                               if len(line.strip()) > 50 and 
                               not line.strip().startswith(('RUBRIC', '1:', '2:', '3:', '4:', '5:', '6:', '7:')) and
                               not any(char.isdigit() and ':' in line[:10] for char in line[:10])]
            
            # Take last few substantial lines as summative content
            if substantial_lines:
                summative_content = substantial_lines[-2:] if len(substantial_lines) > 1 else substantial_lines[-1:]
        
        # Combine the summative content
        if summative_content:
            return ' '.join(summative_content)
        
        # Final fallback - extract any meaningful content from the middle of the narrative
        meaningful_lines = [line.strip() for line in lines 
                          if len(line.strip()) > 40 and 
                          not line.strip().startswith(('RUBRIC', 'Score:')) and
                          not any(str(i) + ':' in line for i in range(1, 8))]
        
        if meaningful_lines:
            # Take a representative sample from the middle
            mid_point = len(meaningful_lines) // 2
            return meaningful_lines[mid_point] if meaningful_lines else "AI-generated summative feedback not available."
        
        return "Enhanced narrative analysis required for detailed summative feedback."
    
    def _create_detailed_narrative_section(self, assessment_data: Dict[str, Any]) -> List:
        """Create detailed narrative analysis with timestamps and actionable advice."""
        story = []
        
        story.append(Paragraph("Video Analysis Narrative", self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Enhanced narrative content
        enhanced_narrative = assessment_data.get('enhanced_narrative', '')
        
        if enhanced_narrative:
            # Clean up and format the narrative
            narrative_text = enhanced_narrative.replace('RUBRIC_SCORES_START', '').replace('RUBRIC_SCORES_END', '')
            
            # Remove scoring section
            lines = narrative_text.split('\n')
            cleaned_lines = []
            skip_scoring = False
            
            for line in lines:
                if ':' in line and len(line.strip()) < 10 and any(char.isdigit() for char in line):
                    skip_scoring = True
                    continue
                if skip_scoring and not line.strip():
                    skip_scoring = False
                    continue
                if not skip_scoring and line.strip():
                    cleaned_lines.append(line.strip())
            
            # Format as narrative paragraphs
            narrative_content = ' '.join(cleaned_lines)
            paragraphs = narrative_content.split('. ')
            
            current_paragraph = ""
            for i, sentence in enumerate(paragraphs):
                if sentence.strip():
                    current_paragraph += sentence + ". "
                
                    # Create paragraph breaks every 3-4 sentences
                    if (i + 1) % 3 == 0 or i == len(paragraphs) - 1:
                        if current_paragraph.strip():
                            story.append(Paragraph(current_paragraph.strip(), self.styles['Normal']))
                            story.append(Spacer(1, 12))
                        current_paragraph = ""
        else:
            story.append(Paragraph("Detailed narrative analysis not available. Enhanced narrative generation required for comprehensive assessment.", self.styles['Normal']))
        
        return story

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
