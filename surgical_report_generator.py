"""
PDF Report Generator for Surgical VOP Assessments
Creates professional assessment reports for surgical residents.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.platypus.flowables import KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import os
import io
import re
import numpy as np
import cv2
from PIL import Image

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
        
        # Visual references after summative comment
        story.append(PageBreak())
        story.extend(self._create_visual_references_after_summative(assessment_data))
        
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
        
        # Result box - just score and adjective
        score = overall_result['average_score']
        if score >= 4.5:
            adjective = "excellent"
        elif score >= 3.5:
            adjective = "proficient"
        elif score >= 2.5:
            adjective = "competent"
        elif score >= 1.5:
            adjective = "developing"
        else:
            adjective = "inadequate"
            
        result_text = f"<b>{score:.1f} - {adjective}</b>"
        result_color = colors.black
        
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
        
        # Handle enhanced_narrative as dictionary or string
        if isinstance(enhanced_narrative, dict):
            # Use full_response only; avoid including summative content in rubric points
            narrative_text = enhanced_narrative.get("full_response", "")
        else:
            narrative_text = enhanced_narrative
        
        # The GPT-5 enhanced narrative should contain assessments for each rubric point
        # Look for content structured by rubric point numbers or titles
        lines = narrative_text.split('\n')
        
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
                                line_clean.startswith("RUBRIC_SCORES") or
                                'summative' in line_clean.lower() or
                                'final assessment' in line_clean.lower()):
                break
            
            # If collecting, add substantial content
            elif collecting and len(line_clean) > 10:
                rubric_content.append(line_clean)
        
        # If we found specific rubric content, use it
        if rubric_content and len(rubric_content) > 1:
            # Combine the content, excluding the header line
            content = " ".join(rubric_content[1:])
            # Remove any existing "Summary:" or "summary:" prefixes
            content = content.replace("Summary:", "").replace("summary:", "").strip()
            return content
        
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
            # Remove any existing "Summary:" or "summary:" prefixes
            best_para = best_para.replace("Summary:", "").replace("summary:", "").strip()
            return best_para
        
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
                if not line_clean.startswith(('RUBRIC', 'Score:')) and ('summative' not in line_clean.lower()):
                    current_para.append(line_clean)
        
        # Check last paragraph
        if current_para and len(" ".join(current_para)) > 50:
            para_text = " ".join(current_para)
            if not any(skip in para_text.lower() for skip in ['rubric_scores', 'score:', 'overall']):
                substantial_paras.append(para_text)
        
        if substantial_paras:
            # Take the first substantial paragraph as general assessment
            para = substantial_paras[0]
            # Remove any existing "Summary:" or "summary:" prefixes
            para = para.replace("Summary:", "").replace("summary:", "").strip()
            return para
        
        return f"Assessment of {title.lower()} based on video analysis shows {self._get_score_interpretation(score).lower()} performance level."
    
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
        
        # Format summative feedback into paragraphs
        formatted_feedback = self._format_summative_paragraphs(summative_feedback.strip())
        story.append(Paragraph(formatted_feedback, self.styles['Normal']))
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
    
    def _create_visual_references_after_summative(self, assessment_data: Dict[str, Any]) -> List:
        """Place learner final product (unaltered) full-width, then gold standard with header kept on same page."""
        story = []
        
        try:
            story.append(Paragraph("Visual Assessment", self.styles['CustomHeading']))
            story.append(Spacer(1, 10))
            
            pattern_id = assessment_data['video_info']['pattern']
            
            # Map pattern IDs to gold standard images
            gold_standard_images = {
                'simple_interrupted': 'Simple_Interrupted_Suture_example.png',
                'vertical_mattress': 'Vertical_Mattress_Suture_example.png',
                'subcuticular': 'subcuticular_example.png'
            }
            
            gold_standard_path = gold_standard_images.get(pattern_id)
            
            # Learner final product (unaltered) full page width
            learner_header = Paragraph("<b>Learner Final Product</b>", self.styles['Normal'])
            learner_img = self._extract_final_product_image_enhanced_full(assessment_data, 6*inch)
            story.append(KeepTogether([learner_header, Spacer(1, 6), learner_img, Spacer(1, 20)]))
            
            # Add gold standard image below
            if gold_standard_path and os.path.exists(gold_standard_path):
                gold_header = Paragraph("<b>Gold Standard Reference</b>", self.styles['Normal'])
                try:
                    gold_img = RLImage(gold_standard_path)
                    pil_img = Image.open(gold_standard_path)
                    original_width, original_height = pil_img.size
                    aspect_ratio = original_width / original_height
                    gold_width = 6 * inch
                    gold_height = gold_width / aspect_ratio
                    gold_img.drawHeight = gold_height
                    gold_img.drawWidth = gold_width
                    story.append(KeepTogether([gold_header, Spacer(1, 6), gold_img, Spacer(1, 20)]))
                except Exception as e:
                    story.append(Paragraph(f"Error loading gold standard image: {e}", self.styles['Normal']))
            else:
                story.append(Paragraph(f"Gold standard image not found for pattern: {pattern_id}", self.styles['Normal']))
            
        except Exception as e:
            story.append(Paragraph(f"Error creating image comparison: {e}", self.styles['Normal']))
        
        return story

    def _extract_final_product_image_enhanced_full(self, assessment_data: Dict[str, Any], target_width: float):
        """Select best final product frame using enhanced algorithm but return full unaltered image."""
        try:
            video_path = assessment_data.get('video_path')
            if not video_path:
                return Paragraph("Video not available for final product extraction", self.styles['Normal'])
            
            from video_processor import VideoProcessor
            import cv2
            
            processor = VideoProcessor()
            success = processor.load_video(video_path)
            
            if not success:
                return Paragraph("Could not load video for final product extraction", self.styles['Normal'])
            
            # PROPER SOLUTION: Multi-tier sampling with enhanced selection for clean final product
            duration = processor.duration
            candidate_frames = []
            
            # Tier 1: Last 1% of video (most likely to show final product)
            self._sample_video_segment_enhanced(processor, duration * 0.99, duration, 20, candidate_frames, "final_1pct")
            
            # Tier 2: Last 3% of video
            self._sample_video_segment_enhanced(processor, duration * 0.97, duration * 0.99, 15, candidate_frames, "final_3pct")
            
            # Tier 3: Last 5% of video (backup)
            self._sample_video_segment_enhanced(processor, duration * 0.95, duration * 0.97, 10, candidate_frames, "final_5pct")
            
            if not candidate_frames:
                return Paragraph("No frames could be extracted from final portion of video", self.styles['Normal'])
            
            # Use AI semantic understanding to select best final product image
            # Pass API key through assessment data
            assessment_data_with_key = assessment_data.copy()
            if 'api_key' not in assessment_data_with_key:
                assessment_data_with_key['api_key'] = os.getenv('OPENAI_API_KEY')
            
            best_frame, best_timestamp = self._ai_select_final_product_frame(candidate_frames, assessment_data_with_key)
            
            if best_frame is None:
                return Paragraph("AI could not select suitable final product frame", self.styles['Normal'])
            
            # Convert to PIL Image - FULL IMAGE, NO CROPPING
            pil_image = Image.fromarray(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB))
            
            # Calculate dimensions maintaining aspect ratio
            aspect_ratio = pil_image.height / pil_image.width
            target_height = int(target_width * aspect_ratio)
            
            # Resize to target width while maintaining aspect ratio
            resized_image = pil_image.resize((int(target_width), target_height), Image.Resampling.LANCZOS)
            
            # Convert to bytes for ReportLab
            img_buffer = io.BytesIO()
            resized_image.save(img_buffer, format='JPEG', quality=95)
            img_buffer.seek(0)
            
            return RLImage(img_buffer, width=target_width, height=target_height)
            
        except Exception as e:
            return Paragraph(f"Error extracting enhanced final product image: {str(e)}", self.styles['Normal'])

    def _extract_final_product_image_raw(self, assessment_data: Dict[str, Any], target_width: float):
        """Select a frame near the end and return unaltered image scaled to target width."""
        try:
            video_path = assessment_data.get('video_path')
            if not video_path:
                return Paragraph("Video not available for final product extraction", self.styles['Normal'])
            from video_processor import VideoProcessor
            import cv2
            vp = VideoProcessor()
            if not vp.load_video(video_path):
                return Paragraph("Could not load video for final product extraction", self.styles['Normal'])
            # Choose a frame near the end (last 5%)
            duration = vp.duration
            start = max(0.0, duration * 0.95)
            end = duration
            cap = cv2.VideoCapture(vp.video_path)
            if not cap.isOpened():
                return Paragraph("Could not open video", self.styles['Normal'])
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_idx = int(((start + end) / 2.0) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return Paragraph("Could not read frame for final product", self.styles['Normal'])
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ar = pil.height / pil.width
            target_height = int(target_width * ar)
            resized = pil.resize((int(target_width), target_height), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, format='JPEG', quality=95)
            buf.seek(0)
            return RLImage(buf, width=target_width, height=target_height)
        except Exception as e:
            return Paragraph(f"Error extracting final product image: {str(e)}", self.styles['Normal'])
    
    def _generate_summative_feedback(self, scores: Dict[int, int], avg_score: float, assessment_data: Dict[str, Any]) -> str:
        """Extract summative feedback from AI-generated enhanced narrative."""
        enhanced_narrative = assessment_data.get('enhanced_narrative', '')
        
        if not enhanced_narrative:
            return "Comprehensive summative feedback requires enhanced narrative generation from AI analysis."
        
        # Handle enhanced_narrative as dictionary or string
        if isinstance(enhanced_narrative, dict):
            narrative_text = enhanced_narrative.get("summative_assessment", enhanced_narrative.get("full_response", ""))
        else:
            narrative_text = enhanced_narrative
        
        # Look for conclusion, summary, or assessment sections in the narrative
        lines = narrative_text.split('\n')
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
            # Handle enhanced_narrative as dictionary or string
            if isinstance(enhanced_narrative, dict):
                narrative_text = enhanced_narrative.get("summative_assessment", enhanced_narrative.get("full_response", ""))
            else:
                narrative_text = enhanced_narrative
            
            # Clean up and format the narrative
            narrative_text = narrative_text.replace('RUBRIC_SCORES_START', '').replace('RUBRIC_SCORES_END', '')
            
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

    def _extract_final_product_image_with_pad_detection(self, assessment_data: Dict[str, Any], target_width: float):
        """Extract and crop final product image with intelligent suturing pad detection."""
        try:
            # Get video path from assessment data
            video_path = assessment_data.get('video_path')
            if not video_path:
                return Paragraph("Video not available for final product extraction", self.styles['Normal'])
            
            # Import video processor
            from video_processor import VideoProcessor
            import tempfile
            import os
            
            processor = VideoProcessor()
            success = processor.load_video(video_path)
            
            if not success:
                return Paragraph("Could not load video for final product extraction", self.styles['Normal'])
            
            # Multi-tier sampling strategy for best final product image
            duration = processor.duration
            candidate_frames = []
            
            # Tier 1: Last 3% of video (most likely to show final product without hands)
            self._sample_video_segment_enhanced(processor, duration * 0.97, duration, 10, candidate_frames, "final_3pct")
            
            # Tier 2: Last 7% of video (backup)
            self._sample_video_segment_enhanced(processor, duration * 0.93, duration * 0.97, 8, candidate_frames, "final_7pct")
            
            # Tier 3: Last 15% of video (final fallback)
            self._sample_video_segment_enhanced(processor, duration * 0.85, duration * 0.93, 6, candidate_frames, "final_15pct")
            
            if not candidate_frames:
                return Paragraph("No frames could be extracted from final portion of video", self.styles['Normal'])
            
            # Score all frames with enhanced criteria (hand/instrument avoidance)
            scored_frames = []
            for timestamp, frame_data, tier in candidate_frames:
                quality_score = self._score_frame_quality_enhanced(frame_data, tier)
                scored_frames.append((timestamp, frame_data, quality_score, tier))
            
            # Sort by quality score (highest first)
            scored_frames.sort(key=lambda x: x[2], reverse=True)
            
            # Get the best frame
            best_timestamp, best_frame, best_score, best_tier = scored_frames[0]
            
            # Convert to PIL Image and crop
            pil_image = Image.fromarray(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB))
            
            # Intelligent cropping to focus on suturing area
            cropped_image = self._intelligent_crop_suturing_area(pil_image)
            
            # Resize to target width while maintaining aspect ratio
            aspect_ratio = cropped_image.height / cropped_image.width
            target_height = int(target_width * aspect_ratio)
            resized_image = cropped_image.resize((int(target_width), target_height), Image.Resampling.LANCZOS)
            
            # Convert to ReportLab Image
            img_buffer = io.BytesIO()
            resized_image.save(img_buffer, format='JPEG', quality=95)
            img_buffer.seek(0)
            
            reportlab_image = RLImage(img_buffer, width=target_width, height=target_height)
            
            return reportlab_image
            
        except Exception as e:
            return Paragraph(f"Error extracting final product image: {str(e)}", self.styles['Normal'])

    def _sample_video_segment_enhanced(self, processor, start_time: float, end_time: float, 
                                     num_samples: int, candidate_frames: list, tier: str):
        """Enhanced video sampling with better frame quality assessment."""
        try:
            import cv2
            
            cap = cv2.VideoCapture(processor.video_path)
            if not cap.isOpened():
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame range
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Sample frames evenly across the time range
            if end_frame > start_frame:
                frame_indices = np.linspace(start_frame, end_frame - 1, num_samples, dtype=int)
                
                for frame_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        timestamp = frame_idx / fps
                        candidate_frames.append((timestamp, frame, tier))
            
            cap.release()
            
        except Exception as e:
            print(f"Error sampling video segment: {e}")

    def _ai_select_final_product_frame(self, candidate_frames, assessment_data):
        """Use AI to semantically evaluate and select the best final product frame."""
        try:
            from openai import OpenAI
            import base64
            import cv2
            
            # Get API key from assessment data or environment
            api_key = assessment_data.get('api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("No API key available for AI frame selection, using fallback")
                return candidate_frames[-1][1], candidate_frames[-1][0]  # Use last frame as fallback
            
            client = OpenAI(api_key=api_key)
            
            # Convert candidate frames to base64 for AI analysis
            frame_data = []
            for i, (timestamp, frame, tier) in enumerate(candidate_frames):
                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                frame_data.append({
                    'index': i,
                    'timestamp': timestamp,
                    'tier': tier,
                    'base64': frame_b64
                })
            
            # Create AI prompt for frame selection
            prompt = f"""You are selecting the best final product image from {len(frame_data)} candidate frames near the end of a surgical suturing video.

TASK: Choose the frame that best shows "a clear image of finished suturing without hands, gloves, or instruments."

CRITERIA FOR BEST FINAL PRODUCT IMAGE:
1. Shows completed sutures clearly visible
2. NO hands or surgical gloves (blue/green gloves) in the image
3. NO instruments (scissors, forceps, needle drivers) visible
4. Clear view of the suturing pad/practice surface with finished work
5. Good image quality (sharp, well-lit)

FRAMES TO EVALUATE:
{[f"Frame {f['index']}: timestamp {f['timestamp']:.1f}s, tier {f['tier']}" for f in frame_data]}

Analyze each frame and respond with ONLY the frame index number (0-{len(frame_data)-1}) that best meets the criteria.
If no frame is suitable, respond with "NONE".

Your response:"""

            # Prepare messages with all frame images
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt}
                    ] + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{f['base64']}",
                                "detail": "high"
                            }
                        } for f in frame_data
                    ]
                }
            ]
            
            # Make API call
            response = client.chat.completions.create(
                model="gpt-4o",  # Use GPT-4o for vision analysis
                messages=messages,
                max_tokens=50,
                temperature=0.1
            )
            
            ai_response = response.choices[0].message.content.strip()
            print(f"DEBUG AI FRAME SELECTION: AI chose '{ai_response}'")
            
            # Parse AI response
            if ai_response == "NONE":
                print("AI found no suitable frames, using last available")
                return candidate_frames[-1][1], candidate_frames[-1][0]
            
            try:
                selected_index = int(ai_response)
                if 0 <= selected_index < len(candidate_frames):
                    selected_frame = candidate_frames[selected_index]
                    print(f"AI selected frame {selected_index}: timestamp {selected_frame[0]:.1f}s, tier {selected_frame[2]}")
                    return selected_frame[1], selected_frame[0]
                else:
                    print(f"AI returned invalid index {selected_index}, using fallback")
                    return candidate_frames[-1][1], candidate_frames[-1][0]
            except ValueError:
                print(f"AI returned non-numeric response '{ai_response}', using fallback")
                return candidate_frames[-1][1], candidate_frames[-1][0]
                
        except Exception as e:
            print(f"Error in AI frame selection: {e}")
            # Fallback to last frame
            return candidate_frames[-1][1], candidate_frames[-1][0]

    def _score_final_product_quality(self, frame, tier: str) -> float:
        """Score frame specifically for final product selection - prioritizes clean suture images."""
        try:
            import cv2
            import numpy as np
            
            # Base quality metrics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            contrast = np.std(frame)
            
            # Normalize base metrics
            sharpness_norm = min(sharpness / 1000, 1.0)
            contrast_norm = min(contrast / 100, 1.0)
            
            # CRITICAL: Heavily penalize hands/instruments in ANY part of image
            h, w = frame.shape[:2]
            
            # Check multiple regions for hands/instruments
            regions_to_check = [
                frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)],  # Center region
                frame[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)],  # Wider region
                frame  # Entire frame
            ]
            
            total_penalty = 0.0
            for region in regions_to_check:
                glove_penalty = self._detect_surgical_gloves(region)  # Look for blue/green gloves
                metallic_penalty = self._detect_metallic_objects(region)
                total_penalty += (glove_penalty + metallic_penalty)
            
            # Average penalty across regions
            avg_penalty = total_penalty / len(regions_to_check)
            
            # Tier bonus (prefer very end of video)
            tier_bonus = {"final_1pct": 0.5, "final_3pct": 0.3, "final_5pct": 0.1}.get(tier, 0.0)
            
            # Final product score: heavily weight the absence of hands/instruments
            final_score = (
                sharpness_norm * 0.2 +
                contrast_norm * 0.1 +
                (1.0 - avg_penalty) * 0.6 +  # 60% weight on avoiding hands/instruments
                tier_bonus
            )
            
            # Debug output for frame selection
            print(f"    Frame at {tier}: sharpness={sharpness_norm:.3f}, contrast={contrast_norm:.3f}, penalty={avg_penalty:.3f}, final_score={final_score:.3f}")
            
            return final_score
            
        except Exception as e:
            print(f"Error scoring final product quality: {e}")
            return 0.0

    def _score_frame_quality_enhanced(self, frame, tier: str) -> float:
        """Enhanced frame quality scoring with hand/instrument detection."""
        try:
            import cv2
            import numpy as np
            
            # Base quality metrics
            sharpness = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            contrast = np.std(frame)
            brightness = np.mean(frame)
            
            # Normalize metrics (0-1 scale)
            sharpness_norm = min(sharpness / 1000, 1.0)  # Laplacian variance normalization
            contrast_norm = min(contrast / 100, 1.0)     # Standard deviation normalization
            brightness_norm = 1.0 - abs(brightness - 127.5) / 127.5  # Distance from ideal brightness
            
            # Edge density (more edges = more detail)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Penalize hand/instrument presence in center region
            center_penalty = self._calculate_center_penalty(frame)
            
            # Tier bonus (prefer later frames)
            tier_bonus = {"final_3pct": 0.3, "final_7pct": 0.2, "final_15pct": 0.1}.get(tier, 0.0)
            
            # Combined score
            quality_score = (
                sharpness_norm * 0.3 +
                contrast_norm * 0.25 +
                brightness_norm * 0.2 +
                edge_density * 0.15 +
                (1.0 - center_penalty) * 0.1 +
                tier_bonus
            )
            
            return quality_score
            
        except Exception as e:
            print(f"Error scoring frame quality: {e}")
            return 0.0

    def _calculate_center_penalty(self, frame) -> float:
        """Calculate penalty for hands/instruments in center region."""
        try:
            import cv2
            import numpy as np
            
            h, w = frame.shape[:2]
            center_h_start, center_h_end = int(h * 0.3), int(h * 0.7)
            center_w_start, center_w_end = int(w * 0.3), int(w * 0.7)
            
            center_region = frame[center_h_start:center_h_end, center_w_start:center_w_end]
            
            # Detect skin tones (hands)
            skin_penalty = self._detect_skin_tones(center_region)
            
            # Detect metallic objects (instruments)
            metallic_penalty = self._detect_metallic_objects(center_region)
            
            return min(skin_penalty + metallic_penalty, 1.0)
            
        except Exception as e:
            print(f"Error calculating center penalty: {e}")
            return 0.0

    def _detect_surgical_gloves(self, region) -> float:
        """Detect blue and green surgical gloves in the region."""
        try:
            import cv2
            import numpy as np
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Define blue surgical glove ranges in HSV
            lower_blue1 = np.array([100, 50, 50], dtype=np.uint8)  # Dark blue
            upper_blue1 = np.array([130, 255, 255], dtype=np.uint8)
            lower_blue2 = np.array([90, 30, 80], dtype=np.uint8)   # Light blue
            upper_blue2 = np.array([110, 200, 220], dtype=np.uint8)
            
            # Define green surgical glove ranges in HSV  
            lower_green1 = np.array([40, 40, 40], dtype=np.uint8)   # Dark green
            upper_green1 = np.array([80, 255, 255], dtype=np.uint8)
            lower_green2 = np.array([35, 25, 60], dtype=np.uint8)   # Light green
            upper_green2 = np.array([85, 180, 200], dtype=np.uint8)
            
            # Create masks for all glove colors
            blue_mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
            blue_mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
            green_mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
            green_mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
            
            # Combine all glove masks
            glove_mask = blue_mask1 + blue_mask2 + green_mask1 + green_mask2
            
            # Calculate percentage of region that contains glove colors
            glove_pixels = np.sum(glove_mask > 0)
            total_pixels = region.shape[0] * region.shape[1]
            glove_percentage = glove_pixels / total_pixels if total_pixels > 0 else 0
            
            # Return penalty (0 = no gloves, 1 = lots of gloves)
            return min(glove_percentage * 3.0, 1.0)  # Amplify penalty
            
        except Exception as e:
            print(f"Error detecting surgical gloves: {e}")
            return 0.0

    def _detect_skin_tones(self, region) -> float:
        """Detect skin tones in the region."""
        try:
            import cv2
            import numpy as np
            
            # Convert to HSV for better skin detection
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Define skin tone ranges in HSV
            lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
            lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
            upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
            
            # Create masks
            mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            skin_mask = mask1 + mask2
            
            # Calculate skin percentage
            skin_pixels = np.sum(skin_mask > 0)
            total_pixels = region.shape[0] * region.shape[1]
            skin_percentage = skin_pixels / total_pixels
            
            return min(skin_percentage * 2, 1.0)  # Scale penalty
            
        except Exception as e:
            print(f"Error detecting skin tones: {e}")
            return 0.0

    def _detect_metallic_objects(self, region) -> float:
        """Detect metallic objects (instruments) in the region."""
        try:
            import cv2
            import numpy as np
            
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Detect high-contrast edges (typical of metallic instruments)
            edges = cv2.Canny(gray, 100, 200)
            
            # Look for long, straight edges (characteristic of surgical instruments)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                # Calculate edge density from detected lines
                line_pixels = sum(len(line) for line in lines)
                total_pixels = region.shape[0] * region.shape[1]
                metallic_percentage = line_pixels / total_pixels
                return min(metallic_percentage * 3, 1.0)  # Scale penalty
            
            return 0.0
            
        except Exception as e:
            print(f"Error detecting metallic objects: {e}")
            return 0.0
    
    def _get_final_product_image(self, video_path: str, frames: list = None):
        """Get the best final product image using enhanced selection algorithm."""
        try:
            from PIL import Image
            import cv2
            import numpy as np
            
            # If frames are provided, use them; otherwise extract from video
            if frames and len(frames) > 0:
                # Use the enhanced algorithm on provided frames
                candidate_frames = []
                
                # Sample from the last 20% of frames for final product
                total_frames = len(frames)
                start_idx = max(0, int(total_frames * 0.8))
                
                for i in range(start_idx, total_frames, max(1, (total_frames - start_idx) // 10)):
                    if i < len(frames):
                        frame_data = frames[i]
                        # Convert base64 to image if needed
                        if isinstance(frame_data, dict) and 'base64' in frame_data:
                            import base64
                            from io import BytesIO
                            img_data = base64.b64decode(frame_data['base64'])
                            frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                        else:
                            frame = frame_data
                        
                        if frame is not None:
                            candidate_frames.append((i, frame, "final_frames"))
                
                if not candidate_frames:
                    return None
                
                # Score all frames
                scored_frames = []
                for idx, frame, tier in candidate_frames:
                    quality_score = self._score_frame_quality_enhanced(frame, tier)
                    scored_frames.append((idx, frame, quality_score, tier))
                
                # Sort by quality score (highest first)
                scored_frames.sort(key=lambda x: x[2], reverse=True)
                
                # Return the best frame as PIL Image - NO CROPPING
                best_frame = scored_frames[0][1]
                pil_image = Image.fromarray(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB))
                
                return pil_image
            
            else:
                # Fall back to video processing
                from video_processor import VideoProcessor
                
                processor = VideoProcessor()
                success = processor.load_video(video_path)
                
                if not success:
                    return None
                
                # Use the enhanced extraction method
                assessment_data = {'video_path': video_path}
                return self._extract_final_product_image_enhanced_full(assessment_data, 400)
                
        except Exception as e:
            print(f"Error getting final product image: {e}")
            return None
    
    def _intelligent_crop_suture_pad(self, pil_image):
        """Intelligently crop image to focus on suture pad area."""
        try:
            import cv2
            import numpy as np
            
            # Convert PIL to OpenCV
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Find the practice pad boundaries
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Use edge detection to find pad boundaries
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest rectangular contour (likely the pad)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add some padding around the detected area
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(opencv_image.shape[1] - x, w + 2 * padding)
                h = min(opencv_image.shape[0] - y, h + 2 * padding)
                
                # Crop the image
                cropped = opencv_image[y:y+h, x:x+w]
                
                # Convert back to PIL
                return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            
            # If no good contour found, crop to center square
            h, w = opencv_image.shape[:2]
            size = min(h, w)
            start_y = (h - size) // 2
            start_x = (w - size) // 2
            
            cropped = opencv_image[start_y:start_y+size, start_x:start_x+size]
            return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            print(f"Error in intelligent cropping: {e}")
            return pil_image  # Return original if cropping fails

    def _intelligent_crop_suturing_area(self, image) -> Image.Image:
        """Intelligently crop to focus on the suturing area with practice pad detection."""
        try:
            # Convert to numpy array for processing
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            
            # Start search from center of image
            center_x, center_y = w // 2, h // 2
            
            # Detect practice pad boundaries using edge detection and contour analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use adaptive thresholding to find practice pad boundaries
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Find contours to identify practice pad
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for the largest rectangular contour (likely the practice pad)
            best_contour = None
            best_area = 0
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly rectangular and large enough
                if len(approx) >= 4:
                    area = cv2.contourArea(contour)
                    if area > best_area and area > (w * h * 0.1):  # At least 10% of image
                        best_contour = contour
                        best_area = area
            
            if best_contour is not None:
                # Get bounding rectangle of practice pad
                x, y, pad_w, pad_h = cv2.boundingRect(best_contour)
                
                # Expand the crop area slightly to include suturing area around pad
                padding = 50
                x = max(0, x - padding)
                y = max(0, y - padding)
                pad_w = min(w - x, pad_w + 2 * padding)
                pad_h = min(h - y, pad_h + 2 * padding)
                
                # Make it square by taking the larger dimension
                size = max(pad_w, pad_h)
                
                # Center the square around the practice pad
                center_pad_x = x + pad_w // 2
                center_pad_y = y + pad_h // 2
                
                crop_x = max(0, center_pad_x - size // 2)
                crop_y = max(0, center_pad_y - size // 2)
                
                # Ensure we don't go outside image bounds
                crop_x = min(crop_x, w - size)
                crop_y = min(crop_y, h - size)
                
                cropped = image.crop((crop_x, crop_y, crop_x + size, crop_y + size))
                
            else:
                # Fallback: search from center for best suturing area
                crop_size = min(w, h) // 2  # Start with half the smaller dimension
                
                best_score = 0
                best_crop = (center_x - crop_size // 2, center_y - crop_size // 2)
                
                # Search in expanding squares from center
                for size_factor in [0.5, 0.6, 0.7, 0.8]:
                    current_size = int(min(w, h) * size_factor)
                    start_x = max(0, center_x - current_size // 2)
                    start_y = max(0, center_y - current_size // 2)
                    end_x = min(w, start_x + current_size)
                    end_y = min(h, start_y + current_size)
                    
                    # Adjust if we hit boundaries
                    if end_x - start_x < current_size:
                        start_x = max(0, end_x - current_size)
                    if end_y - start_y < current_size:
                        start_y = max(0, end_y - current_size)
                    
                    crop_region = gray[start_y:end_y, start_x:end_x]
                    edges = cv2.Canny(crop_region, 50, 150)
                    score = np.sum(edges > 0) / (current_size * current_size)
                    
                    if score > best_score:
                        best_score = score
                        best_crop = (start_x, start_y)
                        crop_size = current_size
                
                x, y = best_crop
                cropped = image.crop((x, y, x + crop_size, y + crop_size))
            
            return cropped
            
        except Exception as e:
            print(f"Error in intelligent cropping: {e}")
            # Fallback to center crop
            w, h = image.size
            crop_size = min(w, h) // 2
            x, y = (w - crop_size) // 2, (h - crop_size) // 2
            return image.crop((x, y, x + crop_size, y + crop_size))

    def _format_summative_paragraphs(self, summative_text: str) -> str:
        """Format summative text into well-structured paragraphs."""
        if not summative_text:
            return ""
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', summative_text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            return summative_text
        
        # Group sentences into paragraphs
        paragraphs = []
        current_paragraph = []
        
        for sentence in sentences:
            current_paragraph.append(sentence)
            
            # Check for paragraph breaks (transition words, length, or content)
            if (len(current_paragraph) >= 2 and 
                (any(transition in sentence.lower() for transition in 
                     ['however', 'furthermore', 'additionally', 'moreover', 'in contrast', 'on the other hand']) or
                 len(' '.join(current_paragraph)) > 200)):
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        # Add remaining sentences
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Ensure we have 2-3 paragraphs
        if len(paragraphs) > 3:
            # Merge smaller paragraphs
            merged = []
            for para in paragraphs:
                if len(merged) == 0 or len(merged[-1]) > 150:
                    merged.append(para)
                else:
                    merged[-1] += " " + para
            paragraphs = merged
        
        return '<br/><br/>'.join(paragraphs)

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
