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
                    
                    # Score with interpretation (handle provisional scores)
                    provisional_scores = getattr(self, 'provisional_scores', {})
                    is_provisional = provisional_scores.get(pid, False)
                    
                    if is_provisional:
                        score_para = f"<b>Score: {score}*/5 ({score_text}) - Provisional, requires review</b>"
                    else:
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
        """Extract clean, concise rubric point assessment WITHOUT timestamps for first page."""
        
        if not enhanced_narrative:
            return f"Summary: Detailed assessment requires enhanced narrative from GPT-5 analysis of complete video."
        
        # Look for the specific rubric point in the enhanced narrative
        lines = enhanced_narrative.split('\n')
        
        # Find the section for this specific rubric point
        rubric_section = None
        collecting = False
        collected_content = []
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Check if this line starts our target rubric point
            if (line_clean.startswith(f"**{pid}.") or 
                line_clean.startswith(f"{pid}.") or
                (f"**{pid}." in line_clean and title.lower() in line_clean.lower())):
                collecting = True
                continue
            
            # If we're collecting and hit another rubric point, stop
            elif collecting and (any(line_clean.startswith(f"**{j}.") or line_clean.startswith(f"{j}.") for j in range(1, 8) if j != pid) or
                                line_clean.startswith("**Summative") or
                                line_clean.startswith("Summative")):
                break
            
            # If collecting, add content but filter out timestamps and scores
            elif collecting:
                # Skip lines that are just timestamps or scores
                if not (line_clean.startswith("Score:") or 
                       "00:" in line_clean or
                       line_clean.startswith("**") and ":" in line_clean):
                    collected_content.append(line_clean)
        
        # Clean up the collected content
        if collected_content:
            # Join all content and clean it up
            content = " ".join(collected_content)
            
            # Remove any remaining timestamp patterns
            import re
            content = re.sub(r'\b\d{2}:\d{2}(?::\d{2})?(?:–\d{2}:\d{2}(?::\d{2})?)?[;,\s]*', '', content)
            content = re.sub(r'\(e\.g\.,\s*[^)]*\)', '', content)  # Remove timestamp examples
            content = re.sub(r'\s+', ' ', content)  # Clean up multiple spaces
            content = content.strip()
            
            # Remove "Score: X/5" if present
            content = re.sub(r'\s*Score:\s*\d+/5\s*', '', content)
            
            if len(content) > 50:  # Make sure we have substantial content
                return content
        
        # Fallback: create a basic assessment based on score
        score_text = self._get_score_interpretation(score)
        return f"Assessment of {title.lower()} based on video analysis shows {score_text.lower()} performance level."
    
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
        summative_formatted = self._format_summative_paragraphs(summative_feedback)
        
        story.append(Paragraph("<b>Summative Comment:</b>", self.styles['Normal']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(summative_formatted.strip(), self.styles['Normal']))
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
                    
                    # Extract final product image from video
                    learner_img = self._extract_final_product_image(assessment_data, target_height)
                    
                    # Create table with images side by side
                    image_data = [
                        ['Gold Standard', 'Learner Performance'],
                        [gold_img, learner_img]
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
        """Extract the actual summative assessment from GPT-5 enhanced narrative."""
        enhanced_narrative = assessment_data.get('enhanced_narrative', '')
        
        if not enhanced_narrative:
            return "Comprehensive summative feedback requires enhanced narrative generation from AI analysis."
        
        # GPT-5 writes summative assessment after all rubric points
        # Look for content after all "Score: X/5" entries
        lines = enhanced_narrative.split('\n')
        summative_started = False
        summative_lines = []
        
        # Find where rubric scoring ends and summative begins
        for line in lines:
            line_clean = line.strip()
            
            # If we see "Score: X/5", we're still in rubric section
            if "Score:" in line_clean and "/5" in line_clean:
                summative_started = False
                continue
            
            # If we've passed all scores and see substantial content, it's summative
            if not summative_started and line_clean and len(line_clean) > 50:
                # Check if this line doesn't start with rubric indicators
                if not (line_clean.startswith(('**1.', '**2.', '**3.', '**4.', '**5.', '**6.', '**7.', '1.', '2.', '3.', '4.', '5.', '6.', '7.'))):
                    summative_started = True
                    summative_lines.append(line_clean)
            elif summative_started and line_clean and len(line_clean) > 20:
                summative_lines.append(line_clean)
        
        # If we found summative content, return it
        if summative_lines:
            return ' '.join(summative_lines)
        
        # Fallback: look for any content after the last score
        last_score_index = -1
        for i, line in enumerate(lines):
            if "Score:" in line and "/5" in line:
                last_score_index = i
        
        if last_score_index >= 0:
            # Take substantial content after the last score
            for i in range(last_score_index + 1, len(lines)):
                line_clean = lines[i].strip()
                if len(line_clean) > 50:
                    # Collect this and following substantial lines
                    summative_content = []
                    for j in range(i, len(lines)):
                        if lines[j].strip() and len(lines[j].strip()) > 20:
                            summative_content.append(lines[j].strip())
                    return ' '.join(summative_content)
        
        return "Enhanced narrative analysis required for detailed summative feedback."
    
    def _format_summative_paragraphs(self, summative_text: str) -> str:
        """Format summative comment into meaningful paragraphs for readability."""
        if not summative_text or len(summative_text.strip()) < 50:
            return summative_text
        
        # Split into sentences
        sentences = []
        current_sentence = ""
        
        for char in summative_text:
            current_sentence += char
            if char in '.!?' and len(current_sentence.strip()) > 20:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add any remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        if len(sentences) <= 3:
            return summative_text  # Too short for meaningful paragraphing
        
        # Group sentences into 2-3 paragraphs based on content
        paragraphs = []
        current_para = []
        
        # Simple heuristic: group every 2-3 sentences, but break on key transition words
        transition_words = ['however', 'furthermore', 'additionally', 'nevertheless', 'conversely', 'in contrast', 'overall']
        
        for i, sentence in enumerate(sentences):
            current_para.append(sentence)
            
            # Check if we should start a new paragraph
            should_break = False
            
            # Break after 2-3 sentences
            if len(current_para) >= 3:
                should_break = True
            elif len(current_para) >= 2 and i < len(sentences) - 1:
                # Look ahead to see if next sentence starts with transition word
                next_sentence = sentences[i + 1].lower()
                for word in transition_words:
                    if next_sentence.startswith(word):
                        should_break = True
                        break
            
            if should_break and current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
        
        # Add any remaining sentences
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        # Join paragraphs with double line break for PDF formatting
        return '<br/><br/>'.join(paragraphs)
    
    def _extract_final_product_image(self, assessment_data: Dict[str, Any], target_height: float):
        """Extract a high-quality final product image with robust hand/instrument avoidance."""
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
                quality_score = self._score_frame_quality_enhanced(frame_data['frame'], tier)
                scored_frames.append((timestamp, frame_data, quality_score, tier))
            
            # Select best frame (highest quality score)
            best_frame = max(scored_frames, key=lambda x: x[2])
            frame_data = best_frame[1]
            
            # Create ReportLab Image directly from PIL image using BytesIO
            import io
            img_buffer = io.BytesIO()
            frame_data['frame_pil'].save(img_buffer, format='JPEG', quality=95)  # Higher quality
            img_buffer.seek(0)
            
            # Create ReportLab Image from buffer
            learner_img = Image(img_buffer)
            
            # Get original dimensions and scale to match target height
            pil_img = frame_data['frame_pil']
            original_width, original_height = pil_img.size
            aspect_ratio = original_width / original_height
            learner_width = target_height * aspect_ratio
            
            learner_img.drawHeight = target_height
            learner_img.drawWidth = learner_width
            
            return learner_img
                
        except Exception as e:
            return Paragraph(f"Error extracting final product image: {e}", self.styles['Normal'])
    
    def _sample_video_segment_enhanced(self, processor, start_time, end_time, num_samples, candidate_frames, tier_name):
        """Sample frames from a specific video segment with enhanced selection."""
        try:
            segment_duration = end_time - start_time
            if segment_duration <= 0:
                return
            
            for i in range(num_samples):
                if num_samples == 1:
                    sample_time = start_time + segment_duration * 0.5
                else:
                    sample_time = start_time + (segment_duration * i / (num_samples - 1))
                
                frame_data = processor.get_frame_at_time(sample_time)
                if frame_data:
                    candidate_frames.append((sample_time, frame_data, tier_name))
                    
        except Exception as e:
            print(f"Error sampling {tier_name} segment: {e}")
    
    def _score_frame_quality_enhanced(self, frame_array, tier):
        """Enhanced frame quality scoring with hand/instrument detection avoidance."""
        try:
            import numpy as np
            import cv2
            
            # Convert to grayscale for analysis
            if len(frame_array.shape) == 3:
                gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame_array
            
            # Basic quality metrics
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            contrast = gray.std()
            
            # Brightness analysis (prefer well-lit images)
            brightness_mean = gray.mean()
            brightness_score = 1.0 - abs(brightness_mean - 128) / 128
            
            # Edge detection (good for seeing suture lines)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = (edges > 0).sum() / edges.size
            
            # Hand/instrument avoidance scoring
            h, w = gray.shape
            center_region = frame_array[h//4:3*h//4, w//4:3*w//4] if len(frame_array.shape) == 3 else gray[h//4:3*h//4, w//4:3*w//4]
            
            hand_penalty = 0
            if len(frame_array.shape) == 3:
                # Skin tone detection (crude but effective)
                skin_score = self._detect_skin_tones(center_region)
                hand_penalty += skin_score * 0.4
                
                # Metallic instrument detection
                metallic_score = self._detect_metallic_objects(center_region)
                hand_penalty += metallic_score * 0.3
            
            # Combine scores
            base_quality = (sharpness / 1000 + contrast / 100 + brightness_score + edge_density * 10) / 4
            
            # Tier bonus (prefer very latest frames)
            tier_bonus = {"final_3pct": 0.4, "final_7pct": 0.2, "final_15pct": 0.0}.get(tier, 0)
            
            # Apply hand penalty
            final_score = base_quality + tier_bonus - hand_penalty
            
            return max(0, final_score)
            
        except Exception as e:
            print(f"Error in enhanced quality scoring: {e}")
            return 0.0
    
    def _detect_skin_tones(self, image_region):
        """Detect skin tones in image region."""
        try:
            import cv2
            import numpy as np
            
            # Convert to HSV for better skin detection
            hsv = cv2.cvtColor(image_region, cv2.COLOR_RGB2HSV)
            
            # Define skin tone ranges (conservative to avoid false positives)
            lower_skin1 = np.array([0, 20, 70])
            upper_skin1 = np.array([20, 255, 255])
            lower_skin2 = np.array([160, 20, 70])
            upper_skin2 = np.array([180, 255, 255])
            
            skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            skin_mask = skin_mask1 | skin_mask2
            
            return (skin_mask > 0).sum() / skin_mask.size
            
        except:
            return 0.0
    
    def _detect_metallic_objects(self, image_region):
        """Detect metallic instruments in image region."""
        try:
            import cv2
            import numpy as np
            
            # Convert to grayscale
            gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
            
            # Look for very bright regions (instruments reflect light)
            bright_mask = gray > 200
            
            # Look for high contrast edges (instruments have sharp edges)
            edges = cv2.Canny(gray, 100, 200)
            edge_mask = edges > 0
            
            # Combine bright areas with high edge density
            metallic_score = (bright_mask & edge_mask).sum() / bright_mask.size
            
            return metallic_score
            
        except:
            return 0.0
    
    def _score_frame_quality(self, frame_array):
        """Fallback frame quality scoring (backwards compatibility)."""
        try:
            import numpy as np
            
            # Convert to grayscale for analysis
            if len(frame_array.shape) == 3:
                gray = np.dot(frame_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = frame_array
            
            # Simple sharpness metric using gradient magnitude
            grad_x = np.gradient(gray, axis=1)
            grad_y = np.gradient(gray, axis=0)
            sharpness = np.sqrt(grad_x**2 + grad_y**2).mean()
            
            # Calculate contrast (standard deviation)
            contrast = gray.std()
            
            # Calculate brightness distribution (prefer mid-range brightness)
            mean_brightness = gray.mean()
            brightness_score = 1.0 - abs(mean_brightness - 127) / 127
            
            # Combine scores (weights favor sharpness and contrast)
            quality_score = (sharpness * 0.5) + (contrast * 0.3) + (brightness_score * 0.2)
            
            return quality_score
            
        except Exception:
            # Fallback scoring - just use standard deviation as proxy for content
            try:
                return frame_array.std()
            except:
                return 0.0

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
