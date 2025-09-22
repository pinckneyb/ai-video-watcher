#!/usr/bin/env python3
"""
Crash Recovery System for Surgical VOP Assessment
Handles checkpointing and resuming of long-running batch jobs
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib


class CrashRecoveryManager:
    """Manages checkpoints and recovery for batch processing"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def _generate_job_id(self, video_path: str, batch_size: int, fps: float) -> str:
        """Generate unique job ID based on video and processing parameters"""
        key_string = f"{os.path.basename(video_path)}_{batch_size}_{fps}"
        return hashlib.md5(key_string.encode()).hexdigest()[:12]
    
    def save_checkpoint(self, video_path: str, batch_size: int, fps: float, 
                       completed_batches: List[Dict], current_stage: str,
                       total_batches: int, additional_data: Dict = None) -> str:
        """Save processing checkpoint"""
        job_id = self._generate_job_id(video_path, batch_size, fps)
        
        checkpoint_data = {
            "job_id": job_id,
            "video_path": video_path,
            "batch_size": batch_size,
            "fps": fps,
            "current_stage": current_stage,
            "total_batches": total_batches,
            "completed_batches": len(completed_batches),
            "batch_results": completed_batches,
            "timestamp": datetime.now().isoformat(),
            "additional_data": additional_data or {}
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{job_id}.json")
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"💾 Checkpoint saved: {checkpoint_path} ({len(completed_batches)}/{total_batches} batches)")
        return job_id
    
    def load_checkpoint(self, job_id: str) -> Optional[Dict]:
        """Load checkpoint by job ID"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{job_id}.json")
        
        if not os.path.exists(checkpoint_path):
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            print(f"📂 Checkpoint loaded: {checkpoint_data['completed_batches']}/{checkpoint_data['total_batches']} batches completed")
            return checkpoint_data
            
        except Exception as e:
            print(f"❌ Error loading checkpoint {job_id}: {e}")
            return None
    
    def find_checkpoint_for_video(self, video_path: str, batch_size: int, fps: float) -> Optional[Dict]:
        """Find existing checkpoint for given video and parameters"""
        job_id = self._generate_job_id(video_path, batch_size, fps)
        return self.load_checkpoint(job_id)
    
    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints"""
        checkpoints = []
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith("checkpoint_") and filename.endswith(".json"):
                job_id = filename.replace("checkpoint_", "").replace(".json", "")
                checkpoint = self.load_checkpoint(job_id)
                if checkpoint:
                    checkpoints.append({
                        "job_id": job_id,
                        "video": os.path.basename(checkpoint["video_path"]),
                        "stage": checkpoint["current_stage"],
                        "progress": f"{checkpoint['completed_batches']}/{checkpoint['total_batches']}",
                        "timestamp": checkpoint["timestamp"]
                    })
        
        return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)
    
    def cleanup_checkpoint(self, job_id: str) -> bool:
        """Remove checkpoint after successful completion"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{job_id}.json")
        
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"🗑️ Checkpoint cleaned up: {job_id}")
                return True
        except Exception as e:
            print(f"⚠️ Failed to cleanup checkpoint {job_id}: {e}")
        
        return False
    
    def cleanup_old_checkpoints(self, days_old: int = 7) -> int:
        """Remove checkpoints older than specified days"""
        removed_count = 0
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith("checkpoint_") and filename.endswith(".json"):
                file_path = os.path.join(self.checkpoint_dir, filename)
                if os.path.getmtime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                        print(f"🗑️ Removed old checkpoint: {filename}")
                    except Exception as e:
                        print(f"⚠️ Failed to remove old checkpoint {filename}: {e}")
        
        if removed_count > 0:
            print(f"🧹 Cleaned up {removed_count} old checkpoints")
        
        return removed_count


def create_resilient_batch_processor(recovery_manager: CrashRecoveryManager):
    """Create a wrapper function for resilient batch processing"""
    
    def process_with_recovery(process_function, video_path: str, batch_size: int, 
                            fps: float, *args, **kwargs):
        """
        Wrapper that adds crash recovery to any batch processing function
        """
        # Check for existing checkpoint
        checkpoint = recovery_manager.find_checkpoint_for_video(video_path, batch_size, fps)
        
        if checkpoint:
            print(f"🔄 RECOVERY MODE: Found checkpoint with {checkpoint['completed_batches']}/{checkpoint['total_batches']} batches completed")
            print(f"📍 Last stage: {checkpoint['current_stage']}")
            
            # Offer recovery options
            import streamlit as st
            recovery_choice = st.selectbox(
                "Recovery Options:",
                ["Resume from checkpoint", "Start fresh (discard checkpoint)", "View checkpoint details"]
            )
            
            if recovery_choice == "Resume from checkpoint":
                # Resume processing from checkpoint
                return process_function(
                    video_path, batch_size, fps, 
                    checkpoint=checkpoint, 
                    *args, **kwargs
                )
            elif recovery_choice == "View checkpoint details":
                st.json(checkpoint)
                return None
            else:
                # Clean up checkpoint and start fresh
                recovery_manager.cleanup_checkpoint(checkpoint['job_id'])
        
        # Start fresh processing
        return process_function(video_path, batch_size, fps, *args, **kwargs)
    
    return process_with_recovery