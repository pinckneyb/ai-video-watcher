"""
Google Drive Uploader for Surgical VOP Assessment Reports
Automatically uploads HTML reports to specified Google Drive folder
"""

import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
import streamlit as st

class GoogleDriveUploader:
    def __init__(self, service_account_info=None, folder_id=None):
        """
        Initialize Google Drive uploader
        
        Args:
            service_account_info: Dict containing service account credentials
            folder_id: Target Google Drive folder ID
        """
        self.service = None
        self.folder_id = folder_id or "1IgNwbY9Py9fdSKWva4Lrf1bQzTqYC"  # Default folder
        
        if service_account_info:
            self.authenticate(service_account_info)
    
    def authenticate(self, service_account_info):
        """Authenticate with Google Drive API using service account"""
        try:
            SCOPES = ['https://www.googleapis.com/auth/drive']
            
            # Create credentials from service account info
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info, scopes=SCOPES
            )
            
            # Build the service
            self.service = build('drive', 'v3', credentials=credentials)
            return True
            
        except Exception as e:
            st.error(f"❌ Google Drive authentication failed: {str(e)}")
            return False
    
    def upload_html_report(self, file_path, custom_name=None):
        """
        Upload HTML report to Google Drive folder
        
        Args:
            file_path: Local path to HTML file
            custom_name: Optional custom name for the file
            
        Returns:
            dict: Upload result with file info or error
        """
        if not self.service:
            return {"success": False, "error": "Not authenticated with Google Drive"}
        
        if not os.path.exists(file_path):
            return {"success": False, "error": f"File not found: {file_path}"}
        
        try:
            # Prepare file metadata
            file_name = custom_name or os.path.basename(file_path)
            file_metadata = {
                'name': file_name,
                'parents': [self.folder_id]  # Upload to specific folder
            }
            
            # Create media upload object
            media = MediaFileUpload(
                file_path, 
                mimetype='text/html',
                resumable=True
            )
            
            # Upload the file
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name,webViewLink,parents'
            ).execute()
            
            return {
                "success": True,
                "file_id": file.get('id'),
                "file_name": file.get('name'),
                "web_link": file.get('webViewLink'),
                "message": f"✅ Successfully uploaded to Google Drive: {file.get('name')}"
            }
            
        except HttpError as e:
            error_msg = f"Google Drive API error: {str(e)}"
            if "404" in str(e):
                error_msg += " (Check if folder is shared with service account)"
            return {"success": False, "error": error_msg}
            
        except Exception as e:
            return {"success": False, "error": f"Upload failed: {str(e)}"}
    
    def test_connection(self):
        """Test Google Drive connection and folder access"""
        if not self.service:
            return {"success": False, "error": "Not authenticated"}
        
        try:
            # Try to access the target folder
            folder = self.service.files().get(fileId=self.folder_id).execute()
            return {
                "success": True,
                "folder_name": folder.get('name'),
                "message": f"✅ Connected to Google Drive folder: {folder.get('name')}"
            }
            
        except HttpError as e:
            if "404" in str(e):
                return {
                    "success": False, 
                    "error": "Folder not found or not shared with service account"
                }
            else:
                return {"success": False, "error": f"Access error: {str(e)}"}
        
        except Exception as e:
            return {"success": False, "error": f"Connection test failed: {str(e)}"}

def get_drive_uploader():
    """Get configured Google Drive uploader from session state"""
    if 'google_drive_uploader' in st.session_state:
        return st.session_state.google_drive_uploader
    return None

def setup_google_drive_integration():
    """Setup Google Drive integration in Streamlit sidebar"""
    with st.sidebar:
        st.subheader("☁️ Google Drive Integration")
        
        # Check if already configured
        uploader = get_drive_uploader()
        if uploader and uploader.service:
            # Test connection
            test_result = uploader.test_connection()
            if test_result["success"]:
                st.success("✅ Google Drive connected")
                st.info(f"📁 Target folder: {test_result.get('folder_name', 'Unknown')}")
                
                if st.button("🔄 Reset Google Drive Connection"):
                    if 'google_drive_uploader' in st.session_state:
                        del st.session_state.google_drive_uploader
                    st.rerun()
                    
                return uploader
            else:
                st.error(f"❌ Connection failed: {test_result['error']}")
        
        # Configuration form
        st.write("**Setup Instructions:**")
        st.write("1. Create a Google Cloud Project")
        st.write("2. Enable Google Drive API")
        st.write("3. Create Service Account & download JSON key")
        st.write("4. Share your Drive folder with service account email")
        
        uploaded_json = st.file_uploader(
            "Upload Service Account JSON", 
            type=['json'],
            help="Upload the JSON key file from your Google Cloud service account"
        )
        
        if uploaded_json:
            try:
                # Parse the JSON credentials
                service_account_info = json.loads(uploaded_json.read().decode())
                
                # Create uploader instance
                uploader = GoogleDriveUploader(
                    service_account_info=service_account_info,
                    folder_id="1IgNwbY9Py9fdSKWva4Lrf1bQzTqYC"  # User's folder ID
                )
                
                if uploader.service:
                    # Test the connection
                    test_result = uploader.test_connection()
                    
                    if test_result["success"]:
                        st.success(test_result["message"])
                        st.session_state.google_drive_uploader = uploader
                        st.rerun()
                    else:
                        st.error(f"❌ {test_result['error']}")
                        st.write("**Troubleshooting:**")
                        st.write("- Ensure the folder is shared with the service account email")
                        st.write(f"- Service account email: `{service_account_info.get('client_email', 'N/A')}`")
                else:
                    st.error("❌ Failed to authenticate with Google Drive")
                    
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file")
            except Exception as e:
                st.error(f"❌ Setup error: {str(e)}")
        
        return None