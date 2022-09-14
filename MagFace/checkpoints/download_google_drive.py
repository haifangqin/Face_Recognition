from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account # will be installed if you have installed googleapiclient
import json
import sys
import io
file_id, out = sys.argv[1:3]
# auth, you may need change the path of key.json
with open('./key.json') as f:
    key = json.load(f)
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = service_account.Credentials.from_service_account_info(key, scopes=SCOPES)
    with build('drive', 'v3', credentials=creds) as service:
    #fh = io.BytesIO() # keep file in memory
        fh = io.FileIO(out, 'wb') # write file to disk
        request = service.files().get_media(fileId=file_id)
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print("Download %d%%." % int(status.progress() * 100))
