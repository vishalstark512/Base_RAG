import React, { useState } from 'react';
import axios from 'axios';

function UploadDocs() {
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState('');

  const handleFileChange = (e) => {
    setFiles(e.target.files);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    for (let file of files) {
      formData.append('files', file);
    }
    try {
      setStatus('Uploading...');
      await axios.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setStatus('Upload successful!');
    } catch (error) {
      setStatus('Upload failed.');
      console.error(error);
    }
  };

  return (
    <div>
      <h2>Upload Documents</h2>
      <input
        type="file"
        multiple
        onChange={handleFileChange}
        aria-label="Upload Documents"
      />
      <button onClick={handleUpload}>Upload</button>
      <p>{status}</p>
    </div>
  );
}

export default UploadDocs;
