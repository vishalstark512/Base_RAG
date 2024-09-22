import React from 'react';
import { render, fireEvent, screen } from '@testing-library/react';
import UploadDocs from '../components/UploadDocs';
import axios from 'axios';

jest.mock('axios');

test('renders upload component and handles file upload', async () => {
  axios.post.mockResolvedValue({ data: { status: 'success' } });
  render(<UploadDocs />);

  const fileInput = screen.getByLabelText(/upload documents/i);
  const uploadButton = screen.getByText(/upload/i);

  const file = new File(['doc content'], 'doc1.txt', { type: 'text/plain' });
  fireEvent.change(fileInput, { target: { files: [file] } });
  fireEvent.click(uploadButton);

  expect(screen.getByText(/uploading/i)).toBeInTheDocument();

  // Wait for the status update
  const successMessage = await screen.findByText(/upload successful/i);
  expect(successMessage).toBeInTheDocument();
});
