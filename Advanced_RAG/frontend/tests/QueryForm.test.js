import React from 'react';
import { render, fireEvent, screen } from '@testing-library/react';
import QueryForm from '../components/QueryForm';
import axios from 'axios';

jest.mock('axios');

test('renders query form and handles submission', async () => {
  axios.post.mockResolvedValue({ data: { task_id: '12345' } });
  const setTaskId = jest.fn();
  render(<QueryForm setTaskId={setTaskId} />);

  const textarea = screen.getByPlaceholderText(/enter your query here/i);
  const submitButton = screen.getByText(/submit/i);

  fireEvent.change(textarea, { target: { value: 'Test query' } });
  fireEvent.click(submitButton);

  expect(screen.getByText(/submitting query/i)).toBeInTheDocument();

  // Wait for the status update
  const successMessage = await screen.findByText(/query submitted successfully/i);
  expect(successMessage).toBeInTheDocument();
  expect(setTaskId).toHaveBeenCalledWith('12345');
});
