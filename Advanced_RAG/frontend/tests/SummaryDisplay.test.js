import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import SummaryDisplay from '../components/SummaryDisplay';
import axios from 'axios';

jest.mock('axios');

test('renders summary display and fetches summary', async () => {
  axios.get.mockResolvedValueOnce({ data: { status: 'SUCCESS', summary: 'Generated summary.' } });
  render(<SummaryDisplay taskId="12345" />);

  expect(screen.getByText(/fetching summary/i)).toBeInTheDocument();

  await waitFor(() => expect(screen.getByText(/summary fetched successfully/i)).toBeInTheDocument());
  expect(screen.getByText(/generated summary/i)).toBeInTheDocument();
});
