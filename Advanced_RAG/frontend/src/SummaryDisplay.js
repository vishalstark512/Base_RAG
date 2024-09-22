import React, { useEffect, useState } from 'react';
import axios from 'axios';

function SummaryDisplay({ taskId }) {
  const [summary, setSummary] = useState('');
  const [status, setStatus] = useState('Fetching summary...');

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const response = await axios.get(`/summarize/${taskId}`);
        if (response.data.status === 'SUCCESS') {
          setSummary(response.data.summary);
          setStatus('Summary fetched successfully.');
        } else if (response.data.status === 'FAILURE') {
          setStatus('Failed to generate summary.');
        } else {
          setStatus('Summary is being generated...');
          setTimeout(fetchSummary, 3000); // Poll every 3 seconds
        }
      } catch (error) {
        setStatus('Error fetching summary.');
        console.error(error);
      }
    };

    fetchSummary();
  }, [taskId]);

  return (
    <div>
      <h2>Summary</h2>
      <p>{status}</p>
      {summary && <div>{summary}</div>}
    </div>
  );
}

export default SummaryDisplay;
