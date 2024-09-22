import React, { useState } from 'react';
import axios from 'axios';

function QueryForm({ setTaskId }) {
  const [query, setQuery] = useState('');
  const [status, setStatus] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setStatus('Submitting query...');
      const response = await axios.post('/summarize', { query });
      setTaskId(response.data.task_id);
      setStatus('Query submitted successfully.');
    } catch (error) {
      setStatus('Query submission failed.');
      console.error(error);
    }
  };

  return (
    <div>
      <h2>Ask a Question</h2>
      <form onSubmit={handleSubmit}>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your query here..."
          rows="4"
          cols="50"
        />
        <br />
        <button type="submit">Submit</button>
      </form>
      <p>{status}</p>
    </div>
  );
}

export default QueryForm;
