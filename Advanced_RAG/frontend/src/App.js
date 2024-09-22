import React, { useState } from 'react';
import UploadDocs from './components/UploadDocs';
import QueryForm from './components/QueryForm';
import SummaryDisplay from './components/SummaryDisplay';
import Navbar from './components/Navbar';
import './App.css';

function App() {
  const [taskId, setTaskId] = useState('');

  return (
    <div className="App">
      <Navbar />
      <UploadDocs />
      <QueryForm setTaskId={setTaskId} />
      {taskId && <SummaryDisplay taskId={taskId} />}
    </div>
  );
}

export default App;
