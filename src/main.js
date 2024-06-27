import React, { useState } from 'react';
import './App.css';
import { Button, IconButton, CircularProgress } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

function Main() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [userQuery, setUserQuery] = useState('');
    const [uploadUrl, setUploadUrl] = useState('');
    const [processingResult, setProcessingResult] = useState(null);
    const [processingRelevantOnlyResult, setProcessingRelevantOnlyResult] = useState(null);
    const [relevantOnlyFilePath, setRelevantOnlyFilePath] = useState('');
    const [searchTerms, setSearchTerms] = useState([{ category: '', query: '' }]);
    const [loading, setLoading] = useState(false);

    // const handleFileChange = (event) => {
    //     setSelectedFile(event.target.files[0]);
    // };

    const handleQueryChange = (event) => {
        setUserQuery(event.target.value);
    };

    const handleSearchTermChange = (index, field, value) => {
        const newSearchTerms = [...searchTerms];
        newSearchTerms[index][field] = value;
        setSearchTerms(newSearchTerms);
    };

    const addSearchTerm = () => {
        setSearchTerms([...searchTerms, { category: '', query: '' }]);
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
        handleFileUpload(file);  // Call handleFileUpload immediately after setting the file
    };

    const handleFileUpload = async (file) => {
        const formData = new FormData();
        formData.append('file', file);

        try {
            setLoading(true); // Set loading state to true
            const response = await fetch('http://127.0.0.1:5000/upload', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const data = await response.json();
            setUploadUrl(data.file_path);
        } catch (error) {
            console.error('Error uploading file:', error);
        } finally {
            setLoading(false); // Set loading state to false
        }
    };


    const handleFileProcessing = async () => {
        try {
            const response = await fetch('http://127.0.0.1:5000/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_query: userQuery,
                    file_path: uploadUrl,
                }),
            });
            const data = await response.json();
            setProcessingResult(data);
        } catch (error) {
            console.error('Error processing file:', error);
        }
    };

    const handleRelevantOnlyFileProcessing = async () => {
        try {
            const response = await fetch('http://127.0.0.1:5000/process-relevant-only', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    relevant_only_file_path: relevantOnlyFilePath,
                    search_terms: searchTerms.map(term => [term.category, term.query]),
                }),
            });
            const data = await response.json();
            setProcessingRelevantOnlyResult(data);
        } catch (error) {
            console.error('Error processing relevant only file:', error);
        }
    };

    return (
        <div className="App">
            {loading && (
                <div className="loading-overlay">
                    <CircularProgress />
                </div>
            )} {/* Loading overlay */}
            <h1>File Upload and Processing</h1>
            <div style={{ display: 'flex', alignItems: 'center' }}>
                <input
                    accept="*"
                    style={{ display: 'none' }}
                    id="contained-button-file"
                    type="file"
                    onChange={handleFileChange}
                />
                <label htmlFor="contained-button-file">
                    <Button
                        variant="contained"
                        color="primary"
                        component="span"
                        startIcon={<CloudUploadIcon />}
                    >
                        Choose File
                    </Button>
                </label>
            </div>
            {uploadUrl && (
                <>
                    <h2>Process File</h2>
                    <input type="text" placeholder="Enter your query" value={userQuery} onChange={handleQueryChange} />
                    <button onClick={handleFileProcessing}>Process File</button>
                </>
            )}

            {processingResult && (
                <div>
                    <h3>Processing Result</h3>
                    <p>
                        Path: <a href={`http://127.0.0.1:5000/download/${processingResult.Path.split('/').pop()}`} download>Download Original File</a>
                    </p>
                    <p>
                        Filtered Path: <a href={`http://127.0.0.1:5000/download/${processingResult.FilteredPath.split('/').pop()}`} download>Download Filtered File</a>
                    </p>

                    <h2>Process Relevant Only File</h2>
                    {/* <input
            type="text"
            placeholder="Relevant Only File Path"
            value={relevantOnlyFilePath}
            onChange={(e) => setRelevantOnlyFilePath(e.target.value)}
          /> */}
                    {searchTerms.map((term, index) => (
                        <div key={index}>
                            <input
                                type="text"
                                placeholder="Category"
                                value={term.category}
                                onChange={(e) => handleSearchTermChange(index, 'category', e.target.value)}
                            />
                            <input
                                type="text"
                                placeholder="Query"
                                value={term.query}
                                onChange={(e) => handleSearchTermChange(index, 'query', e.target.value)}
                            />
                        </div>
                    ))}
                    <button onClick={addSearchTerm}>Add Search Term</button>
                    <button onClick={handleRelevantOnlyFileProcessing}>Process Relevant Only File</button>

                    {processingRelevantOnlyResult && (
                        <div>
                            <h3>Relevant Only Processing Result</h3>
                            <p>
                                Path: <a href={`http://127.0.0.1:5000/download/${processingRelevantOnlyResult.Path.split('/').pop()}`} download>Download Relevant Only Processed File</a>
                            </p>
                        </div>
                    )}
                </div>
            )}



        </div>
    );
}

export default Main;
