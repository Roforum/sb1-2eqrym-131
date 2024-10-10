import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import './index.css';

const rootElement = document.getElementById('root');

if (rootElement) {
  const renderApp = () => {
    try {
      ReactDOM.createRoot(rootElement).render(
        <React.StrictMode>
          <App />
        </React.StrictMode>
      );
    } catch (error) {
      console.error('Error rendering the application:', error);
      rootElement.innerHTML = '<h1>An error occurred while loading the application. Please refresh the page.</h1>';
    }
  };

  // Wrap the rendering in a setTimeout to defer execution
  setTimeout(renderApp, 0);

  // Add global error handlers
  window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    event.preventDefault();
  });

  window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    event.preventDefault();
  });

  // Handle postMessage errors
  const originalPostMessage = window.postMessage;
  window.postMessage = function(...args) {
    try {
      return originalPostMessage.apply(this, args);
    } catch (error) {
      console.error('Error in postMessage:', error);
      // You can add custom error handling here if needed
    }
  };
} else {
  console.error('Root element not found');
}