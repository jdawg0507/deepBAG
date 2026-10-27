import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Send, Bot, User, Sparkles, Flame, RefreshCcw, HelpCircle, Cpu } from 'lucide-react';
import { ChartRenderer, parseChartMarkdownBlock } from './components/ChartRenderer';
import { NetworkGraph } from './components/NetworkGraph';
import { GenerativeSVG } from './components/GenerativeSVG';
import { TelemetryBanner } from './components/TelemetryBanner';
import { ReasoningAccordion } from './components/ReasoningAccordion';
import './index.css';

const SUGGESTIONS = [
  { icon: '🔥', label: "LeBron vs Steph last 5 games", query: "Compare LeBron James and Stephen Curry scoring and 3-pointers over their last 5 games. Include a chart comparing them." },
  { icon: '👑', label: "Jokic triple-double profile", query: "Show Nikola Jokic 2023-24 averages for points, rebounds, assists, steals, and blocks. Include a radar or bar chart." },
  { icon: '🛡️', label: "Top defensive leaders", query: "Who are the top 5 players in defensive rating this season? Include a bar chart comparing their ratings." },
  { icon: '📜', label: "Wilt 100-point game", query: "What was Wilt Chamberlain's box score in his 100-point game on March 2, 1962?" },
];

function App() {
  const [messages, setMessages] = useState([
    {
      role: 'agent',
      content: 'Welcome to **deepBAG Enterprise Gateway**! I am your AI Basketball Analytics Router. Ask complex NBA statistical questions, multi-player comparisons, or historical records, and I will route them to live NBA APIs or historical box scores with interactive visualizations.',
      reasoning: []
    }
  ]);
  const [input, setInput] = useState('');
  const [selectedModel, setSelectedModel] = useState('gemini-3.1-flash-lite');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const sendQuery = async (queryText) => {
    if (!queryText.trim() || isLoading) return;

    const userMessage = queryText.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage, model: selectedModel }),
      });

      if (!response.ok) {
        throw new Error(`Gateway returned HTTP ${response.status}`);
      }

      // Add empty agent message container
      setMessages(prev => [...prev, { role: 'agent', reasoning: [], content: '' }]);

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let done = false;

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        if (value) {
          const chunkValue = decoder.decode(value, { stream: true });
          const events = chunkValue.split('\n\n').filter(Boolean);

          for (const eventStr of events) {
            if (eventStr.startsWith('data: ')) {
              try {
                const data = JSON.parse(eventStr.substring(6));

                setMessages(prev => {
                  if (prev.length === 0) return prev;
                  const newMessages = [...prev];
                  const lastIdx = newMessages.length - 1;
                  const prevMsg = newMessages[lastIdx];
                  const currentMsg = {
                    ...prevMsg,
                    reasoning: prevMsg.reasoning ? [...prevMsg.reasoning] : []
                  };

                  if (data.type === 'final_answer') {
                    currentMsg.content = typeof data.content === 'string' ? data.content : JSON.stringify(data.content, null, 2);
                  } else {
                    // Prevent duplicate trace cards caused by React StrictMode double invocation
                    const isDuplicate = currentMsg.reasoning.some(
                      r => r.type === data.type && r.content === data.content && r.details === data.details
                    );
                    if (!isDuplicate) {
                      currentMsg.reasoning.push(data);
                    }
                  }

                  newMessages[lastIdx] = currentMsg;
                  return newMessages;
                });
              } catch (err) {
                console.error('Error parsing SSE data chunk:', err);
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Error in agentic chat streaming:', error);
      setMessages(prev => [...prev, {
        role: 'agent',
        content: `### ⚠️ Gateway Connection Notice\nUnable to stream response from backend gateway (${error.message}). Ensure the backend is running at \`http://localhost:8000\`.`,
        reasoning: []
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendQuery(input);
  };

  // Custom Markdown Code Component for Universal Generative Visualizations
  const MarkdownComponents = {
    code({ node, className, children, ...props }) {
      const match = /language-(\w+|[\w:]+)/.exec(className || '');
      const lang = match ? match[1].toLowerCase() : '';

      if (lang === 'chart' || lang === 'chartjs' || lang === 'json:chart') {
        const rawCode = String(children).replace(/\n$/, '');
        const chartConfig = parseChartMarkdownBlock(rawCode);
        if (chartConfig) {
          return <ChartRenderer config={chartConfig} />;
        }
      }

      if (lang === 'graph' || lang === 'network' || lang === 'json:graph' || lang === 'json:network') {
        const rawCode = String(children).replace(/\n$/, '');
        const graphConfig = parseChartMarkdownBlock(rawCode);
        if (graphConfig) {
          return <NetworkGraph config={graphConfig} />;
        }
      }

      if (lang === 'svg' || lang === 'xml:svg') {
        const rawCode = String(children).replace(/\n$/, '');
        return <GenerativeSVG svgContent={rawCode} />;
      }

      return (
        <code className={className} {...props}>
          {children}
        </code>
      );
    }
  };

  return (
    <div className="enterprise-layout">
      {/* Top Telemetry Header */}
      <header className="header-container">
        <div className="header-branding">
          <div className="header-logo-badge">
            <Flame size={20} className="logo-flame-icon" />
          </div>
          <div>
            <h1 className="header-title">deepBAG</h1>
            <p className="header-subtitle">Production API Gateway & Semantic NBA Router</p>
          </div>
        </div>
        <div className="header-controls">
          <div className="model-picker-container" title="Select Gemini Model Routing">
            <Cpu size={14} className="model-picker-icon" />
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="model-picker-select"
              aria-label="Select Gemini Model"
            >
              <option value="gemini-3.1-flash-lite">⚡ gemini-3.1-flash-lite (High RPM / Default)</option>
              <option value="gemini-3.1-flash-lite-preview">🚀 gemini-3.1-flash-lite-preview</option>
              <option value="gemini-3.5-flash-lite">✨ gemini-3.5-flash-lite</option>
              <option value="gemini-3.8-flash">🔥 gemini-3.8-flash</option>
              <option value="gemini-3.5-flash">💡 gemini-3.5-flash</option>
              <option value="gemini-flash-latest">📦 gemini-flash-latest</option>
              <option value="gemini-pro-latest">🧠 gemini-pro-latest</option>
            </select>
          </div>
          <TelemetryBanner />
        </div>
      </header>

      {/* Main Chat Stream Container */}
      <main className="chat-viewport">
        <div className="messages-stream">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message-row ${msg.role}`}>
              <div className="avatar-container">
                {msg.role === 'user' ? (
                  <div className="avatar user-avatar"><User size={18} /></div>
                ) : (
                  <div className="avatar bot-avatar"><Bot size={18} /></div>
                )}
              </div>

              <div className="message-content-wrapper">
                {/* Collapsible Reasoning & Tool Execution Trace */}
                {msg.role === 'agent' && msg.reasoning && msg.reasoning.length > 0 && (
                  <ReasoningAccordion steps={msg.reasoning} />
                )}

                {/* Final Markdown Answer + Embedded Chart.js Cards */}
                {(msg.content || (msg.role === 'agent' && !msg.content && !isLoading)) && (
                  <div className="message-bubble">
                    {msg.role === 'agent' ? (
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={MarkdownComponents}>
                        {typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content, null, 2)}
                      </ReactMarkdown>
                    ) : (
                      <p>{typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content, null, 2)}</p>
                    )}
                  </div>
                )}

                {/* Live Stream Pulsing Indicator */}
                {msg.role === 'agent' && isLoading && !msg.content && idx === messages.length - 1 && (
                  <div className="streaming-pulse-indicator">
                    <span className="pulse-dot"></span>
                    <span>Analyzing stats query and executing semantic routing...</span>
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Suggestion Pills & Input Bar */}
      <footer className="footer-dock">
        {messages.length <= 2 && (
          <div className="suggestions-bar">
            {SUGGESTIONS.map((item, sIdx) => (
              <button
                key={sIdx}
                className="suggestion-pill"
                onClick={() => sendQuery(item.query)}
                disabled={isLoading}
              >
                <span className="pill-icon">{item.icon}</span>
                <span className="pill-text">{item.label}</span>
              </button>
            ))}
          </div>
        )}

        <form onSubmit={handleSubmit} className="input-form-bar">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about NBA stats, player comparisons, or request a chart (e.g. 'LeBron vs Curry scoring trend with a chart')..."
            className="enterprise-input"
            disabled={isLoading}
          />
          <button
            type="submit"
            className="enterprise-send-btn"
            disabled={isLoading || !input.trim()}
            title="Send Query"
          >
            <Send size={18} />
          </button>
        </form>
      </footer>
    </div>
  );
}

export default App;
