import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Activity, Terminal, CheckCircle2, Code2 } from 'lucide-react';

export function ReasoningAccordion({ steps = [] }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!steps || steps.length === 0) return null;

  const toolCallsCount = steps.filter(s => s.type === 'tool_call').length;
  const thoughtsCount = steps.filter(s => s.type === 'thought').length;

  const getStepIcon = (type) => {
    switch (type) {
      case 'tool_call':
        return <Terminal size={14} className="step-icon tool-call" />;
      case 'tool_result':
        return <CheckCircle2 size={14} className="step-icon tool-result" />;
      case 'thought':
      default:
        return <Activity size={14} className="step-icon thought" />;
    }
  };

  return (
    <div className="reasoning-accordion">
      <button
        className="reasoning-accordion-header"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="accordion-title-left">
          {isOpen ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
          <span className="accordion-label">Agent Reasoning & Tool Routing Trace</span>
          <span className="accordion-badge">
            {toolCallsCount} Tool Call{toolCallsCount !== 1 ? 's' : ''} • {thoughtsCount} Step{thoughtsCount !== 1 ? 's' : ''}
          </span>
        </div>
        <span className="accordion-action">{isOpen ? 'Hide Trace' : 'View Trace'}</span>
      </button>

      {isOpen && (
        <div className="reasoning-accordion-body">
          {steps.map((step, idx) => {
            const displayContent = typeof step.content === 'object' && step.content !== null
              ? (step.type === 'intent' ? `Intent: ${step.content.question_type || 'Research Query'}` : JSON.stringify(step.content))
              : (step.content || (step.tool ? `Tool Execution: ${step.tool}` : 'Processing step'));
            
            const detailsData = step.details || step.params || (typeof step.content === 'object' ? step.content : null);

            return (
              <div key={idx} className={`trace-step ${step.type}`}>
                <div className="step-header">
                  {getStepIcon(step.type)}
                  <span className="step-content">{displayContent}</span>
                </div>
                {detailsData && (
                  <div className="step-details-card">
                    <div className="details-card-header">
                      <Code2 size={12} />
                      <span>{step.type === 'intent' ? 'Extracted Intent Contract' : (step.details ? 'Validated Ingress Parameters' : 'Parameters')}</span>
                    </div>
                    <pre className="details-card-code">
                      {typeof detailsData === 'string' ? detailsData : JSON.stringify(detailsData, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
