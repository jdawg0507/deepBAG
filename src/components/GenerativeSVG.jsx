import React, { useState, useRef } from 'react';
import { Sparkles, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';

/**
 * Generative SVG Visualizer.
 * Renders any vector graphic generated on-the-fly by the LLM:
 * - Shot charts / spatial coordinate maps
 * - Custom flowcharts / decision trees
 * - Entity state diagrams or system architectures
 * - Domain-agnostic 2D visual layouts
 */
export function GenerativeSVG({ svgContent, title }) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Sanitize and clean raw SVG string
  const cleanSvg = (svgContent || '')
    .trim()
    .replace(/^```(svg|xml)?/i, '')
    .replace(/```$/, '')
    .trim();

  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
  };

  const handleMouseUp = () => setIsDragging(false);

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div className="chart-card-container generative-svg-card">
      <div className="chart-card-header">
        <div className="chart-card-title">
          <Sparkles size={16} className="chart-icon" />
          <span>{title || 'Generative Vector Visualization'}</span>
          <span className="network-badge">Generative SVG</span>
        </div>
        <div className="network-controls">
          <button className="chart-toggle-btn" onClick={() => setZoom(z => Math.min(z + 0.2, 3))} title="Zoom In">
            <ZoomIn size={14} />
          </button>
          <button className="chart-toggle-btn" onClick={() => setZoom(z => Math.max(z - 0.2, 0.4))} title="Zoom Out">
            <ZoomOut size={14} />
          </button>
          <button className="chart-toggle-btn" onClick={resetView} title="Reset View">
            <RotateCcw size={14} />
          </button>
        </div>
      </div>

      <div
        className="generative-svg-viewport"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
      >
        <div
          className="generative-svg-inner"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'center center',
            transition: isDragging ? 'none' : 'transform 0.15s ease-out'
          }}
          dangerouslySetInnerHTML={{ __html: cleanSvg }}
        />
      </div>
    </div>
  );
}
