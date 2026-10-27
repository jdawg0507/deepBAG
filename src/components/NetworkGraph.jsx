import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Share2, ZoomIn, ZoomOut, RotateCcw, Maximize2 } from 'lucide-react';

/**
 * Universal Interactive Node-Link Network Graph Renderer.
 * Fully domain-agnostic: works for passing networks, financial flows,
 * microservice dependencies, supply chains, or entity relationships.
 *
 * Spec format:
 * {
 *   "title": "Entity Relationship Graph",
 *   "nodes": [
 *     { "id": "A", "label": "Node A", "value": 15, "group": "cluster1", "color": "#38bdf8", "subtext": "Detail" }
 *   ],
 *   "links": [
 *     { "source": "A", "target": "B", "weight": 5, "label": "5.0", "color": "#64748b" }
 *   ]
 * }
 */
export function NetworkGraph({ config }) {
  const containerRef = useRef(null);
  const [positions, setPositions] = useState({});
  const [hoveredNode, setHoveredNode] = useState(null);
  const [hoveredLink, setHoveredLink] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const width = 600;
  const height = 340;
  const centerX = width / 2;
  const centerY = height / 2;

  const nodes = useMemo(() => config?.nodes || [], [config]);
  const links = useMemo(() => config?.links || [], [config]);
  const title = config?.title || 'Interactive Network Graph';

  // Compute Layout: iterative spring-electrical or circular embedding
  useEffect(() => {
    if (!nodes.length) return;

    const initial = {};
    const n = nodes.length;

    // Check if nodes have predefined x, y (e.g. coordinate maps or shot grids normalized to 0-100)
    const hasCoordinates = nodes.every(nd => nd.x !== undefined && nd.y !== undefined);

    if (hasCoordinates) {
      nodes.forEach(nd => {
        initial[nd.id] = {
          x: (nd.x / 100) * (width - 80) + 40,
          y: (nd.y / 100) * (height - 80) + 40,
        };
      });
      setPositions(initial);
      return;
    }

    // Circular initialization
    nodes.forEach((nd, i) => {
      const angle = (2 * Math.PI * i) / n;
      const radius = Math.min(width, height) * 0.35;
      initial[nd.id] = {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
      };
    });

    // Run 50 iterations of lightweight force relaxation
    const pos = { ...initial };
    const k = Math.sqrt((width * height) / Math.max(1, n));

    for (let iter = 0; iter < 45; iter++) {
      const disp = {};
      nodes.forEach(u => { disp[u.id] = { x: 0, y: 0 }; });

      // Repulsion between all node pairs
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const u = nodes[i].id;
          const v = nodes[j].id;
          const dx = pos[u].x - pos[v].x;
          const dy = pos[u].y - pos[v].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = (k * k) / dist;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;

          disp[u].x += fx;
          disp[u].y += fy;
          disp[v].x -= fx;
          disp[v].y -= fy;
        }
      }

      // Attraction along links
      links.forEach(link => {
        const u = typeof link.source === 'object' ? link.source.id : link.source;
        const v = typeof link.target === 'object' ? link.target.id : link.target;
        if (pos[u] && pos[v]) {
          const dx = pos[u].x - pos[v].x;
          const dy = pos[u].y - pos[v].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = (dist * dist) / k;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;

          disp[u].x -= fx;
          disp[u].y -= fy;
          disp[v].x += fx;
          disp[v].y += fy;
        }
      });

      // Gravity towards center
      nodes.forEach(u => {
        const dx = centerX - pos[u.id].x;
        const dy = centerY - pos[u.id].y;
        disp[u.id].x += dx * 0.05;
        disp[u.id].y += dy * 0.05;
      });

      // Update positions with damping
      const temp = Math.max(1, 25 * (1 - iter / 45));
      nodes.forEach(u => {
        const d = disp[u.id];
        const dist = Math.sqrt(d.x * d.x + d.y * d.y) || 1;
        pos[u.id].x += (d.x / dist) * Math.min(dist, temp);
        pos[u.id].y += (d.y / dist) * Math.min(dist, temp);
        // Constrain within bounds
        pos[u.id].x = Math.max(35, Math.min(width - 35, pos[u.id].x));
        pos[u.id].y = Math.max(35, Math.min(height - 35, pos[u.id].y));
      });
    }

    setPositions(pos);
  }, [nodes, links]);

  // Pan interaction handlers
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
    <div className="chart-card-container">
      <div className="chart-card-header">
        <div className="chart-card-title">
          <Share2 size={16} className="chart-icon" />
          <span>{title}</span>
          <span className="network-badge">{nodes.length} nodes · {links.length} connections</span>
        </div>
        <div className="network-controls">
          <button className="chart-toggle-btn" onClick={() => setZoom(z => Math.min(z + 0.2, 2.5))} title="Zoom In">
            <ZoomIn size={14} />
          </button>
          <button className="chart-toggle-btn" onClick={() => setZoom(z => Math.max(z - 0.2, 0.5))} title="Zoom Out">
            <ZoomOut size={14} />
          </button>
          <button className="chart-toggle-btn" onClick={resetView} title="Reset View">
            <RotateCcw size={14} />
          </button>
        </div>
      </div>

      <div
        className="network-canvas-wrapper"
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
      >
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="network-svg"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <marker
              id="arrowhead"
              viewBox="0 0 10 10"
              refX="18"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 10 5 L 0 9 z" fill="rgba(148, 163, 184, 0.5)" />
            </marker>
            <filter id="nodeGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`} transform-origin="center">
            {/* Render Links */}
            {links.map((link, idx) => {
              const srcId = typeof link.source === 'object' ? link.source.id : link.source;
              const tgtId = typeof link.target === 'object' ? link.target.id : link.target;
              const p1 = positions[srcId];
              const p2 = positions[tgtId];
              if (!p1 || !p2) return null;

              const isConnectedToHover = hoveredNode && (srcId === hoveredNode.id || tgtId === hoveredNode.id);
              const isLinkHovered = hoveredLink === idx;
              const strokeWidth = link.weight ? Math.max(1.5, Math.min(link.weight, 6)) : 2;
              const strokeColor = isLinkHovered
                ? '#38bdf8'
                : isConnectedToHover
                ? '#f97316'
                : (link.color || 'rgba(148, 163, 184, 0.35)');

              return (
                <g key={`link-${idx}`}>
                  <line
                    x1={p1.x}
                    y1={p1.y}
                    x2={p2.x}
                    y2={p2.y}
                    stroke={strokeColor}
                    strokeWidth={strokeWidth}
                    strokeDasharray={link.dashed ? '4,4' : undefined}
                    markerEnd="url(#arrowhead)"
                    onMouseEnter={() => setHoveredLink(idx)}
                    onMouseLeave={() => setHoveredLink(null)}
                    style={{ transition: 'stroke 0.2s, stroke-width 0.2s' }}
                  />
                  {link.label && (
                    <text
                      x={(p1.x + p2.x) / 2}
                      y={(p1.y + p2.y) / 2 - 4}
                      fill="#94a3b8"
                      fontSize="9"
                      fontFamily="Inter, sans-serif"
                      textAnchor="middle"
                      className="network-edge-label"
                    >
                      {link.label}
                    </text>
                  )}
                </g>
              );
            })}

            {/* Render Nodes */}
            {nodes.map(node => {
              const pos = positions[node.id];
              if (!pos) return null;

              const isHovered = hoveredNode?.id === node.id;
              const radius = Math.max(14, Math.min(26, (node.value || 10) * 1.4));
              const nodeColor = node.color || (node.group === 'target' ? '#f97316' : '#3b82f6');

              return (
                <g
                  key={`node-${node.id}`}
                  transform={`translate(${pos.x}, ${pos.y})`}
                  onMouseEnter={() => setHoveredNode(node)}
                  onMouseLeave={() => setHoveredNode(null)}
                  className="network-node-group"
                  style={{ cursor: 'pointer' }}
                >
                  <circle
                    r={radius}
                    fill={nodeColor}
                    fillOpacity={0.88}
                    stroke={isHovered ? '#ffffff' : 'rgba(255, 255, 255, 0.3)'}
                    strokeWidth={isHovered ? 2.5 : 1.5}
                    filter={isHovered ? 'url(#nodeGlow)' : undefined}
                  />
                  <text
                    y={radius + 12}
                    textAnchor="middle"
                    fill="#f1f5f9"
                    fontSize="11"
                    fontWeight="600"
                    fontFamily="Inter, sans-serif"
                    className="network-node-text"
                  >
                    {node.label || node.id}
                  </text>
                  {node.subtext && (
                    <text
                      y={radius + 22}
                      textAnchor="middle"
                      fill="#94a3b8"
                      fontSize="8.5"
                      fontFamily="Inter, sans-serif"
                    >
                      {node.subtext}
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        </svg>

        {/* Floating Tooltip Card */}
        {hoveredNode && (
          <div className="network-tooltip">
            <div className="network-tooltip-title">{hoveredNode.label || hoveredNode.id}</div>
            {hoveredNode.subtext && <div className="network-tooltip-sub">{hoveredNode.subtext}</div>}
            {hoveredNode.value !== undefined && (
              <div className="network-tooltip-val">Metric Value: <strong>{hoveredNode.value}</strong></div>
            )}
            {hoveredNode.group && (
              <div className="network-tooltip-group">Group: <span>{hoveredNode.group}</span></div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
