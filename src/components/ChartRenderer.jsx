import React, { useMemo, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadialLinearScale,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Bar, Line, Radar, Doughnut, Scatter, Bubble } from 'react-chartjs-2';
import { BarChart3, LineChart, PieChart, Table as TableIcon, ScatterChart as ScatterIcon, Share2 } from 'lucide-react';
import { NetworkGraph } from './NetworkGraph';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadialLinearScale,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const PALETTE = [
  { bg: 'rgba(59, 130, 246, 0.75)', border: '#3b82f6' },   // Blue
  { bg: 'rgba(16, 185, 129, 0.75)', border: '#10b981' },   // Emerald
  { bg: 'rgba(249, 115, 22, 0.75)', border: '#f97316' },   // Orange
  { bg: 'rgba(168, 85, 247, 0.75)', border: '#a855f7' },   // Purple
  { bg: 'rgba(236, 72, 153, 0.75)', border: '#ec4899' },   // Pink
  { bg: 'rgba(234, 179, 8, 0.75)', border: '#eab308' },    // Yellow
];

export function ChartRenderer({ config }) {
  const [showTable, setShowTable] = useState(false);

  // If configuration represents a Node-Link Network Graph, delegate immediately
  if (config && (config.type === 'network' || config.type === 'graph' || Array.isArray(config.nodes))) {
    return <NetworkGraph config={config} />;
  }

  const { chartData, options, chartType } = useMemo(() => {
    if (!config || !config.data) {
      return { chartData: null, options: {}, chartType: 'bar' };
    }

    const type = (config.type || 'bar').toLowerCase();

    // Enhance datasets with modern styling
    const styledDatasets = (config.data.datasets || []).map((ds, idx) => {
      const color = PALETTE[idx % PALETTE.length];
      const isLine = type === 'line';
      const isRadar = type === 'radar';
      const isScatter = type === 'scatter' || type === 'bubble';

      return {
        ...ds,
        backgroundColor: ds.backgroundColor || (isLine ? 'rgba(59, 130, 246, 0.15)' : color.bg),
        borderColor: ds.borderColor || color.border,
        borderWidth: ds.borderWidth || (isLine ? 3 : 1.5),
        pointBackgroundColor: ds.pointBackgroundColor || color.border,
        pointBorderColor: '#ffffff',
        pointRadius: isLine || isRadar ? 4 : (isScatter ? 6 : undefined),
        pointHoverRadius: isScatter ? 9 : 6,
        tension: isLine ? 0.35 : undefined,
        fill: isLine,
      };
    });

    const defaultOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: Boolean(config.data.datasets && config.data.datasets.length > 1),
          position: 'top',
          labels: {
            color: '#cbd5e1',
            font: { family: 'Inter', size: 12, weight: '500' },
            boxWidth: 14,
            usePointStyle: true,
          },
        },
        title: {
          display: Boolean(config.options?.plugins?.title?.text || config.title),
          text: config.options?.plugins?.title?.text || config.title || '',
          color: '#f8fafc',
          font: { family: 'Inter', size: 14, weight: '600' },
          padding: { bottom: 16 },
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleColor: '#f8fafc',
          bodyColor: '#cbd5e1',
          borderColor: 'rgba(255, 255, 255, 0.12)',
          borderWidth: 1,
          padding: 10,
          boxPadding: 4,
          usePointStyle: true,
          callbacks: {
            label: function (context) {
              if (type === 'scatter' || type === 'bubble') {
                const raw = context.raw || {};
                const label = raw.label || context.dataset.label || '';
                return `${label ? label + ': ' : ''}(x: ${raw.x}, y: ${raw.y}${raw.r ? ', size: ' + raw.r : ''})`;
              }
              const label = context.dataset.label || '';
              const value = context.parsed.y !== undefined ? context.parsed.y : context.parsed;
              return `${label ? label + ': ' : ''}${value}`;
            },
          },
        },
      },
      scales: type === 'radar' || type === 'doughnut' || type === 'pie' ? undefined : {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } },
          title: config.options?.scales?.x?.title ? {
            display: true,
            text: config.options.scales.x.title.text,
            color: '#94a3b8',
            font: { family: 'Inter', size: 12, weight: '500' }
          } : undefined
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } },
          title: config.options?.scales?.y?.title ? {
            display: true,
            text: config.options.scales.y.title.text,
            color: '#94a3b8',
            font: { family: 'Inter', size: 12, weight: '500' }
          } : undefined
        },
      },
    };

    const mergedOptions = {
      ...defaultOptions,
      ...config.options,
      plugins: {
        ...defaultOptions.plugins,
        ...config.options?.plugins,
      },
    };

    return {
      chartData: { ...config.data, datasets: styledDatasets },
      options: mergedOptions,
      chartType: type,
    };
  }, [config]);

  if (!chartData) {
    return null;
  }

  const renderChart = () => {
    switch (chartType) {
      case 'line':
        return <Line data={chartData} options={options} />;
      case 'radar':
        return <Radar data={chartData} options={options} />;
      case 'doughnut':
      case 'pie':
        return <Doughnut data={chartData} options={options} />;
      case 'scatter':
        return <Scatter data={chartData} options={options} />;
      case 'bubble':
        return <Bubble data={chartData} options={options} />;
      case 'bar':
      default:
        return <Bar data={chartData} options={options} />;
    }
  };

  const labels = config.data.labels || [];
  const datasets = config.data.datasets || [];

  return (
    <div className="chart-card-container">
      <div className="chart-card-header">
        <div className="chart-card-title">
          {chartType === 'line' && <LineChart size={16} className="chart-icon" />}
          {chartType === 'bar' && <BarChart3 size={16} className="chart-icon" />}
          {chartType === 'radar' && <PieChart size={16} className="chart-icon" />}
          {(chartType === 'scatter' || chartType === 'bubble') && <ScatterIcon size={16} className="chart-icon" />}
          <span>{config.title || (config.options?.plugins?.title?.text) || 'Statistical Visualization'}</span>
        </div>
        <button
          className="chart-toggle-btn"
          onClick={() => setShowTable(!showTable)}
          title="Toggle Chart / Table View"
        >
          {showTable ? <BarChart3 size={14} /> : <TableIcon size={14} />}
          <span>{showTable ? 'Chart View' : 'Table View'}</span>
        </button>
      </div>

      <div className="chart-card-body">
        {showTable ? (
          <div className="chart-data-table-wrapper">
            <table className="chart-data-table">
              <thead>
                <tr>
                  <th>Category / Point</th>
                  {datasets.map((ds, i) => (
                    <th key={i}>{ds.label || `Metric ${i + 1}`}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {labels.length > 0 ? labels.map((label, rowIdx) => (
                  <tr key={rowIdx}>
                    <td><strong>{label}</strong></td>
                    {datasets.map((ds, colIdx) => (
                      <td key={colIdx}>{ds.data?.[rowIdx] ?? '-'}</td>
                    ))}
                  </tr>
                )) : (
                  // For scatter or bubble datasets where data is array of objects {x, y}
                  datasets[0]?.data?.map((pt, ptIdx) => (
                    <tr key={ptIdx}>
                      <td><strong>{pt.label || `Point ${ptIdx + 1}`}</strong></td>
                      {datasets.map((ds, dsIdx) => (
                        <td key={dsIdx}>{JSON.stringify(ds.data?.[ptIdx])}</td>
                      ))}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ height: '320px', width: '100%', position: 'relative' }}>
            {renderChart()}
          </div>
        )}
      </div>
    </div>
  );
}

export function parseChartMarkdownBlock(rawContent) {
  try {
    return JSON.parse(rawContent.trim());
  } catch (e) {
    console.warn('Could not parse chart block JSON:', e);
    return null;
  }
}
