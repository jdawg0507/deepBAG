import React, { useState, useEffect } from 'react';
import { ShieldCheck, Activity, Cpu, Database, RefreshCw } from 'lucide-react';

export function TelemetryBanner() {
  const [telemetry, setTelemetry] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchStatus = async () => {
    try {
      setIsRefreshing(true);
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setTelemetry(data);
      }
    } catch (e) {
      console.warn('Telemetry fetch error:', e);
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 8000);
    return () => clearInterval(interval);
  }, []);

  const cache = telemetry?.cache || {};
  const hitRatioPercent = Math.round((cache.hit_ratio || 0) * 100);

  return (
    <div className="telemetry-banner">
      <div className="telemetry-left">
        <div className="gateway-badge">
          <span className="status-dot-pulse"></span>
          <span className="gateway-name">Gateway v{telemetry?.gateway_version || '2.0'}</span>
        </div>
        <div className="telemetry-chip">
          <Database size={13} className="chip-icon" />
          <span>Cache: <strong>{cache.backend || 'redis'}</strong></span>
        </div>
        <div className="telemetry-chip">
          <Activity size={13} className="chip-icon" />
          <span>Hit Rate: <strong>{hitRatioPercent}%</strong> ({cache.hits || 0}/{cache.total_lookups || 0})</span>
        </div>
        <div className="telemetry-chip rate-limit-chip">
          <ShieldCheck size={13} className="chip-icon" />
          <span>Sliding-Window Token Bucket: <strong>Active</strong></span>
        </div>
      </div>

      <div className="telemetry-right">
        <button
          className={`telemetry-refresh-btn ${isRefreshing ? 'spinning' : ''}`}
          onClick={fetchStatus}
          title="Refresh Gateway Telemetry"
        >
          <RefreshCw size={13} />
        </button>
      </div>
    </div>
  );
}
