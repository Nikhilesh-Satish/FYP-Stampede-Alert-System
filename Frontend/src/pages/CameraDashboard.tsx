import { useState } from 'react';
import { AlertTriangle, Activity, TrendingUp, Eye } from 'lucide-react';
import { Layout, SidebarItem, Button, Card, StatCard, Badge, Alert } from '../components';

export const CameraDashboard = () => {
  const [activeCamera, setActiveCamera] = useState(1);

  const cameras = [
    { id: 1, name: 'Main Entrance', status: 'active' },
    { id: 2, name: 'East Gate', status: 'active' },
    { id: 3, name: 'West Parking', status: 'inactive' },
  ];

  const selectedCamera = cameras.find((c) => c.id === activeCamera)!;

  const sidebar = (
    <div className="p-4 space-y-2">
      {cameras.map((camera) => (
        <SidebarItem
          key={camera.id}
          icon={<Eye size={20} />}
          label={camera.name}
          active={activeCamera === camera.id}
          onClick={() => setActiveCamera(camera.id)}
        />
      ))}
    </div>
  );

  const crowdData = {
    1: { current: 156, max: 300, alerts: 2, trend: 12 },
    2: { current: 89, max: 250, alerts: 0, trend: -5 },
    3: { current: 0, max: 0, alerts: 0, trend: 0 },
  };

  const data = crowdData[activeCamera as keyof typeof crowdData];

  const alerts = [
    { id: 1, severity: 'high', message: 'High crowd density detected', time: '2 mins ago' },
    { id: 2, severity: 'medium', message: 'Moderate crowd level', time: '15 mins ago' },
    { id: 3, severity: 'low', message: 'Normal crowd level', time: '30 mins ago' },
  ];

  return (
    <Layout sidebar={sidebar}>
      <div>
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            {selectedCamera.name}
          </h1>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${selectedCamera.status === 'active' ? 'bg-green-500' : 'bg-gray-400'}`} />
            <p className="text-gray-600 capitalize">{selectedCamera.status}</p>
          </div>
        </div>

        {/* Live Feed */}
        <Card className="mb-8 h-96 bg-gray-900 flex items-center justify-center">
          <div className="text-center text-white">
            <Eye size={48} className="mx-auto mb-4 opacity-50" />
            <p className="text-xl opacity-75">Live Feed Stream</p>
            <p className="text-sm opacity-50 mt-2">RTSP Feed from Camera</p>
          </div>
        </Card>

        {/* Stats */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard label="Current Count" value={data.current} change={data.trend} />
          <StatCard label="Max Capacity" value={data.max} />
          <StatCard label="Occupancy" value={`${Math.round((data.current / data.max) * 100)}%`} />
          <StatCard label="Active Alerts" value={data.alerts} />
        </div>

        {/* Alert Status */}
        {data.current > data.max * 0.8 && (
          <Alert type="warning" className="mb-8">
            <strong>Warning:</strong> Crowd density is approaching maximum capacity. Please monitor closely.
          </Alert>
        )}

        {data.current > data.max * 0.9 && (
          <Alert type="error" className="mb-8">
            <strong>Critical:</strong> Crowd density has reached critical levels. Immediate action required!
          </Alert>
        )}

        {/* Recent Alerts */}
        <Card>
          <h2 className="text-xl font-bold text-gray-900 mb-4">Alert History</h2>
          <div className="space-y-3">
            {alerts.map((alert) => (
              <div key={alert.id} className="flex items-start gap-4 p-3 bg-gray-50 rounded-lg">
                <div className={`w-2 h-2 mt-1.5 rounded-full ${
                  alert.severity === 'high' ? 'bg-red-500' :
                  alert.severity === 'medium' ? 'bg-yellow-500' :
                  'bg-green-500'
                }`} />
                <div className="flex-1">
                  <p className="text-sm text-gray-600">{alert.message}</p>
                  <p className="text-xs text-gray-500 mt-1">{alert.time}</p>
                </div>
                <Badge variant={alert.severity === 'high' ? 'danger' : alert.severity === 'medium' ? 'warning' : 'success'}>
                  {alert.severity}
                </Badge>
              </div>
            ))}
          </div>
        </Card>

        {/* Actions */}
        <div className="mt-8 flex gap-4">
          <Button variant="primary">Export Report</Button>
          <Button variant="secondary">Configure Alerts</Button>
          <Button variant="ghost">View Advanced Analytics</Button>
        </div>
      </div>
    </Layout>
  );
};
