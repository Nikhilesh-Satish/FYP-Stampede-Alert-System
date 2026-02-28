import { useState } from 'react';
import { Plus, Edit2, Trash2, AlertTriangle, Activity, Users, Video } from 'lucide-react';
import { Layout, SidebarItem, Button, Card, StatCard, Badge, Modal, FormInput } from '../components';

export const AdminDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [showAddCamera, setShowAddCamera] = useState(false);
  const [newCameraName, setNewCameraName] = useState('');

  const cameras = [
    { id: 1, name: 'Main Entrance', location: 'Building A', status: 'active', count: 156 },
    { id: 2, name: 'East Gate', location: 'Building B', status: 'active', count: 89 },
    { id: 3, name: 'West Parking', location: 'Building C', status: 'inactive', count: 0 },
  ];

  const alerts = [
    { id: 1, camera: 'Main Entrance', severity: 'high', message: 'High crowd density detected', time: '2 mins ago' },
    { id: 2, camera: 'East Gate', severity: 'medium', message: 'Moderate crowd detected', time: '15 mins ago' },
    { id: 3, camera: 'Main Entrance', severity: 'low', message: 'Normal crowd level', time: '1 hour ago' },
  ];

  const sidebar = (
    <div className="p-4 space-y-2">
      <SidebarItem
        icon={<Activity size={20} />}
        label="Overview"
        active={activeTab === 'overview'}
        onClick={() => setActiveTab('overview')}
      />
      <SidebarItem
        icon={<Video size={20} />}
        label="Cameras"
        active={activeTab === 'cameras'}
        onClick={() => setActiveTab('cameras')}
      />
      <SidebarItem
        icon={<AlertTriangle size={20} />}
        label="Alerts"
        active={activeTab === 'alerts'}
        onClick={() => setActiveTab('alerts')}
      />
      <SidebarItem
        icon={<Users size={20} />}
        label="Users"
        active={activeTab === 'users'}
        onClick={() => setActiveTab('users')}
      />
    </div>
  );

  return (
    <Layout sidebar={sidebar}>
      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-8">Dashboard Overview</h1>
          
          {/* Stats */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <StatCard label="Active Cameras" value={2} />
            <StatCard label="Total Crowd Count" value={245} change={12} />
            <StatCard label="Active Alerts" value={1} />
            <StatCard label="System Uptime" value="99.9%" />
          </div>

          {/* Recent Alerts */}
          <Card>
            <h2 className="text-xl font-bold text-gray-900 mb-4">Recent Alerts</h2>
            <div className="space-y-3">
              {alerts.map((alert) => (
                <div key={alert.id} className="flex items-start gap-4 p-3 bg-gray-50 rounded-lg">
                  <div className={`w-2 h-2 mt-1.5 rounded-full ${
                    alert.severity === 'high' ? 'bg-red-500' :
                    alert.severity === 'medium' ? 'bg-yellow-500' :
                    'bg-green-500'
                  }`} />
                  <div className="flex-1">
                    <p className="font-semibold text-gray-900">{alert.camera}</p>
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
        </div>
      )}

      {/* Cameras Tab */}
      {activeTab === 'cameras' && (
        <div>
          <div className="flex justify-between items-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900">Manage Cameras</h1>
            <Button variant="primary" onClick={() => setShowAddCamera(true)}>
              <Plus size={20} className="mr-2" /> Add Camera
            </Button>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {cameras.map((camera) => (
              <Card key={camera.id}>
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-lg font-bold text-gray-900">{camera.name}</h3>
                    <p className="text-sm text-gray-600">{camera.location}</p>
                  </div>
                  <Badge variant={camera.status === 'active' ? 'success' : 'danger'}>
                    {camera.status}
                  </Badge>
                </div>
                <p className="text-3xl font-bold text-primary-500 mb-4">{camera.count}</p>
                <p className="text-sm text-gray-600 mb-4">Current crowd count</p>
                <div className="flex gap-2">
                  <Button variant="secondary" size="sm" className="flex-1">
                    <Edit2 size={16} /> Edit
                  </Button>
                  <Button variant="danger" size="sm" className="flex-1">
                    <Trash2 size={16} /> Delete
                  </Button>
                </div>
              </Card>
            ))}
          </div>

          <Modal
            isOpen={showAddCamera}
            onClose={() => {
              setShowAddCamera(false);
              setNewCameraName('');
            }}
            title="Add New Camera"
          >
            <div className="space-y-4">
              <FormInput
                label="Camera Name"
                placeholder="e.g., Main Entrance"
                value={newCameraName}
                onChange={setNewCameraName}
              />
              <FormInput label="Location" placeholder="e.g., Building A" />
              <FormInput label="Camera URL" placeholder="rtsp://..." />
              <div className="flex gap-2 justify-end pt-4">
                <Button variant="ghost" onClick={() => setShowAddCamera(false)}>
                  Cancel
                </Button>
                <Button variant="primary">Add Camera</Button>
              </div>
            </div>
          </Modal>
        </div>
      )}

      {/* Alerts Tab */}
      {activeTab === 'alerts' && (
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-8">Alert History</h1>
          <Card>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">Camera</th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">Message</th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">Severity</th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {alerts.map((alert) => (
                    <tr key={alert.id} className="border-b border-gray-200 hover:bg-gray-50">
                      <td className="px-4 py-3 text-sm text-gray-900">{alert.camera}</td>
                      <td className="px-4 py-3 text-sm text-gray-600">{alert.message}</td>
                      <td className="px-4 py-3">
                        <Badge variant={alert.severity === 'high' ? 'danger' : alert.severity === 'medium' ? 'warning' : 'success'}>
                          {alert.severity}
                        </Badge>
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-600">{alert.time}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Users Tab */}
      {activeTab === 'users' && (
        <div>
          <div className="flex justify-between items-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900">Manage Users</h1>
            <Button variant="primary">
              <Plus size={20} className="mr-2" /> Add User
            </Button>
          </div>

          <Card>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">Email</th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">Role</th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">Status</th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { email: 'john@example.com', role: 'Admin', status: 'Active' },
                    { email: 'jane@example.com', role: 'Operator', status: 'Active' },
                    { email: 'bob@example.com', role: 'Operator', status: 'Inactive' },
                  ].map((user, idx) => (
                    <tr key={idx} className="border-b border-gray-200 hover:bg-gray-50">
                      <td className="px-4 py-3 text-sm text-gray-900">{user.email}</td>
                      <td className="px-4 py-3 text-sm text-gray-600">{user.role}</td>
                      <td className="px-4 py-3">
                        <Badge variant={user.status === 'Active' ? 'success' : 'danger'}>
                          {user.status}
                        </Badge>
                      </td>
                      <td className="px-4 py-3 text-sm flex gap-2">
                        <button className="text-primary-600 hover:text-primary-700">Edit</button>
                        <button className="text-red-600 hover:text-red-700">Delete</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </Layout>
  );
};
