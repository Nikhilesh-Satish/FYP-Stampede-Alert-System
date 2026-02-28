import { ArrowRight, Shield, Zap, TrendingUp, MapPin } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Layout, Button, Card } from '../components';

export const Home = () => (
  <Layout>
    {/* Hero Section */}
    <section className="mb-16">
      <div className="grid lg:grid-cols-2 gap-12 items-center">
        <div>
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            Prevent Stampede Incidents with Real-Time Monitoring
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Advanced crowd density monitoring using AI-powered crowd counting technology. Detect dangerous situations instantly and alert authorities in real-time.
          </p>
          <div className="flex gap-4">
            <Link to="/signup">
              <Button variant="primary" className="flex items-center gap-2">
                Get Started <ArrowRight size={20} />
              </Button>
            </Link>
            <Link to="/about">
              <Button variant="ghost">Learn More</Button>
            </Link>
          </div>
        </div>
        <div className="bg-gradient-to-br from-primary-500 to-secondary-500 rounded-lg h-96 flex items-center justify-center text-white text-2xl font-bold">
          Live Crowd Monitoring Dashboard
        </div>
      </div>
    </section>

    {/* Features Grid */}
    <section className="mb-16">
      <h2 className="text-4xl font-bold text-gray-900 mb-12 text-center">
        Why Choose Stampede Alert?
      </h2>
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          {
            icon: <Shield className="w-8 h-8" />,
            title: 'Real-Time Alerts',
            description: 'Instant notifications when crowd density reaches dangerous levels',
          },
          {
            icon: <Zap className="w-8 h-8" />,
            title: 'AI-Powered Analysis',
            description: 'Advanced crowd counting with deep learning models',
          },
          {
            icon: <TrendingUp className="w-8 h-8" />,
            title: 'Live Statistics',
            description: 'Comprehensive analytics and historical data tracking',
          },
          {
            icon: <MapPin className="w-8 h-8" />,
            title: 'Multi-Camera Support',
            description: 'Monitor multiple locations from a single dashboard',
          },
        ].map((feature, idx) => (
          <Card key={idx}>
            <div className="text-primary-500 mb-4">{feature.icon}</div>
            <h3 className="text-lg font-bold text-gray-900 mb-2">{feature.title}</h3>
            <p className="text-gray-600 text-sm">{feature.description}</p>
          </Card>
        ))}
      </div>
    </section>

    {/* Stats Section */}
    <section className="mb-16 bg-white rounded-lg p-12">
      <h2 className="text-4xl font-bold text-gray-900 mb-12 text-center">
        Platform Statistics
      </h2>
      <div className="grid md:grid-cols-4 gap-8">
        {[
          { label: 'Active Cameras', value: '1,200+' },
          { label: 'Events Detected', value: '50K+' },
          { label: 'Lives Protected', value: '5M+' },
          { label: 'Uptime', value: '99.9%' },
        ].map((stat, idx) => (
          <div key={idx} className="text-center">
            <p className="text-4xl font-bold text-primary-500 mb-2">{stat.value}</p>
            <p className="text-gray-600">{stat.label}</p>
          </div>
        ))}
      </div>
    </section>

    {/* CTA Section */}
    <section className="bg-gradient-to-r from-primary-600 to-secondary-600 rounded-lg p-12 text-white text-center">
      <h2 className="text-4xl font-bold mb-4">
        Ready to Protect Your Events?
      </h2>
      <p className="text-xl mb-8 opacity-90">
        Join thousands of event organizers using Stampede Alert to keep crowds safe.
      </p>
      <Link to="/signup">
        <Button variant="primary" className="bg-white text-primary-600 hover:bg-gray-100">
          Start Free Trial
        </Button>
      </Link>
    </section>
  </Layout>
);
