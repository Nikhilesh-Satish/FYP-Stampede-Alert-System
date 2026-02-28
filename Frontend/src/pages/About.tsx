import { Card, Layout } from '../components';

export const About = () => (
  <Layout>
    {/* Hero Section */}
    <section className="mb-16">
      <h1 className="text-5xl font-bold text-gray-900 mb-4">About Stampede Alert</h1>
      <p className="text-xl text-gray-600">
        Dedicated to preventing crowd tragedies through intelligent monitoring and real-time alerts.
      </p>
    </section>

    {/* Mission Section */}
    <section className="mb-16 grid lg:grid-cols-2 gap-12 items-center">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Our Mission</h2>
        <p className="text-gray-600 mb-4">
          Stampede Alert is dedicated to preventing crowd disasters through cutting-edge technology. We leverage artificial intelligence and real-time monitoring to detect dangerous crowd situations before they escalate.
        </p>
        <p className="text-gray-600 mb-4">
          Our system continuously monitors crowd density using advanced computer vision models, alerting event organizers and authorities instantly when dangerous thresholds are approached.
        </p>
      </div>
      <Card className="bg-primary-50 border-2 border-primary-200">
        <h3 className="text-2xl font-bold text-primary-700 mb-4">Our Vision</h3>
        <p className="text-gray-700">
          A world where crowd-related tragedies are preventable through intelligent monitoring and rapid response systems, making large public gatherings safer for everyone.
        </p>
      </Card>
    </section>

    {/* Technology Section */}
    <section className="mb-16">
      <h2 className="text-3xl font-bold text-gray-900 mb-8">Our Technology</h2>
      <div className="grid md:grid-cols-3 gap-6">
        {[
          {
            title: 'CSRNet Model',
            description: 'Deep learning architecture trained on large-scale crowd datasets for accurate density estimation',
          },
          {
            title: 'Real-Time Processing',
            description: 'Processes video feeds in real-time with sub-second latency for instant alerts',
          },
          {
            title: 'Multi-Camera Fusion',
            description: 'Combines data from multiple camera angles for comprehensive crowd analysis',
          },
        ].map((tech, idx) => (
          <Card key={idx}>
            <h3 className="text-lg font-bold text-gray-900 mb-2">{tech.title}</h3>
            <p className="text-gray-600">{tech.description}</p>
          </Card>
        ))}
      </div>
    </section>

    {/* Team Section */}
    <section className="mb-16">
      <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">Our Team</h2>
      <p className="text-center text-gray-600 mb-8 max-w-2xl mx-auto">
        Built by a dedicated team of machine learning engineers, security experts, and public safety professionals committed to preventing crowd disasters.
      </p>
      <div className="grid md:grid-cols-3 gap-6">
        {[
          { name: 'Dr. Sarah Chen', role: 'AI Lead' },
          { name: 'John Smith', role: 'Security Expert' },
          { name: 'Lisa Kumar', role: 'Operations Lead' },
        ].map((member, idx) => (
          <Card key={idx} className="text-center">
            <div className="w-20 h-20 bg-primary-500 rounded-full mx-auto mb-4"></div>
            <h3 className="text-lg font-bold text-gray-900">{member.name}</h3>
            <p className="text-gray-600">{member.role}</p>
          </Card>
        ))}
      </div>
    </section>

    {/* Contact Section */}
    <section className="bg-white rounded-lg p-12 text-center">
      <h2 className="text-3xl font-bold text-gray-900 mb-4">Get In Touch</h2>
      <p className="text-gray-600 mb-8">
        Have questions about our system? Contact us anytime.
      </p>
      <div className="space-y-2">
        <p className="text-gray-700">
          <span className="font-semibold">Email:</span> support@stampedealertsystem.com
        </p>
        <p className="text-gray-700">
          <span className="font-semibold">Phone:</span> +1 (555) 123-4567
        </p>
      </div>
    </section>
  </Layout>
);
