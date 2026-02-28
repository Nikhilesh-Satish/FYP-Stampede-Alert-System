import { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
}

export const Card = ({ children, className = '' }: CardProps) => (
  <div className={`card ${className}`}>
    {children}
  </div>
);

interface BadgeProps {
  children: ReactNode;
  variant?: 'primary' | 'secondary' | 'success' | 'danger' | 'warning';
}

export const Badge = ({ children, variant = 'primary' }: BadgeProps) => {
  const variants = {
    primary: 'bg-primary-100 text-primary-800',
    secondary: 'bg-secondary-100 text-secondary-800',
    success: 'bg-green-100 text-green-800',
    danger: 'bg-red-100 text-red-800',
    warning: 'bg-yellow-100 text-yellow-800',
  };

  return (
    <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${variants[variant]}`}>
      {children}
    </span>
  );
};

interface AlertProps {
  children: ReactNode;
  type?: 'success' | 'error' | 'warning' | 'info';
  onClose?: () => void;
}

export const Alert = ({ children, type = 'info', onClose }: AlertProps) => {
  const styles = {
    success: 'bg-green-50 border-green-200 text-green-800',
    error: 'bg-red-50 border-red-200 text-red-800',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    info: 'bg-blue-50 border-blue-200 text-blue-800',
  };

  return (
    <div className={`border rounded-lg p-4 flex justify-between items-start gap-4 ${styles[type]}`}>
      <div className="text-sm">{children}</div>
      {onClose && (
        <button
          onClick={onClose}
          className="text-lg font-bold hover:opacity-70 transition"
        >
          ×
        </button>
      )}
    </div>
  );
};

interface StatProps {
  label: string;
  value: string | number;
  change?: number;
}

export const StatCard = ({ label, value, change }: StatProps) => (
  <Card className="text-center">
    <p className="text-gray-500 text-sm mb-2">{label}</p>
    <p className="text-3xl font-bold text-gray-800 mb-2">{value}</p>
    {change !== undefined && (
      <p className={`text-sm font-semibold ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
        {change >= 0 ? '+' : ''}{change}%
      </p>
    )}
  </Card>
);
