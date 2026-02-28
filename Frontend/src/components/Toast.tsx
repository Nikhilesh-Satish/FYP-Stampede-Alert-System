import { ReactNode } from 'react';
import { AlertCircle, CheckCircle, Info, AlertTriangle } from 'lucide-react';

interface ToastProps {
  message: string;
  type?: 'success' | 'error' | 'warning' | 'info';
}

const typeStyles = {
  success: { bg: 'bg-green-100', border: 'border-green-500', icon: CheckCircle },
  error: { bg: 'bg-red-100', border: 'border-red-500', icon: AlertCircle },
  warning: { bg: 'bg-yellow-100', border: 'border-yellow-500', icon: AlertTriangle },
  info: { bg: 'bg-blue-100', border: 'border-blue-500', icon: Info },
};

export const Toast = ({ message, type = 'info' }: ToastProps) => {
  const { bg, border, icon: Icon } = typeStyles[type];

  return (
    <div className={`${bg} ${border} border-l-4 p-4 rounded flex items-center gap-3`}>
      <Icon size={20} />
      <p>{message}</p>
    </div>
  );
};
