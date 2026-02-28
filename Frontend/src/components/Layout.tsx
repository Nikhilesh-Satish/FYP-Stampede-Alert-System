import { ReactNode } from 'react';
import { Navbar } from './Navbar';

interface LayoutProps {
  children: ReactNode;
  sidebar?: ReactNode;
}

export const Layout = ({ children, sidebar }: LayoutProps) => (
  <div className="min-h-screen bg-gray-50">
    <Navbar />
    <div className="flex">
      {sidebar && (
        <aside className="hidden lg:block w-64 bg-white border-r border-gray-200">
          {sidebar}
        </aside>
      )}
      <main className="flex-1">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </div>
      </main>
    </div>
  </div>
);

interface SidebarItemProps {
  icon: ReactNode;
  label: string;
  active?: boolean;
  onClick?: () => void;
}

export const SidebarItem = ({ icon, label, active, onClick }: SidebarItemProps) => (
  <button
    onClick={onClick}
    className={`w-full flex items-center gap-3 px-4 py-3 text-left font-medium transition ${
      active
        ? 'bg-primary-100 text-primary-700 border-r-4 border-primary-500'
        : 'text-gray-600 hover:bg-gray-100'
    }`}
  >
    <span className="text-xl">{icon}</span>
    <span>{label}</span>
  </button>
);
