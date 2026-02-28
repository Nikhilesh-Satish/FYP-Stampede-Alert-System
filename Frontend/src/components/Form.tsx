import { ReactNode } from 'react';

interface FormInputProps {
  label: string;
  type?: string;
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  required?: boolean;
  error?: string;
}

export const FormInput = ({
  label,
  type = 'text',
  placeholder,
  value,
  onChange,
  required,
  error,
}: FormInputProps) => (
  <div className="mb-4">
    <label className="block text-sm font-medium text-gray-700 mb-1">
      {label}
      {required && <span className="text-red-500">*</span>}
    </label>
    <input
      type={type}
      placeholder={placeholder}
      value={value}
      onChange={(e) => onChange?.(e.target.value)}
      className={`input-base ${error ? 'border-red-500' : ''}`}
    />
    {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
  </div>
);

interface FormSelectProps {
  label: string;
  options: { value: string; label: string }[];
  value?: string;
  onChange?: (value: string) => void;
  required?: boolean;
  error?: string;
}

export const FormSelect = ({
  label,
  options,
  value,
  onChange,
  required,
  error,
}: FormSelectProps) => (
  <div className="mb-4">
    <label className="block text-sm font-medium text-gray-700 mb-1">
      {label}
      {required && <span className="text-red-500">*</span>}
    </label>
    <select
      value={value}
      onChange={(e) => onChange?.(e.target.value)}
      className={`input-base ${error ? 'border-red-500' : ''}`}
    >
      <option value="">Select {label}</option>
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
    {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
  </div>
);

interface FormCheckboxProps {
  label: ReactNode;
  checked?: boolean;
  onChange?: (checked: boolean) => void;
}

export const FormCheckbox = ({
  label,
  checked,
  onChange,
}: FormCheckboxProps) => (
  <div className="mb-4 flex items-center gap-2">
    <input
      type="checkbox"
      checked={checked}
      onChange={(e) => onChange?.(e.target.checked)}
      className="w-4 h-4"
    />
    <label className="text-sm text-gray-700">{label}</label>
  </div>
);
