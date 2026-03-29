
import React from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';

interface ButtonProps extends HTMLMotionProps<"button"> {
  variant?: 'primary' | 'ghost' | 'outline' | 'danger';
  icon?: React.ReactNode;
  isLoading?: boolean;
  children?: React.ReactNode;
  className?: string;
  disabled?: boolean;
  onClick?: React.MouseEventHandler<HTMLButtonElement>;
}

const Button: React.FC<ButtonProps> = ({ 
  children, 
  variant = 'primary', 
  className = '', 
  icon, 
  isLoading, 
  disabled,
  ...props 
}) => {
  
  const baseStyles = "relative px-6 py-3 rounded-md font-sans font-medium transition-all duration-200 flex items-center justify-center gap-3 overflow-hidden group";
  
  const variants = {
    primary: "bg-mystic-100 dark:bg-mystic-800 border border-mystic-accent/30 text-mystic-accent hover:bg-mystic-accent/10 hover:border-mystic-accent hover:shadow-[0_0_15px_rgba(56,189,248,0.3)]",
    ghost: "bg-transparent text-stone-600 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white hover:bg-stone-200 dark:hover:bg-white/5",
    outline: "border border-stone-300 dark:border-slate-600 text-stone-600 dark:text-slate-300 hover:border-stone-400 hover:text-stone-900 dark:hover:text-white",
    danger: "border border-red-200 dark:border-red-900/50 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 hover:border-red-500 hover:text-red-700 dark:hover:text-red-200"
  };

  const disabledStyles = "opacity-50 cursor-not-allowed grayscale";

  return (
    <motion.button
      whileHover={!disabled ? { scale: 1.02 } : {}}
      whileTap={!disabled ? { scale: 0.98 } : {}}
      className={`${baseStyles} ${variants[variant]} ${disabled || isLoading ? disabledStyles : ''} ${className}`}
      disabled={disabled || isLoading}
      onClick={props.onClick}
      {...props}
    >
      {/* Background sweep effect for primary buttons */}
      {variant === 'primary' && !disabled && (
        <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:animate-[shimmer_1.5s_infinite]" />
      )}

      {isLoading ? (
        <span className="animate-spin h-5 w-5 border-2 border-current border-t-transparent rounded-full" />
      ) : (
        <>
          {icon && <span className="w-5 h-5">{icon}</span>}
          <span className="tracking-wide uppercase text-sm">{children}</span>
        </>
      )}
    </motion.button>
  );
};

export default Button;