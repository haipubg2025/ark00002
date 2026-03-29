
import { useState, useEffect } from 'react';

/**
 * Hook dùng để kiểm tra xem thiết bị hiện tại có phải là di động hay không.
 * Ngưỡng mặc định là 768px (MD breakpoint của Tailwind).
 */
export const useResponsive = (breakpoint: number = 768) => {
  const [isMobile, setIsMobile] = useState<boolean>(
    typeof window !== 'undefined' ? window.innerWidth < breakpoint : false
  );

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < breakpoint);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [breakpoint]);

  return { isMobile, width: typeof window !== 'undefined' ? window.innerWidth : 0 };
};
