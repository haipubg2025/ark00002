
import React from 'react';
import { useResponsive } from '../../hooks/useResponsive';

interface AdaptiveUIProps {
  /** Giao diện dành cho máy tính */
  desktop: React.ReactNode;
  /** Giao diện dành cho điện thoại */
  mobile: React.ReactNode;
  /** Cho phép ghi đè breakpoint nếu cần (mặc định 768px) */
  breakpoint?: number;
}

/**
 * AdaptiveUI là thành phần bắt buộc người dùng phải cung cấp cả hai giao diện.
 * Điều này đảm bảo khi cập nhật tính năng mới, lập trình viên không quên cập nhật bản di động.
 */
const AdaptiveUI: React.FC<AdaptiveUIProps> = ({ desktop, mobile, breakpoint }) => {
  const { isMobile } = useResponsive(breakpoint);

  return (
    <div className="w-full h-full overflow-hidden">
      {isMobile ? mobile : desktop}
    </div>
  );
};

export default AdaptiveUI;
