
import React from 'react';
import { motion } from 'framer-motion';
import { Play, Clock, FileText, Settings } from 'lucide-react';
import Button from '../components/ui/Button';
import { GameState } from '../types';

/**
 * MobileLibrary: Thư viện quản lý các thành phần giao diện dành riêng cho di động.
 * Mọi thay đổi về cấu trúc di động nên được thực hiện tại đây để đảm bảo tính tập trung.
 */

interface MobileMenuProps {
  onNavigate: (state: GameState) => void;
  onContinue: () => void;
  onLoadGame: () => void;
  onShowCardSTAnalyzer: () => void;
  hasSaves: boolean;
}

export const MobileMainMenu: React.FC<MobileMenuProps> = ({ 
  onNavigate, 
  onContinue, 
  onLoadGame, 
  onShowCardSTAnalyzer,
  hasSaves 
}) => {
  return (
    <div className="flex flex-col items-center justify-center h-full px-6 py-12 space-y-8">
      {/* Logo di động - Nhỏ gọn hơn */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="font-serif text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-b from-slate-100 to-slate-500 tracking-tighter">
          Ark v2 SillyTavern
        </h1>
        <p className="text-[10px] text-mystic-accent tracking-[0.3em] uppercase mt-2 opacity-60">
          Mobile Edition
        </p>
      </motion.div>

      {/* Danh sách nút bấm - Tối ưu cho cảm ứng (Touch targets) */}
      <div className="grid grid-cols-1 gap-4 w-full max-w-[280px]">
        <Button 
          variant="primary" 
          className="h-16 text-lg rounded-2xl shadow-lg shadow-mystic-accent/20"
          onClick={() => onNavigate(GameState.WORLD_CREATION)}
        >
          <div className="flex items-center gap-3">
            <Play size={20} fill="currentColor" />
            KHỞI TẠO
          </div>
        </Button>

        <Button 
          variant="ghost" 
          className="h-14 text-base rounded-2xl border border-slate-700 bg-slate-900/40 text-slate-200 active:scale-95 transition-all"
          onClick={() => onNavigate(GameState.FANFIC)}
        >
          <div className="flex items-center gap-3">
            <FileText size={18} className="text-mystic-accent" />
            ĐỒNG NHÂN
          </div>
        </Button>

        <Button 
          variant="ghost" 
          className="h-14 text-base rounded-2xl border border-slate-700 bg-slate-900/40 text-slate-200 active:scale-95 transition-all"
          onClick={onShowCardSTAnalyzer}
        >
          <div className="flex items-center gap-3">
            <FileText size={18} className="text-mystic-accent" />
            TẢI CARD ST
          </div>
        </Button>

        <div className="grid grid-cols-2 gap-3">
          <button 
            onClick={onContinue}
            disabled={!hasSaves}
            className={`flex flex-col items-center justify-center gap-2 p-4 rounded-2xl border transition-all ${
              hasSaves 
              ? 'bg-slate-900/40 border-slate-700 text-slate-200 active:scale-95' 
              : 'bg-slate-900/10 border-slate-800 text-slate-600 opacity-50'
            }`}
          >
            <Clock size={24} />
            <span className="text-[10px] font-bold uppercase tracking-tighter">Tiếp tục</span>
          </button>

          <button 
            onClick={onLoadGame}
            className="flex flex-col items-center justify-center gap-2 p-4 bg-slate-900/40 border border-slate-700 text-slate-200 rounded-2xl active:scale-95 transition-all"
          >
            <FileText size={24} />
            <span className="text-[10px] font-bold uppercase tracking-tighter">Dữ liệu</span>
          </button>
        </div>

        <button 
          onClick={() => onNavigate(GameState.SETTINGS)}
          className="flex items-center justify-center gap-3 p-4 bg-slate-900/20 border border-slate-800/50 text-slate-400 rounded-2xl active:scale-95 transition-all"
        >
          <Settings size={18} />
          <span className="text-xs font-bold uppercase tracking-widest">Cấu hình hệ thống</span>
        </button>
      </div>

    </div>
  );
};

// Bạn có thể thêm các thành phần Mobile khác vào đây (MobileSettings, MobileGameplay, v.v.)
