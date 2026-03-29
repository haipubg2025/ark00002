
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BrainCircuit, Settings2, RotateCcw, Edit2, ToggleLeft, ToggleRight, Check, X } from 'lucide-react';
import { TawaPresetConfig } from '../../../../types';
import { DEFAULT_PRESET_CONFIG } from '../../../../constants/tawa_modules';

interface TawaPresetManagerProps {
  onConfigChange: (config: TawaPresetConfig) => void;
}

const STORAGE_KEY = 'tawa_preset_config_v1';

const TawaPresetManager: React.FC<TawaPresetManagerProps> = ({ onConfigChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [config, setConfig] = useState<TawaPresetConfig>(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        return { ...DEFAULT_PRESET_CONFIG, ...parsed };
      } catch (e) {
        console.error("Failed to parse saved Tawa Config", e);
        return DEFAULT_PRESET_CONFIG;
      }
    }
    return DEFAULT_PRESET_CONFIG;
  });
  const [editingId, setEditingId] = useState<string | null>(null);

  // Persist: Save to LocalStorage whenever config changes
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
    onConfigChange(config);
  }, [config, onConfigChange]);

  // --- Handlers ---

  const handleToggleModule = (moduleId: string) => {
    setConfig(prev => ({
      ...prev,
      modules: prev.modules.map(m => m.id === moduleId ? { ...m, isActive: !m.isActive } : m)
    }));
  };

  const handleUpdateContent = (id: string, newContent: string) => {
    // Check if it's COT
    if (id === config.cot.id) {
        setConfig(prev => ({ ...prev, cot: { ...prev.cot, content: newContent } }));
    } else {
        // Check modules
        setConfig(prev => ({
            ...prev,
            modules: prev.modules.map(m => m.id === id ? { ...m, content: newContent } : m)
        }));
    }
  };

  const handleReset = () => {
      try {
        // 1. Deep copy từ file constants để lấy lại dữ liệu gốc sạch sẽ nhất
        const freshConfig = JSON.parse(JSON.stringify(DEFAULT_PRESET_CONFIG));

        // 2. Cập nhật State React
        setConfig(freshConfig);

        // 3. Cập nhật LocalStorage ngay lập tức
        localStorage.setItem(STORAGE_KEY, JSON.stringify(freshConfig));
        
        setEditingId(null);
      } catch {
        // Silent fail
      }
  };

  // --- Render Helpers ---

  const renderEditor = (id: string, label: string, content: string) => (
      <motion.div 
        initial={{ opacity: 0, height: 0 }}
        animate={{ opacity: 1, height: 'auto' }}
        exit={{ opacity: 0, height: 0 }}
        className="mt-2 pl-4 border-l-2 border-stone-400 dark:border-slate-700"
      >
          <div className="flex justify-between items-center mb-1">
              <span className="text-[10px] text-stone-500 uppercase font-bold">Editing: {label}</span>
              <button onClick={() => setEditingId(null)} className="text-[10px] text-green-600 dark:text-green-400 flex items-center gap-1 hover:underline">
                  <Check size={12}/> Done
              </button>
          </div>
          <textarea 
            value={content}
            onChange={(e) => handleUpdateContent(id, e.target.value)}
            className="w-full h-64 bg-stone-200 dark:bg-slate-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-xs font-mono text-stone-800 dark:text-slate-300 focus:border-mystic-accent outline-none resize-y custom-scrollbar leading-relaxed"
            placeholder="Nhập nội dung prompt..."
          />
      </motion.div>
  );

  return (
    <>
        {/* Trigger Button - Renamed to "Preset" */}
        <button 
            onClick={() => setIsOpen(true)}
            className="w-full p-3 flex items-center justify-between text-left hover:bg-stone-400 dark:hover:bg-slate-700/50 transition-colors group rounded-lg border border-stone-400 dark:border-slate-700 bg-stone-300 dark:bg-slate-800/30"
        >
            <div className="flex items-center gap-2 text-sm font-bold text-stone-700 dark:text-slate-300 group-hover:text-mystic-accent transition-colors">
                <Settings2 size={16} />
                Preset
            </div>
            <div className="text-[10px] text-stone-500 bg-stone-400 dark:bg-slate-800 px-2 py-0.5 rounded border border-stone-400 dark:border-slate-700">
                {config.modules.filter(m => m.isActive).length} Active
            </div>
        </button>

        {/* Modal Popup */}
        <AnimatePresence>
            {isOpen && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
                    <motion.div 
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="bg-stone-200 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 w-full max-w-3xl rounded-xl shadow-2xl flex flex-col max-h-[85vh] overflow-hidden"
                    >
                        {/* Modal Header */}
                        <div className="p-4 border-b border-stone-400 dark:border-slate-800 flex justify-between items-center bg-stone-300 dark:bg-slate-900/50 shrink-0">
                            <h2 className="text-lg font-bold text-stone-800 dark:text-slate-200 flex items-center gap-2">
                                <Settings2 size={20} className="text-mystic-accent"/> Cấu hình Preset & Prompt
                            </h2>
                            <button onClick={() => setIsOpen(false)} className="text-stone-500 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white p-1 rounded hover:bg-stone-400 dark:hover:bg-slate-800 transition-colors">
                                <X size={24} />
                            </button>
                        </div>

                        {/* Modal Body - Scrollable */}
                        <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-6 bg-stone-200 dark:bg-mystic-900">
                            {/* 1. Core COT Section */}
                            <div className="bg-stone-300 dark:bg-slate-800/30 p-4 rounded-lg border border-stone-400 dark:border-slate-700/50">
                                <div className="flex justify-between items-center mb-2">
                                    <div className="flex items-center gap-2 text-sm font-bold text-mystic-accent">
                                        <BrainCircuit size={16} />
                                        {config.cot.label}
                                    </div>
                                    <button 
                                        onClick={() => setEditingId(editingId === config.cot.id ? null : config.cot.id)}
                                        className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors border ${editingId === config.cot.id ? 'bg-stone-400 dark:bg-slate-700 border-mystic-accent text-mystic-accent' : 'border-stone-400 dark:border-slate-600 text-stone-500 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white'}`}
                                    >
                                        <Edit2 size={12} /> {editingId === config.cot.id ? 'Đang sửa' : 'Chỉnh sửa'}
                                    </button>
                                </div>
                                <p className="text-xs text-stone-500 mb-2">
                                    Logic cốt lõi điều khiển luồng suy nghĩ của AI. Không thể tắt.
                                </p>
                                {editingId === config.cot.id && renderEditor(config.cot.id, config.cot.label, config.cot.content)}
                            </div>

                            {/* 2. Modules List */}
                            <div className="space-y-3">
                                <h4 className="text-xs font-bold text-stone-400 dark:text-slate-500 uppercase tracking-wider flex items-center gap-2">
                                    Modules Bổ Sung
                                    <div className="h-[1px] flex-1 bg-stone-400 dark:bg-slate-800"></div>
                                </h4>
                                
                                {config.modules
                                    .filter(mod => mod.id !== 'conf_word_count') // FILTER HIDDEN MODULE
                                    .map(mod => (
                                    <div key={mod.id} className={`p-3 rounded-lg border transition-all ${mod.isActive ? 'bg-stone-300 dark:bg-slate-800/60 border-stone-400 dark:border-slate-600' : 'bg-stone-200 dark:bg-slate-900/50 border-stone-400 dark:border-slate-800 opacity-70'}`}>
                                        <div className="flex justify-between items-start">
                                            <div className="flex-1 mr-4">
                                                <div className="flex items-center gap-2 mb-1">
                                                     <span className={`text-sm font-medium ${mod.isActive ? 'text-stone-800 dark:text-slate-200' : 'text-stone-400 dark:text-slate-500'}`}>
                                                        {mod.label}
                                                    </span>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <button 
                                                        onClick={() => setEditingId(editingId === mod.id ? null : mod.id)}
                                                        className={`text-xs flex items-center gap-1 hover:underline ${editingId === mod.id ? 'text-mystic-accent' : 'text-stone-500'}`}
                                                    >
                                                        <Edit2 size={10} /> {editingId === mod.id ? 'Đóng editor' : 'Xem/Sửa nội dung'}
                                                    </button>
                                                </div>
                                            </div>

                                            <button 
                                                onClick={() => handleToggleModule(mod.id)}
                                                className={`${mod.isActive ? 'text-green-600 dark:text-green-400' : 'text-stone-300 dark:text-slate-600'} hover:scale-110 transition-transform`}
                                                title={mod.isActive ? "Đang BẬT" : "Đang TẮT"}
                                            >
                                                {mod.isActive ? <ToggleRight size={28} /> : <ToggleLeft size={28} />}
                                            </button>
                                        </div>
                                        {editingId === mod.id && renderEditor(mod.id, mod.label, mod.content)}
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Modal Footer */}
                        <div className="p-4 border-t border-stone-400 dark:border-slate-800 bg-stone-300 dark:bg-slate-900/50 flex justify-between items-center shrink-0">
                            <button 
                                onClick={handleReset}
                                className="flex items-center gap-1.5 text-xs text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 hover:bg-red-100 dark:hover:bg-red-900/20 px-3 py-2 rounded transition-colors"
                            >
                                <RotateCcw size={14} /> Reset Mặc định
                            </button>
                            
                            <button 
                                onClick={() => setIsOpen(false)}
                                className="px-6 py-2 bg-stone-400 dark:bg-mystic-800 border border-stone-400 dark:border-mystic-accent/30 text-stone-700 dark:text-mystic-accent hover:bg-stone-500 dark:hover:bg-mystic-accent/10 rounded-lg text-sm font-bold transition-all"
                            >
                                Đóng
                            </button>
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>
    </>
  );
};

export default TawaPresetManager;
