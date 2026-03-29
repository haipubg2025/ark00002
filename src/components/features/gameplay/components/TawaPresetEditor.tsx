
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BrainCircuit, RotateCcw, Edit2, ChevronDown, ChevronUp } from 'lucide-react';

interface TawaPresetEditorProps {
  currentProtocol: string;
  onChange: (newProtocol: string) => void;
}

const TawaPresetEditor: React.FC<TawaPresetEditorProps> = ({ currentProtocol, onChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activePreset, setActivePreset] = useState<'standard' | 'custom'>('standard');

  const handlePresetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    if (val === 'standard') {
        setActivePreset('standard');
        onChange(""); // Reset to empty, managed by modules now
    } else {
        setActivePreset('custom');
    }
  };

  const handleReset = () => {
    setActivePreset('standard');
    onChange("");
  };

  return (
    <div className="border border-stone-400 dark:border-slate-700 rounded-lg bg-stone-300 dark:bg-slate-800/30 overflow-hidden">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full p-3 flex justify-between items-center text-left hover:bg-stone-400 dark:hover:bg-slate-700/50 transition-colors"
      >
          <div className="flex items-center gap-2 text-sm font-bold text-mystic-accent uppercase">
             <BrainCircuit size={16} />
             Cấu hình Tư duy (Tawa Preset)
          </div>
          {isOpen ? <ChevronUp size={16} className="text-stone-500 dark:text-slate-400"/> : <ChevronDown size={16} className="text-stone-500 dark:text-slate-400"/>}
      </button>

      <AnimatePresence>
        {isOpen && (
            <motion.div 
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
            >
                <div className="p-3 pt-0 space-y-3">
                    <div className="flex gap-2">
                        <select 
                            value={activePreset}
                            onChange={handlePresetChange}
                            className="flex-1 bg-stone-200 dark:bg-slate-900 border border-stone-400 dark:border-slate-600 rounded p-2 text-xs text-stone-800 dark:text-slate-200 outline-none focus:border-mystic-accent"
                        >
                            <option value="standard">Tawa Standard (Mặc định)</option>
                            <option value="custom">Custom (Tùy chỉnh)</option>
                        </select>
                        
                        <button 
                            onClick={handleReset}
                            className="p-2 bg-stone-400 dark:bg-slate-700 hover:bg-stone-500 dark:hover:bg-slate-600 rounded text-stone-600 dark:text-slate-300 transition-colors"
                            title="Khôi phục mặc định"
                        >
                            <RotateCcw size={14} />
                        </button>
                    </div>

                    <div className="relative">
                         <div className="absolute top-2 right-2 text-[10px] text-stone-400 dark:text-slate-500 flex items-center gap-1 pointer-events-none">
                            <Edit2 size={10} /> Editor
                         </div>
                         <textarea 
                            value={currentProtocol}
                            onChange={(e) => {
                                setActivePreset('custom');
                                onChange(e.target.value);
                            }}
                            className="w-full h-48 bg-stone-200 dark:bg-slate-900/80 border border-stone-400 dark:border-slate-700 rounded p-2 text-[10px] font-mono text-stone-500 dark:text-slate-400 focus:text-stone-800 dark:focus:text-slate-200 focus:border-mystic-accent outline-none resize-y custom-scrollbar leading-relaxed"
                            placeholder="Nhập quy trình tư duy của AI..."
                         />
                    </div>
                    
                    <p className="text-[10px] text-stone-500 italic">
                        *Chỉnh sửa trực tiếp prompt logic để thay đổi cách AI suy nghĩ. Nội dung này sẽ được nạp vào System Instruction.
                    </p>
                </div>
            </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default TawaPresetEditor;
