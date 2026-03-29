
import React from 'react';
import { motion } from 'framer-motion';
import { AppSettings } from '../../../types';

const CATEGORY_NAMES: Record<string, string> = {
  'HARM_CATEGORY_HARASSMENT': 'Quấy rối (Harassment)',
  'HARM_CATEGORY_HATE_SPEECH': 'Ngôn từ thù ghét (Hate Speech)',
  'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'Nội dung khiêu dâm (Sexually Explicit)',
  'HARM_CATEGORY_DANGEROUS_CONTENT': 'Nội dung nguy hiểm (Dangerous)',
  'HARM_CATEGORY_CIVIC_INTEGRITY': 'Liêm chính công dân (Civic Integrity)',
};

interface SafetySettingsProps {
  settings: AppSettings;
  onUpdate: (newSettings: AppSettings) => void;
}

const SafetySettings: React.FC<SafetySettingsProps> = ({ settings, onUpdate }) => {

  const handleToggle = (category: string) => {
    if (!settings.safetySettings) return;

    const newSafetySettings = settings.safetySettings.map(s => {
      if (s.category === category) {
        // Toggle between BLOCK_NONE (OFF) and BLOCK_MEDIUM_AND_ABOVE (ON/Default)
        return { 
          ...s, 
          threshold: s.threshold === 'BLOCK_NONE' ? 'BLOCK_MEDIUM_AND_ABOVE' : 'BLOCK_NONE' 
        };
      }
      return s;
    });

    onUpdate({
      ...settings,
      safetySettings: newSafetySettings
    });
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="bg-stone-300 dark:bg-mystic-800/50 p-4 rounded-lg border border-stone-400 dark:border-slate-700">
        <div className="flex items-start gap-3 mb-4">
           <div>
             <h3 className="font-bold text-stone-800 dark:text-slate-200">Bộ lọc an toàn AI</h3>
             <p className="text-xs text-stone-500 dark:text-slate-400 mt-1">
               Cấu hình mức độ kiểm duyệt nội dung của AI.
             </p>
           </div>
        </div>

        <div className="space-y-3">
          {settings.safetySettings?.map((setting) => {
            const isOff = setting.threshold === 'BLOCK_NONE';
            return (
              <motion.div 
                key={setting.category} 
                className={`flex justify-between items-center p-3 rounded border transition-colors ${isOff ? 'bg-red-900/10 border-red-900/30' : 'bg-stone-200 dark:bg-slate-800 border-stone-400 dark:border-slate-700'}`}
              >
                <div className="flex flex-col">
                  <span className="text-sm font-medium text-stone-700 dark:text-slate-300">
                    {CATEGORY_NAMES[setting.category] || setting.category}
                  </span>
                  <span className={`text-[10px] uppercase font-bold ${isOff ? 'text-red-500' : 'text-green-500'}`}>
                    {isOff ? 'Đã tắt bộ lọc (Không an toàn)' : 'Đang bật (Mặc định)'}
                  </span>
                </div>
                
                <button
                  onClick={() => handleToggle(setting.category)}
                  className={`w-12 h-6 rounded-full p-1 transition-colors flex items-center ${isOff ? 'bg-stone-400 dark:bg-slate-700 justify-start' : 'bg-mystic-accent justify-end'}`}
                >
                   <motion.div layout className="w-4 h-4 bg-white rounded-full shadow-md" />
                </button>
              </motion.div>
            );
          })}
        </div>
      </div>
      
      {/* Removed redundant button */}
    </div>
  );
};

export default SafetySettings;