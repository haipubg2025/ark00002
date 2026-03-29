
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { X, Save, Sparkles, Eye, EyeOff } from 'lucide-react';
import { Entity, EntityType } from '../../../types';
import Button from '../../ui/Button';
import MarkdownRenderer from '../../common/MarkdownRenderer';
import { worldAiService } from '../../../services/ai/world-creation/service';

interface EntityFormProps {
  initialData?: Entity;
  onSave: (entity: Omit<Entity, 'id'>) => void;
  onCancel: () => void;
}

const EntityForm: React.FC<EntityFormProps> = ({ initialData, onSave, onCancel }) => {
  const [type, setType] = useState<EntityType>(initialData?.type || 'NPC');
  const [name, setName] = useState(initialData?.name || '');
  const [age, setAge] = useState(initialData?.age || '');
  const [gender, setGender] = useState(initialData?.gender || '');
  const [description, setDescription] = useState(initialData?.description || '');
  const [personality, setPersonality] = useState(initialData?.personality || '');
  const [rarity, setRarity] = useState(initialData?.rarity || '');
  const [price, setPrice] = useState(initialData?.price || '');
  const [customType, setCustomType] = useState(initialData?.customType || '');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPreview, setIsPreview] = useState(false);
  
  const handleSave = () => {
    if (!name.trim()) {
        return;
    }
    
    const entity: Omit<Entity, 'id'> = {
      type,
      name,
      description,
      ...(type === 'NPC' && { personality, age, gender }),
      ...(type === 'ITEM' && { rarity, price }),
      ...(type === 'CUSTOM' && { customType })
    };
    onSave(entity);
  };

  const handleAiSuggest = async (field: 'description' | 'personality') => {
    if (!name.trim()) {
        return;
    }

    setIsGenerating(true);
    try {
      const contextData = { name, type, genre: '' }; // Genre could be passed in props for better context if available
      
      // Determine current value for enrichment
      let currentValue = "";
      if (field === 'description') currentValue = description;
      if (field === 'personality') currentValue = personality;

      const content = await worldAiService.generateFieldContent('entity', field, contextData, 'gemini-3-pro-preview', currentValue);
      
      if (field === 'description') {
          setDescription(content);
      } else {
          setPersonality(content);
      }
    } catch (error) {
        console.error(error);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <motion.div 
        initial={{ scale: 0.9, opacity: 0 }} 
        animate={{ scale: 1, opacity: 1 }} 
        className="bg-stone-200 dark:bg-mystic-900 w-full max-w-lg rounded-lg border border-stone-400 dark:border-slate-700 shadow-2xl overflow-hidden"
      >
        <div className="flex justify-between items-center p-4 border-b border-stone-400 dark:border-slate-800 bg-stone-300 dark:bg-slate-900">
          <h3 className="text-lg font-bold text-stone-800 dark:text-slate-200">
            {initialData ? 'Chỉnh sửa thực thể' : 'Thêm thực thể mới'}
          </h3>
          <button onClick={onCancel} className="text-stone-500 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white">
            <X size={20} />
          </button>
        </div>

        <div className="p-6 space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-mystic-accent">Loại thực thể</label>
            <select 
              value={type} 
              onChange={(e) => setType(e.target.value as EntityType)}
              className="w-full bg-stone-300 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-stone-900 dark:text-slate-100 focus:border-mystic-accent outline-none"
            >
              <option value="NPC">NPC (Nhân vật)</option>
              <option value="LOCATION">Địa điểm</option>
              <option value="ITEM">Vật phẩm</option>
              <option value="CUSTOM">Tùy chỉnh</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-mystic-accent">Tên gọi</label>
            <input 
              type="text" 
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full bg-stone-300 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-stone-900 dark:text-slate-100 focus:border-mystic-accent outline-none"
              placeholder="Ví dụ: Lão Hạc, Thành phố Bay..."
            />
          </div>

          {type === 'NPC' && (
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-mystic-accent">Tuổi</label>
                <input 
                  type="text" 
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  className="w-full bg-stone-300 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-stone-900 dark:text-slate-100 focus:border-mystic-accent outline-none"
                  placeholder="Ví dụ: 25, 100..."
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-mystic-accent">Giới tính</label>
                <select 
                  value={gender}
                  onChange={(e) => setGender(e.target.value)}
                  className="w-full bg-stone-300 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-stone-900 dark:text-slate-100 focus:border-mystic-accent outline-none"
                >
                  <option value="">Chọn...</option>
                  <option value="Nam">Nam</option>
                  <option value="Nữ">Nữ</option>
                  <option value="Khác">Khác</option>
                </select>
              </div>
            </div>
          )}

          {type === 'ITEM' && (
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-mystic-accent">Độ hiếm</label>
                <select 
                  value={rarity}
                  onChange={(e) => setRarity(e.target.value)}
                  className="w-full bg-stone-300 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-stone-900 dark:text-slate-100 focus:border-mystic-accent outline-none"
                >
                  <option value="">Chọn...</option>
                  <option value="Thường">Thường</option>
                  <option value="Hiếm">Hiếm</option>
                  <option value="Cực hiếm">Cực hiếm</option>
                  <option value="Huyền thoại">Huyền thoại</option>
                  <option value="Thần thoại">Thần thoại</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-mystic-accent">Giá cả</label>
                <input 
                  type="text" 
                  value={price}
                  onChange={(e) => setPrice(e.target.value)}
                  className="w-full bg-stone-300 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-stone-900 dark:text-slate-100 focus:border-mystic-accent outline-none"
                  placeholder="Ví dụ: 100 Vàng, Vô giá..."
                />
              </div>
            </div>
          )}

          {type === 'CUSTOM' && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-mystic-accent">Phân loại</label>
              <input 
                type="text" 
                value={customType}
                onChange={(e) => setCustomType(e.target.value)}
                className="w-full bg-stone-300 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-stone-900 dark:text-slate-100 focus:border-mystic-accent outline-none"
                placeholder="Ví dụ: Vật phẩm, Thần thú..."
              />
            </div>
          )}

          <div className="space-y-2 relative">
            <label className="text-sm font-medium text-mystic-accent flex justify-between">
              <div className="flex items-center gap-2">
                <span>Mô tả</span>
                <button 
                  type="button"
                  onClick={() => setIsPreview(!isPreview)}
                  className="text-slate-400 hover:text-mystic-accent transition-colors"
                  title={isPreview ? "Chỉnh sửa" : "Xem trước Markdown"}
                >
                  {isPreview ? <EyeOff size={12} /> : <Eye size={12} />}
                </button>
              </div>
              <button 
                onClick={() => handleAiSuggest('description')} 
                disabled={isGenerating}
                className="text-xs flex items-center gap-1 text-mystic-accent/80 hover:text-mystic-accent"
                title={description ? "Cải thiện nội dung" : "Tạo mới ngẫu nhiên"}
              >
                {isGenerating ? <span className="animate-spin">⏳</span> : <Sparkles size={12} />} 
                {description ? "AI Cải thiện" : "AI Gợi ý"}
              </button>
            </label>
            {isPreview ? (
              <div className="w-full h-24 bg-stone-300 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-stone-900 dark:text-slate-100 overflow-y-auto custom-scrollbar text-xs">
                <MarkdownRenderer content={description || "*Chưa có mô tả*"} />
              </div>
            ) : (
              <textarea 
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="w-full h-24 bg-stone-300 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-stone-900 dark:text-slate-100 focus:border-mystic-accent outline-none resize-none"
                placeholder="Mô tả chi tiết về thực thể này..."
              />
            )}
          </div>

          {type === 'NPC' && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-mystic-accent flex justify-between">
                  <span>Tính cách</span>
                  <button 
                    onClick={() => handleAiSuggest('personality')} 
                    disabled={isGenerating}
                    className="text-xs flex items-center gap-1 text-mystic-accent/80 hover:text-mystic-accent"
                    title={personality ? "Cải thiện nội dung" : "Tạo mới ngẫu nhiên"}
                  >
                    {isGenerating ? <span className="animate-spin">⏳</span> : <Sparkles size={12} />} 
                    {personality ? "AI Cải thiện" : "AI Gợi ý"}
                  </button>
              </label>
              <input 
                type="text" 
                value={personality}
                onChange={(e) => setPersonality(e.target.value)}
                className="w-full bg-stone-300 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-stone-900 dark:text-slate-100 focus:border-mystic-accent outline-none"
                placeholder="Ví dụ: Nóng tính, Thích đùa..."
              />
            </div>
          )}
        </div>

        <div className="p-4 border-t border-stone-400 dark:border-slate-800 bg-stone-300 dark:bg-slate-900 flex justify-end gap-3">
          <Button variant="ghost" onClick={onCancel}>Hủy</Button>
          <Button variant="primary" onClick={handleSave} icon={<Save size={16} />}>Lưu</Button>
        </div>
      </motion.div>
    </div>
  );
};

export default EntityForm;
