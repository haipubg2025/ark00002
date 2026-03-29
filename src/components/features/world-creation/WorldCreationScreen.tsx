
import React, { useReducer, useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ArrowLeft, Sparkles, Plus, Trash2, Edit2, Wand2, Play, 
  User, Globe, Settings, Users, Upload, Download, Clock,
  Eye, EyeOff
} from 'lucide-react';
import { NavigationProps, GameState, WorldData, AppSettings } from '../../../types';
import Button from '../../ui/Button';
import MarkdownRenderer from '../../common/MarkdownRenderer';
import { initialWorldState, worldCreationReducer } from './reducer';
import EntityForm from './EntityForm';
import { worldAiService } from '../../../services/ai/world-creation/service';
import { dbService } from '../../../services/db/indexedDB';
import { OUTPUT_LENGTHS, DIFFICULTY_LEVELS } from '../../../constants/promptTemplates';

const TABS = [
  { id: 0, label: "Nhân vật", icon: User },
  { id: 1, label: "Thế giới", icon: Globe },
  { id: 2, label: "Quy tắc", icon: Settings },
  { id: 3, label: "Thực thể", icon: Users },
  { id: 4, label: "Cấu hình", icon: Sparkles },
];

interface WorldCreationProps extends NavigationProps {
  initialData?: WorldData | null;
}

const WorldCreationScreen: React.FC<WorldCreationProps> = ({ onNavigate, onGameStart, initialData }) => {
  const [state, dispatch] = useReducer(worldCreationReducer, initialWorldState);
  const [showEntityForm, setShowEntityForm] = useState(false);
  const [editingEntityId, setEditingEntityId] = useState<string | null>(null);
  const [conceptInput, setConceptInput] = useState('');
  const [aiModel, setAiModel] = useState<string>('gemini-3-pro-preview');
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Initial Load (Settings & Import Data)
  useEffect(() => {
    dbService.getSettings().then(s => {
      setSettings(s);
      if (s.aiModel) setAiModel(s.aiModel);
    });

    // Check if there is initial data passed from Main Menu Import
    if (initialData) {
       dispatch({ type: 'IMPORT_DATA', payload: initialData });
    }
  }, [initialData]);

  // --- AI Helper Function (UPDATED WITH VALIDATION & ENRICHMENT) ---
  const handleAiGenerate = async (field: string, category: 'player' | 'world') => {
    // 1. Validation Logic
    if (category === 'player') {
        const { name, gender, age } = state.player;
        if (!name || !gender || !age) {
            return;
        }
    } else if (category === 'world') {
        if (!state.world.genre && field !== 'genre') {
            return;
        }
    }

    dispatch({ type: 'SET_GENERATING', isGenerating: true, field });
    try {
      // 2. Build Explicit Context
      const contextData = category === 'player' 
        ? { ...state.player, genre: state.world.genre } 
        : { genre: state.world.genre, worldName: state.world.worldName, concept: conceptInput };

      // 3. Get Current Value for Enrichment
      let currentValue = "";
      if (category === 'player') {
          // @ts-expect-error - Dynamic access
          currentValue = state.player[field] || "";
      } else {
          // @ts-expect-error - Dynamic access
          currentValue = state.world[field] || "";
      }

      const content = await worldAiService.generateFieldContent(category, field, contextData, aiModel, currentValue, settings || undefined);
      
      // Dispatch based on field type
      if (['name', 'gender', 'age', 'personality', 'background', 'appearance', 'skills', 'goal'].includes(field)) {
        dispatch({ type: 'UPDATE_PLAYER', field: field as keyof PlayerProfile, value: content });
      } else if (['worldName', 'context', 'genre'].includes(field)) {
        dispatch({ type: 'UPDATE_WORLD', field: field as keyof WorldSettingConfig, value: content });
      }
    } catch (error: unknown) {
      console.error("AI Error", error);
    } finally {
      dispatch({ type: 'SET_GENERATING', isGenerating: false });
    }
  };

  const handleAiSuggestTime = async () => {
    if (!state.world.genre) {
        return;
    }

    dispatch({ type: 'SET_GENERATING', isGenerating: true, field: 'gameTime' });
    try {
      const timeData = await worldAiService.generateInitialTime(state.world.genre, state.world.context, aiModel, settings || undefined);
      dispatch({ type: 'AUTO_FILL_ALL', payload: { gameTime: timeData } });
    } catch (error: unknown) {
      console.error("AI Time Error", error);
    } finally {
      dispatch({ type: 'SET_GENERATING', isGenerating: false });
    }
  };

  const handleAutoFillAll = async () => {
    if (!conceptInput.trim()) return;
    dispatch({ type: 'SET_GENERATING', isGenerating: true });
    try {
      const data = await worldAiService.generateFullWorld(conceptInput, aiModel, settings || undefined);
      dispatch({ type: 'AUTO_FILL_ALL', payload: data });
    } finally {
      dispatch({ type: 'SET_GENERATING', isGenerating: false });
    }
  };

  // --- Import / Export Logic ---
  const handleExportWorld = () => {
    if (!settings) return;

    const exportData: WorldData = {
        player: state.player,
        world: state.world,
        config: {
            ...state.config,
            difficulty: settings.difficulty,
            outputLength: settings.outputLength,
            perspective: settings.perspective,
            customMinWords: settings.customMinWords,
            customMaxWords: settings.customMaxWords
        },
        entities: state.entities,
        gameTime: state.gameTime
    };
    
    const worldName = state.world.worldName.replace(/\s+/g, '_') || 'unknown_world';
    const playerName = state.player.name.replace(/\s+/g, '_') || 'unknown_player';
    const timestamp = Date.now();
    const fileName = `ARK_${worldName}_${playerName}_${timestamp}.json`;
    
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportData, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", fileName);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const parsedData = JSON.parse(content) as WorldData;
        
        if (!parsedData.player || !parsedData.world || !parsedData.config) {
            throw new Error("Cấu trúc file không hợp lệ");
        }

        dispatch({ type: 'IMPORT_DATA', payload: parsedData });
      } catch (error: unknown) {
        console.error(error);
      }
    };
    reader.readAsText(file);
    event.target.value = ''; // Reset
  };

   // --- Start Game Logic ---
  const handleStartGame = async () => {
     if (!settings) return;

     const worldData: WorldData = {
        player: state.player,
        world: state.world,
        config: {
            ...state.config,
            difficulty: settings.difficulty,
            outputLength: settings.outputLength,
            perspective: settings.perspective,
            customMinWords: settings.customMinWords,
            customMaxWords: settings.customMaxWords
        },
        entities: state.entities,
        gameTime: state.gameTime,
        savedState: { history: [], turnCount: 0 }
     };
     
     if (!worldData.player.name || !worldData.world.worldName) {
         return;
     }

     try {
         await dbService.saveAutosave({
             id: `autosave-${Date.now()}`,
             name: `${worldData.world.worldName} - Khởi tạo`,
             createdAt: Date.now(),
             updatedAt: Date.now(),
             data: worldData
         });
     } catch (err: unknown) {
         console.error("Autosave failed", err);
     }

     if (onGameStart) {
         onGameStart(worldData);
     }
  };

  // --- RENDER FUNCTIONS FOR TABS ---

  const renderPlayerTab = () => (
    <div className="space-y-3 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <h3 className="text-base font-bold text-slate-800 dark:text-slate-200 border-b border-slate-200 dark:border-slate-700 pb-0.5 mb-1 flex items-center gap-2" style={{ fontFamily: 'Arial', lineHeight: '18px' }}>
        <User size={18} className="text-mystic-accent" /> Thiết lập Nhân Vật Chính
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="space-y-2">
          <InputGroup 
            label="Tên nhân vật" 
            value={state.player.name} 
            onChange={(v) => dispatch({ type: 'UPDATE_PLAYER', field: 'name', value: v })}
            placeholder="Ví dụ: Lương Thế Vinh, Trần Bình Trọng..." 
          />
          <div className="flex gap-2">
              <div className="flex-1">
                  <label className="block text-sm font-medium text-mystic-accent mb-1">Giới tính</label>
                  <select 
                      value={state.player.gender} 
                      onChange={(e) => dispatch({ type: 'UPDATE_PLAYER', field: 'gender', value: e.target.value })}
                      className="w-full bg-stone-100 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-stone-900 dark:text-slate-100 outline-none focus:border-mystic-accent text-sm"
                  >
                      <option value="Nam">Nam</option>
                      <option value="Nữ">Nữ</option>
                      <option value="Khác">Khác</option>
                  </select>
              </div>
              <div className="w-24">
                  <InputGroup 
                    label="Tuổi" 
                    value={state.player.age} 
                    onChange={(v) => dispatch({ type: 'UPDATE_PLAYER', field: 'age', value: v })}
                    placeholder="VD: 20"
                  />
              </div>
          </div>
          <TextAreaGroup 
              label="Tính cách" 
              value={state.player.personality} 
              onChange={(v) => dispatch({ type: 'UPDATE_PLAYER', field: 'personality', value: v })} 
              onAi={() => handleAiGenerate('personality', 'player')}
              loading={state.isGenerating && state.generatingField === 'personality'}
              height="h-24"
              placeholder="Ví dụ: Dũng cảm, cương trực nhưng đôi khi nóng nảy, rất trọng tình nghĩa..."
          />
          <TextAreaGroup 
              label="Ngoại hình" 
              value={state.player.appearance} 
              onChange={(v) => dispatch({ type: 'UPDATE_PLAYER', field: 'appearance', value: v })}
              onAi={() => handleAiGenerate('appearance', 'player')}
              loading={state.isGenerating && state.generatingField === 'appearance'}
              height="h-24"
              placeholder="Ví dụ: Cao 1m80, tóc đen dài buộc sau gáy, mắt sáng như sao, mặc áo vải thô..."
          />
        </div>
        <div className="space-y-2">
          <TextAreaGroup 
              label="Tiểu sử" 
              value={state.player.background} 
              onChange={(v) => dispatch({ type: 'UPDATE_PLAYER', field: 'background', value: v })} 
              height="h-28"
              onAi={() => handleAiGenerate('background', 'player')}
              loading={state.isGenerating && state.generatingField === 'background'}
              placeholder="Ví dụ: Sinh ra trong một gia đình thư hương thế gia đã sa sút, từ nhỏ đã nuôi chí lớn khôi phục gia tộc..."
          />
          <TextAreaGroup 
              label="Kỹ năng" 
              value={state.player.skills} 
              onChange={(v) => dispatch({ type: 'UPDATE_PLAYER', field: 'skills', value: v })} 
              onAi={() => handleAiGenerate('skills', 'player')}
              loading={state.isGenerating && state.generatingField === 'skills'}
              height="h-24"
              placeholder="Ví dụ: Thông thạo ngũ kinh, kiếm thuật cơ bản, khả năng nhìn thấy linh khí..."
          />
          <TextAreaGroup 
              label="Mục tiêu" 
              value={state.player.goal} 
              onChange={(v) => dispatch({ type: 'UPDATE_PLAYER', field: 'goal', value: v })} 
              onAi={() => handleAiGenerate('goal', 'player')}
              loading={state.isGenerating && state.generatingField === 'goal'}
              height="h-24"
              placeholder="Ví dụ: Đỗ trạng nguyên, tìm ra bí mật về cái chết của cha..."
          />
        </div>
      </div>
    </div>
  );

  const renderWorldTab = () => {
    return (
      <div className="space-y-3 animate-in fade-in slide-in-from-bottom-4 duration-500">
        <h3 className="text-base font-bold text-slate-800 dark:text-slate-200 border-b border-slate-200 dark:border-slate-700 pb-0.5 mb-1 flex items-center gap-2" style={{ fontFamily: 'Arial', lineHeight: '18px' }}>
          <Globe size={18} className="text-mystic-accent" /> Bối Cảnh Thế Giới
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <InputGroup 
              label="Tên thế giới" 
              value={state.world.worldName} 
              onChange={(v) => dispatch({ type: 'UPDATE_WORLD', field: 'worldName', value: v })} 
              onAi={() => handleAiGenerate('worldName', 'world')}
              loading={state.isGenerating && state.generatingField === 'worldName'}
              placeholder="Ví dụ: Đại Lục Huyền Bí, Thành Phố Ngầm..."
            />
            <InputGroup 
              label="Thể loại" 
              value={state.world.genre} 
              onChange={(v) => dispatch({ type: 'UPDATE_WORLD', field: 'genre', value: v })} 
              onAi={() => handleAiGenerate('genre', 'world')}
              loading={state.isGenerating && state.generatingField === 'genre'}
              placeholder="Ví dụ: Tiên Hiệp, Cyberpunk, Hậu Tận Thế..."
            />
        </div>
        <TextAreaGroup 
          label="Bối cảnh & Lịch sử" 
          value={state.world.context} 
          onChange={(v) => dispatch({ type: 'UPDATE_WORLD', field: 'context', value: v })} 
          height="h-56"
          placeholder="Mô tả xã hội (phong kiến, hiện đại...), công nghệ (hơi nước, AI...), hệ thống phép thuật (tu tiên, ma pháp...), lịch sử hình thành..."
          onAi={() => handleAiGenerate('context', 'world')}
          loading={state.isGenerating && state.generatingField === 'context'}
        />

        <div className="bg-stone-100 dark:bg-slate-800/50 p-4 rounded-lg border border-slate-200 dark:border-slate-700 space-y-3">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm font-medium text-mystic-accent">
                    <Clock size={16} />
                    <span>Thời gian khởi đầu</span>
                </div>
                <button 
                    onClick={handleAiSuggestTime}
                    disabled={state.isGenerating}
                    className="flex items-center gap-1.5 px-2 py-1 rounded border border-slate-300 dark:border-slate-700 bg-stone-50 dark:bg-mystic-900 hover:bg-mystic-accent/10 hover:border-mystic-accent hover:text-mystic-accent text-xs text-slate-500 dark:text-slate-400 transition-all"
                    title="AI Gợi ý thời gian hợp lý"
                >
                    {state.isGenerating && state.generatingField === 'gameTime' ? (
                        <span className="animate-spin block w-3 h-3 border-2 border-mystic-accent border-t-transparent rounded-full" />
                    ) : (
                        <Sparkles size={12} />
                    )}
                    <span>AI Gợi ý</span>
                </button>
            </div>

            <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
                <TimeInput 
                    label="Năm" 
                    value={state.gameTime.year} 
                    onChange={(v) => dispatch({ type: 'UPDATE_GAME_TIME', field: 'year', value: v })} 
                />
                <TimeInput 
                    label="Tháng" 
                    value={state.gameTime.month} 
                    min={1} max={12}
                    onChange={(v) => dispatch({ type: 'UPDATE_GAME_TIME', field: 'month', value: v })} 
                />
                <TimeInput 
                    label="Ngày" 
                    value={state.gameTime.day} 
                    min={1} max={31}
                    onChange={(v) => dispatch({ type: 'UPDATE_GAME_TIME', field: 'day', value: v })} 
                />
                <TimeInput 
                    label="Giờ" 
                    value={state.gameTime.hour} 
                    min={0} max={23}
                    onChange={(v) => dispatch({ type: 'UPDATE_GAME_TIME', field: 'hour', value: v })} 
                />
                <TimeInput 
                    label="Phút" 
                    value={state.gameTime.minute} 
                    min={0} max={59}
                    onChange={(v) => dispatch({ type: 'UPDATE_GAME_TIME', field: 'minute', value: v })} 
                />
            </div>
            <p className="text-[10px] text-slate-500 dark:text-slate-400 italic">
                * Bạn có thể tự nhập mốc thời gian tùy ý (VD: Năm 2026 hoặc Năm 1).
            </p>
        </div>

        <TextAreaGroup 
          label="Kịch bản khởi đầu (Tùy chọn)" 
          value={state.world.startingScenario || ''} 
          onChange={(v) => dispatch({ type: 'UPDATE_WORLD', field: 'startingScenario', value: v })} 
          height="h-28"
          placeholder="Nhập hành động hoặc tình huống bắt đầu cụ thể. VD: Tôi tỉnh dậy trong một nhà tù cháy rực, tay bị xích..."
        />
      </div>
    );
  };

  const renderConfigTab = () => (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 h-full flex flex-col">
       <h3 className="text-base font-bold text-slate-800 dark:text-slate-200 border-b border-slate-200 dark:border-slate-700 pb-0.5 mb-2 flex items-center gap-2" style={{ fontFamily: 'Arial', lineHeight: '18px' }}>
          <Settings size={18} className="text-mystic-accent" /> Quy Tắc Thế Giới
       </h3>
       
       <div className="bg-stone-100 dark:bg-slate-800/30 rounded-lg p-6 border border-stone-400 dark:border-slate-700 flex flex-col flex-1 min-h-[400px]">
            <div className="flex justify-between items-center mb-6">
                <div>
                    <label className="text-lg font-medium text-mystic-accent">Quy tắc bổ sung (Rules)</label>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                        Thiết lập các quy tắc đặc biệt cho thế giới của bạn (VD: Không được dùng từ ngữ hiện đại, Nhân vật không được giết người...).
                    </p>
                </div>
                <button 
                    onClick={() => dispatch({ type: 'ADD_RULE', rule: '' })}
                    className="text-sm flex items-center gap-1.5 px-3 py-1.5 rounded bg-mystic-accent/10 text-mystic-accent hover:bg-mystic-accent/20 transition-colors"
                >
                    <Plus size={16} /> Thêm quy tắc
                </button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 pr-1 custom-scrollbar">
                {state.config.rules.length === 0 && (
                    <div className="flex flex-col items-center justify-center py-12 text-slate-400 dark:text-slate-500 border-2 border-dashed border-slate-200 dark:border-slate-700/50 rounded-lg">
                        <Plus size={32} className="mb-2 opacity-20" />
                        <p className="text-sm italic">Chưa có quy tắc nào được thiết lập.</p>
                    </div>
                )}
                {state.config.rules.map((rule, idx) => (
                    <div key={idx} className="flex gap-2 group animate-in fade-in slide-in-from-left-2 duration-200">
                        <div className="flex items-center justify-center w-8 h-10 text-xs font-bold text-slate-400 dark:text-slate-600">
                            {idx + 1}.
                        </div>
                        <input 
                            type="text" 
                            value={rule}
                            onChange={(e) => {
                                const newRules = [...state.config.rules];
                                newRules[idx] = e.target.value;
                                dispatch({ type: 'UPDATE_CONFIG', field: 'rules', value: newRules });
                            }}
                            className="flex-1 bg-stone-100 dark:bg-slate-900 border border-stone-400 dark:border-slate-700 rounded px-4 py-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none transition-all"
                            placeholder="Nhập quy tắc (VD: Hệ thống tiền tệ là Vàng)..."
                        />
                        <button 
                            onClick={() => dispatch({ type: 'REMOVE_RULE', index: idx })}
                            className="text-slate-400 hover:text-red-500 hover:bg-red-900/10 p-2 rounded transition-all"
                        >
                            <Trash2 size={18} />
                        </button>
                    </div>
                ))}
            </div>
       </div>
    </div>
  );

  const renderEntitiesTab = () => (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 h-full flex flex-col">
       <h3 className="text-base font-bold text-slate-800 dark:text-slate-200 border-b border-slate-200 dark:border-slate-700 pb-0.5 mb-1 flex items-center gap-2" style={{ fontFamily: 'Arial', lineHeight: '18px' }}>
          <Users size={18} className="text-mystic-accent" /> Danh Sách Thực Thể & NPC
       </h3>
       <div className="flex justify-between items-center mb-4">
          <p className="text-sm text-slate-500 dark:text-slate-400">
             Thêm ít nhất 4 thực thể để thế giới sống động hơn. ({state.entities.length}/5 khuyến nghị)
          </p>
          <Button variant="primary" onClick={() => { setEditingEntityId(null); setShowEntityForm(true); }} icon={<Plus size={16} />}>
             Thêm thực thể
          </Button>
       </div>

       <div className="flex-1 overflow-y-auto grid grid-cols-1 md:grid-cols-2 gap-4 pb-4 custom-scrollbar pr-2">
            {state.entities.map(ent => (
                <div key={ent.id} className="bg-stone-100 dark:bg-slate-800 border border-stone-400 dark:border-slate-700 p-4 rounded-lg hover:border-mystic-accent/50 transition-colors group relative">
                    <div className="flex justify-between items-start mb-2">
                        <div className="flex items-center gap-2">
                            <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${
                                ent.type === 'NPC' ? 'bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200' : 
                                ent.type === 'LOCATION' ? 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200' : 
                                ent.type === 'ITEM' ? 'bg-amber-100 dark:bg-amber-900 text-amber-800 dark:text-amber-200' :
                                'bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200'
                            }`}>
                                {ent.type}
                            </span>
                            <h4 className="font-bold text-slate-800 dark:text-slate-200">{ent.name}</h4>
                        </div>
                        <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            <button onClick={() => { setEditingEntityId(ent.id); setShowEntityForm(true); }} className="p-1 hover:text-mystic-accent"><Edit2 size={14}/></button>
                            <button onClick={() => dispatch({type: 'REMOVE_ENTITY', id: ent.id})} className="p-1 hover:text-red-400"><Trash2 size={14}/></button>
                        </div>
                    </div>
                    <p className="text-sm text-slate-500 dark:text-slate-400 line-clamp-2">{ent.description}</p>
                    {ent.type === 'NPC' && <p className="text-xs text-slate-400 dark:text-slate-500 mt-2 italic">Tính cách: {ent.personality}</p>}
                </div>
            ))}
            {state.entities.length === 0 && (
                <div className="col-span-full border-2 border-dashed border-slate-200 dark:border-slate-700 rounded-lg flex items-center justify-center h-32 text-slate-400 dark:text-slate-500">
                    Chưa có thực thể nào (VD: NPC Sư Phụ, Địa danh Cấm Địa...).
                </div>
            )}
       </div>
    </div>
  );

  const renderSettingsTab = () => {
    if (!settings) return null;

    return (
      <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
        <h3 className="text-base font-bold text-slate-800 dark:text-slate-200 border-b border-slate-200 dark:border-slate-700 pb-0.5 mb-1 flex items-center gap-2" style={{ fontFamily: 'Arial', lineHeight: '18px' }}>
          <Sparkles size={18} className="text-mystic-accent" /> Cấu hình AI & Game
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4 bg-stone-100 dark:bg-slate-800/40 p-4 rounded-lg border border-slate-200 dark:border-slate-700">
            <div>
              <label className="block text-sm font-bold text-mystic-accent mb-2">Độ dài phản hồi AI (Mệnh lệnh tối cao)</label>
              <div className="grid grid-cols-1 gap-2">
                {OUTPUT_LENGTHS.map((len) => (
                  <button
                    key={len.id}
                    onClick={() => {
                      const newSettings = { ...settings, outputLength: len };
                      setSettings(newSettings);
                      dbService.saveSettings(newSettings);
                    }}
                    className={`flex items-center justify-between px-4 py-3 rounded-lg border transition-all text-left ${
                      settings.outputLength.id === len.id
                        ? 'bg-mystic-accent/20 border-mystic-accent text-mystic-accent shadow-[0_0_10px_rgba(56,189,248,0.2)]'
                        : 'bg-stone-50 dark:bg-slate-900 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:border-slate-400 dark:hover:border-slate-500'
                    }`}
                  >
                    <div className="flex flex-col">
                      <span className="text-sm font-bold">{len.label}</span>
                      {len.id === 'supreme' && (
                        <span className="text-[10px] uppercase tracking-wider font-black text-red-500 animate-pulse">Mệnh lệnh tối cao: 5000+ từ</span>
                      )}
                    </div>
                    {settings.outputLength.id === len.id && <Sparkles size={16} />}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-4 bg-stone-100 dark:bg-slate-800/40 p-4 rounded-lg border border-slate-200 dark:border-slate-700">
            <div>
              <label className="block text-sm font-bold text-mystic-accent mb-2">Độ khó thế giới</label>
              <div className="grid grid-cols-1 gap-2">
                {DIFFICULTY_LEVELS.map((diff) => (
                  <button
                    key={diff.id}
                    onClick={() => {
                      const newSettings = { ...settings, difficulty: diff };
                      setSettings(newSettings);
                      dbService.saveSettings(newSettings);
                    }}
                    className={`flex items-center justify-between px-4 py-3 rounded-lg border transition-all text-left ${
                      settings.difficulty.id === diff.id
                        ? 'bg-mystic-accent/20 border-mystic-accent text-mystic-accent shadow-[0_0_10px_rgba(56,189,248,0.2)]'
                        : 'bg-stone-50 dark:bg-slate-900 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:border-slate-400 dark:hover:border-slate-500'
                    }`}
                  >
                    <span className="text-sm font-bold">{diff.label}</span>
                    {settings.difficulty.id === diff.id && <Sparkles size={16} />}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-lg">
          <p className="text-xs text-amber-600 dark:text-amber-400 leading-relaxed">
            <strong>Lưu ý:</strong> Chế độ <strong>Tối thượng</strong> yêu cầu AI viết cực kỳ chi tiết (trên 5000 từ). 
            Thời gian phản hồi có thể lâu hơn bình thường (1-2 phút). Vui lòng kiên nhẫn khi AI đang "sáng tác".
          </p>
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full w-full relative overflow-hidden">
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        accept=".json" 
        className="hidden" 
      />

      <div className="shrink-0 p-2 md:p-6 pb-1">
          <div className="flex items-center justify-between mb-2">
            <button onClick={() => onNavigate(GameState.MENU)} className="text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white flex items-center gap-2">
                <ArrowLeft size={18} /> <span className="hidden sm:inline">Quay lại</span>
            </button>
            <h2 className="text-lg font-bold text-mystic-accent text-center" style={{ fontFamily: 'Arial', lineHeight: '18px' }}>Kiến Tạo Thế Giới</h2>
            <div className="w-10 sm:w-20" />
          </div>

          <div className="mb-2 bg-slate-100 dark:bg-mystic-800/40 p-2 rounded-lg border border-slate-200 dark:border-mystic-accent/20 flex gap-2 items-center">
             <Wand2 className="text-mystic-accent shrink-0" size={16} />
             <input 
                value={conceptInput}
                onChange={(e) => setConceptInput(e.target.value)}
                placeholder="Nhập ý tưởng sơ khởi..."
                className="flex-1 bg-transparent border-none text-xs text-slate-800 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none"
             />
          </div>

          <div className="flex border-b border-slate-200 dark:border-slate-700 overflow-x-auto no-scrollbar md:justify-start">
             {TABS.map((tab) => (
                 <button
                    key={tab.id}
                    onClick={() => dispatch({ type: 'SET_TAB', payload: tab.id })}
                    className={`flex-1 md:flex-none px-3 py-2 text-xs font-bold transition-colors whitespace-nowrap relative flex justify-center items-center gap-1.5 ${
                        state.currentTab === tab.id ? 'text-mystic-accent' : 'text-slate-400 dark:text-slate-500 hover:text-slate-900 dark:hover:text-slate-300'
                    }`}
                    title={tab.label}
                 >
                    <tab.icon size={16} />
                    <span className="hidden xs:inline">{tab.label}</span>
                    {state.currentTab === tab.id && (
                        <motion.div layoutId="activeTab" className="absolute bottom-0 left-0 right-0 h-0.5 bg-mystic-accent" />
                    )}
                 </button>
             ))}
          </div>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar relative px-2 md:px-6 pb-2">
         <AnimatePresence mode="wait">
            {state.currentTab === 0 && <motion.div key="tab0" className="h-full">{renderPlayerTab()}</motion.div>}
            {state.currentTab === 1 && <motion.div key="tab1" className="h-full">{renderWorldTab()}</motion.div>}
            {state.currentTab === 2 && <motion.div key="tab2" className="h-full">{renderConfigTab()}</motion.div>}
            {state.currentTab === 3 && <motion.div key="tab3" className="h-full">{renderEntitiesTab()}</motion.div>}
            {state.currentTab === 4 && <motion.div key="tab4" className="h-full">{renderSettingsTab()}</motion.div>}
         </AnimatePresence>
      </div>

      <div className="shrink-0 z-20 bg-stone-200/95 dark:bg-mystic-950/95 backdrop-blur-md border-t border-stone-400 dark:border-slate-800 p-0.5 shadow-lg dark:shadow-[0_-5px_15px_rgba(0,0,0,0.3)]">
         <div className="flex flex-row gap-1 md:gap-4 justify-between items-center w-full overflow-x-auto no-scrollbar">
             <div className="flex-1 min-w-[120px]">
                 <Button 
                    variant="ghost" 
                    icon={<Wand2 size={16}/>} 
                    className="w-full py-1 text-[10px] md:text-sm bg-mystic-accent/5"
                    onClick={handleAutoFillAll}
                    isLoading={state.isGenerating && !state.generatingField}
                 >
                    AI Khởi tạo nhanh
                 </Button>
             </div>

             <div className="flex-1 min-w-[80px]">
                 <Button 
                    variant="ghost" 
                    icon={<Upload size={16}/>} 
                    className="w-full py-1 text-[10px] md:text-sm"
                    onClick={handleImportClick}
                 >
                    Nhập
                 </Button>
             </div>
             
             <div className="flex-1 min-w-[80px]">
                 <Button 
                    variant="ghost" 
                    icon={<Download size={16}/>} 
                    className="w-full py-1 text-[10px] md:text-sm"
                    onClick={handleExportWorld}
                 >
                    Xuất
                 </Button>
             </div>

             <div className="flex-[2] min-w-[150px]">
                  <Button 
                    variant="primary" 
                    className="w-full py-1 shadow-[0_0_20px_rgba(56,189,248,0.4)] text-[10px] md:text-sm"
                    icon={<Play size={20} />}
                    disabled={state.entities.length < 1}
                    onClick={handleStartGame}
                  >
                    BẮT ĐẦU GAME
                  </Button>
             </div>
         </div>
      </div>

      {showEntityForm && (
        <EntityForm 
            initialData={editingEntityId ? state.entities.find(e => e.id === editingEntityId) : undefined}
            onCancel={() => setShowEntityForm(false)}
            onSave={(entity) => {
                if (editingEntityId) {
                    dispatch({ type: 'UPDATE_ENTITY', id: editingEntityId, entity });
                } else {
                    dispatch({ type: 'ADD_ENTITY', entity });
                }
                setShowEntityForm(false);
            }}
        />
      )}
    </div>
  );
};

const InputGroup = ({ label, value, onChange, placeholder, onAi, loading = false }: { label: string, value: string, onChange: (v: string) => void, placeholder?: string, onAi?: () => void, loading?: boolean }) => (
    <div className="mb-1 relative flex flex-col">
        <div className="flex justify-between items-center mb-0.5">
            <label className="text-xs font-medium text-mystic-accent">{label}</label>
            {onAi && (
                <button 
                    type="button"
                    onClick={(e) => {
                        e.stopPropagation();
                        onAi();
                    }} 
                    disabled={loading} 
                    className="group flex items-center gap-1 px-1.5 py-0.5 rounded border border-slate-300 dark:border-slate-700 bg-stone-50 dark:bg-mystic-900 hover:bg-mystic-accent/10 hover:border-mystic-accent hover:text-mystic-accent text-[10px] text-slate-500 dark:text-slate-400 transition-all cursor-pointer z-10"
                >
                    {loading ? (
                        <span className="animate-spin block w-2.5 h-2.5 border-2 border-mystic-accent border-t-transparent rounded-full" />
                    ) : (
                        <Sparkles size={10} className="group-hover:scale-110 transition-transform" />
                    )}
                    <span>AI Gợi ý</span>
                </button>
            )}
        </div>
        <input 
            type="text" 
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="w-full bg-stone-100 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 rounded p-1.5 text-stone-900 dark:text-slate-100 outline-none focus:border-mystic-accent transition-colors text-xs"
            placeholder={placeholder}
        />
    </div>
);

const TimeInput = ({ label, value, onChange, min, max }: { label: string, value: number, onChange: (v: number) => void, min?: number, max?: number }) => (
    <div className="flex flex-col gap-0.5">
        <label className="text-[9px] uppercase font-bold text-slate-500 dark:text-slate-400 text-center">{label}</label>
        <input 
            type="number" 
            value={value}
            min={min}
            max={max}
            onChange={(e) => {
                const val = parseInt(e.target.value);
                if (!isNaN(val)) onChange(val);
            }}
            className="w-full bg-stone-100 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 rounded p-1 text-stone-900 dark:text-slate-100 outline-none focus:border-mystic-accent text-center text-xs font-mono"
        />
    </div>
);

const TextAreaGroup = ({ label, value, onChange, onAi, height = 'h-24', loading = false, placeholder }: { label: string, value: string, onChange: (v: string) => void, onAi?: () => void, height?: string, loading?: boolean, placeholder?: string }) => {
    const [isPreview, setIsPreview] = useState(false);
    
    return (
        <div className="relative flex flex-col mb-1">
            <div className="flex justify-between items-center mb-0.5">
                <div className="flex items-center gap-2">
                    <label className="text-xs font-medium text-mystic-accent">{label}</label>
                    <button 
                        type="button"
                        onClick={() => setIsPreview(!isPreview)}
                        className="text-slate-400 hover:text-mystic-accent transition-colors"
                        title={isPreview ? "Chỉnh sửa" : "Xem trước Markdown"}
                    >
                        {isPreview ? <EyeOff size={12} /> : <Eye size={12} />}
                    </button>
                </div>
                {onAi && (
                    <button 
                        type="button"
                        onClick={(e) => {
                            e.stopPropagation();
                            onAi();
                        }} 
                        disabled={loading} 
                        className="group flex items-center gap-1 px-1.5 py-0.5 rounded border border-slate-300 dark:border-slate-700 bg-stone-50 dark:bg-mystic-900 hover:bg-mystic-accent/10 hover:border-mystic-accent hover:text-mystic-accent text-[10px] text-slate-500 dark:text-slate-400 transition-all cursor-pointer z-10"
                    >
                        {loading ? (
                            <span className="animate-spin block w-2.5 h-2.5 border-2 border-mystic-accent border-t-transparent rounded-full" />
                        ) : (
                            <Sparkles size={10} className="group-hover:scale-110 transition-transform" />
                        )}
                        <span>AI Gợi ý</span>
                    </button>
                )}
            </div>
            {isPreview ? (
                <div className={`w-full bg-stone-100 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-stone-900 dark:text-slate-100 overflow-y-auto custom-scrollbar text-xs ${height}`}>
                    <MarkdownRenderer content={value || "*Chưa có nội dung*"} />
                </div>
            ) : (
                <textarea 
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    className={`w-full bg-stone-100 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-stone-900 dark:text-slate-100 outline-none focus:border-mystic-accent transition-colors text-xs resize-none custom-scrollbar ${height}`}
                    placeholder={placeholder}
                />
            )}
        </div>
    );
};

export default WorldCreationScreen;
