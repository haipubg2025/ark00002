
import React, { useReducer, useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ArrowLeft, Sparkles, Plus, Trash2, Edit2, Wand2, Play, 
  User, Globe, Settings, Users, Upload, Download, Clock,
  X, Search, ChevronRight, Eye, EyeOff
} from 'lucide-react';
import { NavigationProps, GameState, WorldData, AppSettings } from '../../../types';
import Button from '../../ui/Button';
import MarkdownRenderer from '../../common/MarkdownRenderer';
import { initialWorldState, worldCreationReducer } from '../world-creation/reducer';
import EntityForm from '../world-creation/EntityForm';
import { worldAiService } from '../../../services/ai/world-creation/service';
import { fanficAiService, FanficCharacter } from '../../../services/ai/fanfic/service';
import { dbService } from '../../../services/db/indexedDB';

const TABS = [
  { id: 0, label: "Nhân vật", icon: User },
  { id: 1, label: "Thế giới", icon: Globe },
  { id: 2, label: "Quy tắc", icon: Settings },
  { id: 3, label: "Thực thể", icon: Users },
];

interface FanficScreenProps extends NavigationProps {
  onGameStart?: (data: WorldData) => void;
  initialData?: WorldData | null;
}

interface WorkCharacter {
  name?: string;
  role?: string;
  gender?: string;
  age?: string;
  personality?: string;
  description?: string;
  appearance?: string;
  skills?: string;
  goal?: string;
  lineage?: string;
  identities?: { name: string; role: string }[];
}

interface ImportedWork {
  title?: string;
  worldSetting?: string;
  plot?: string;
  characters?: WorkCharacter[];
}

const FanficScreen: React.FC<FanficScreenProps> = ({ onNavigate, onGameStart, initialData }) => {
  const [state, dispatch] = useReducer(worldCreationReducer, initialWorldState);
  const [showEntityForm, setShowEntityForm] = useState(false);
  const [editingEntityId, setEditingEntityId] = useState<string | null>(null);
  const [originalWorkName, setOriginalWorkName] = useState('');
  const [originalContent, setOriginalContent] = useState('');
  const [analyzedCharacters, setAnalyzedCharacters] = useState<FanficCharacter[]>([]);
  const [importedWorks, setImportedWorks] = useState<ImportedWork[]>([]);
  const [showWorkSelector, setShowWorkSelector] = useState(false);
  const [workSearchTerm, setWorkSearchTerm] = useState('');
  const [workFilterCountry, setWorkFilterCountry] = useState('All');
  const [workFilterGenre, setWorkFilterGenre] = useState('All');
  const [workSortBy, setWorkSortBy] = useState<'title' | 'chars_desc' | 'chars_asc'>('title');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isAnalysisCollapsed, setIsAnalysisCollapsed] = useState(false);
  const [aiModel, setAiModel] = useState<string>('gemini-3-pro-preview');
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const txtFileInputRef = useRef<HTMLInputElement>(null);
  
  const handleAnalyze = async () => {
    if (!originalContent.trim() && !originalWorkName.trim()) {
      return;
    }

    setIsAnalyzing(true);
    try {
      const result = await fanficAiService.analyzeFanfic(originalContent, originalWorkName, settings || undefined);
      setAnalyzedCharacters(result.characters);
      
      // Tự động cập nhật bối cảnh thế giới từ tóm tắt
      dispatch({ type: 'UPDATE_WORLD', field: 'context', value: result.summary });
      setIsAnalysisCollapsed(true);
    } catch (error) {
      console.error("Analyze Error", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleTxtFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      setOriginalContent(e.target?.result as string);
    };
    reader.readAsText(file);
    event.target.value = '';
  };

  // Initial Load (Settings & Import Data)
  useEffect(() => {
    dbService.getSettings().then(s => {
      setSettings(s);
      if (s.aiModel) setAiModel(s.aiModel);
    });

    // Check if there is initial data passed
    if (initialData) {
       dispatch({ type: 'IMPORT_DATA', payload: initialData });
    }
  }, [initialData]);

  // --- AI Helper Function ---
  const handleAiGenerate = async (field: string, category: 'player' | 'world') => {
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
      const contextData = category === 'player' 
        ? { ...state.player, genre: state.world.genre } 
        : { genre: state.world.genre, worldName: state.world.worldName, originalWorkName };

      let currentValue = "";
      if (category === 'player') {
          // @ts-expect-error - Dynamic access
          currentValue = state.player[field] || "";
      } else {
          // @ts-expect-error - Dynamic access
          currentValue = state.world[field] || "";
      }

      const content = await worldAiService.generateFieldContent(category, field, contextData, aiModel, currentValue, settings || undefined);
      
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
    if (!originalWorkName.trim()) return;
    dispatch({ type: 'SET_GENERATING', isGenerating: true });
    try {
      // Sử dụng tên tác phẩm gốc và tóm tắt (nếu có) làm ý tưởng để AI tạo thế giới đồng nhân
      let concept = `Đồng nhân của tác phẩm: ${originalWorkName}`;
      if (state.world.context) {
        concept += `\n\nBối cảnh và nội dung gốc:\n${state.world.context}`;
      }
      
      // Thu thập dữ liệu hiện tại để AI tôn trọng
      const existingData = {
        player: state.player,
        world: state.world,
        entities: state.entities,
        rules: state.config.rules,
        gameTime: state.gameTime
      };

      const data = await worldAiService.generateFullWorld(concept, aiModel, settings || undefined, existingData);
      dispatch({ type: 'AUTO_FILL_ALL', payload: data });
    } catch (error: unknown) {
      console.error("AI AutoFill Error", error);
    } finally {
      dispatch({ type: 'SET_GENERATING', isGenerating: false });
    }
  };

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
    const fileName = `FANFIC_${worldName}_${playerName}_${timestamp}.json`;
    
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

  const applyWorkData = (work: ImportedWork) => {
    setOriginalWorkName(work.title || '');
    
    // Kết hợp bối cảnh và cốt truyện để làm context thế giới
    let context = "";
    if (work.worldSetting) context += `BỐI CẢNH:\n${work.worldSetting}\n\n`;
    if (work.plot) context += `CỐT TRUYỆN:\n${work.plot}`;
    
    dispatch({ type: 'UPDATE_WORLD', field: 'context', value: context.trim() });
    
    // Chuyển đổi danh sách nhân vật sang định dạng FanficCharacter
    if (work.characters && Array.isArray(work.characters)) {
      const mappedChars: FanficCharacter[] = work.characters.map((c: WorkCharacter) => {
        let background = c.description || '';
        if (c.lineage) background += `\nLai lịch: ${c.lineage}`;
        if (c.identities && Array.isArray(c.identities)) {
          background += `\nDanh tính khác: ${c.identities.map((i: { name: string; role: string }) => `${i.name} (${i.role})`).join(', ')}`;
        }

        return {
          name: c.name || '',
          role: c.role || '',
          gender: c.gender || 'Chưa rõ',
          age: c.age || 'Chưa rõ',
          personality: c.personality || 'Chưa rõ',
          background: background.trim() || 'Chưa rõ',
          appearance: c.appearance || 'Chưa rõ',
          skills: c.skills || 'Chưa rõ',
          goal: c.goal || 'Chưa rõ'
        };
      });
      setAnalyzedCharacters(mappedChars);
      
      // Thêm các nhân vật quan trọng vào danh sách thực thể (NPC) nếu chưa có nhiều
      if (state.entities.length < 5) {
        work.characters.slice(0, 15).forEach((c: WorkCharacter) => {
          // Kiểm tra xem thực thể đã tồn tại chưa
          if (c.name && !state.entities.some(e => e.name === c.name)) {
            let desc = c.description || c.role || 'Nhân vật từ tác phẩm gốc';
            if (c.identities && Array.isArray(c.identities)) {
              desc += ` (Danh tính: ${c.identities.map((i: { name: string; role: string }) => i.name).join(', ')})`;
            }

            dispatch({ 
              type: 'ADD_ENTITY', 
              entity: {
                name: c.name,
                type: 'NPC',
                description: desc,
                properties: {
                  role: c.role,
                  lineage: c.lineage,
                  identities: c.identities
                }
              }
            });
          }
        });
      }
    }
    
    setIsAnalysisCollapsed(true);
    setShowWorkSelector(false);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const parsedData = JSON.parse(content);
        
        // Kiểm tra xem đây là danh sách tác phẩm gốc (định dạng người dùng cung cấp)
        if (Array.isArray(parsedData) && parsedData.length > 0 && (parsedData[0].title || parsedData[0].worldSetting)) {
          setImportedWorks(parsedData);
          setShowWorkSelector(true);
        } 
        // Kiểm tra nếu là một tác phẩm đơn lẻ
        else if (parsedData.title && (parsedData.worldSetting || parsedData.characters)) {
          applyWorkData(parsedData);
        }
        // Đây là dữ liệu lưu game chuẩn (WorldData)
        else if (parsedData.player && parsedData.world) {
          dispatch({ type: 'IMPORT_DATA', payload: parsedData as WorldData });
        } else {
          console.warn("Định dạng tệp không được hỗ trợ");
        }
      } catch (error: unknown) {
        console.error("Import Error", error);
      }
    };
    reader.readAsText(file);
    event.target.value = '';
  };

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
             id: `autosave-fanfic-${Date.now()}`,
             name: `[Đồng Nhân] ${worldData.world.worldName}`,
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

  const renderPlayerTab = () => (
    <div className="space-y-3 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <h3 className="text-base font-bold text-slate-800 dark:text-slate-200 border-b border-slate-200 dark:border-slate-700 pb-0.5 mb-1 flex items-center gap-2" style={{ fontFamily: 'Arial', lineHeight: '18px' }}>
        <User size={18} className="text-mystic-accent" /> Thiết lập Nhân Vật Chính
      </h3>

      {analyzedCharacters.length > 0 && (
        <div className="bg-mystic-accent/5 p-3 rounded-lg border border-mystic-accent/20 mb-4">
          <label className="block text-xs font-bold text-mystic-accent uppercase mb-2">Chọn nhân vật từ tác phẩm gốc</label>
          <select 
            className="w-full bg-white dark:bg-slate-900 border border-stone-300 dark:border-slate-700 rounded p-2 text-stone-900 dark:text-slate-100 outline-none focus:border-mystic-accent text-sm"
            onChange={(e) => {
              const charName = e.target.value;
              if (charName === 'new') {
                dispatch({ type: 'UPDATE_PLAYER', field: 'name', value: '' });
                return;
              }
              const char = analyzedCharacters.find(c => c.name === charName);
              if (char) {
                dispatch({ type: 'UPDATE_PLAYER', field: 'name', value: char.name });
                dispatch({ type: 'UPDATE_PLAYER', field: 'gender', value: char.gender });
                dispatch({ type: 'UPDATE_PLAYER', field: 'age', value: char.age });
                dispatch({ type: 'UPDATE_PLAYER', field: 'personality', value: char.personality });
                dispatch({ type: 'UPDATE_PLAYER', field: 'background', value: char.background });
                dispatch({ type: 'UPDATE_PLAYER', field: 'appearance', value: char.appearance });
                dispatch({ type: 'UPDATE_PLAYER', field: 'skills', value: char.skills });
                dispatch({ type: 'UPDATE_PLAYER', field: 'goal', value: char.goal });
              }
            }}
          >
            <option value="new">-- Tạo nhân vật mới tự do --</option>
            {analyzedCharacters.map(char => (
              <option key={char.name} value={char.name}>{char.name} ({char.role})</option>
            ))}
          </select>
          <p className="text-[10px] text-slate-500 mt-1 italic">* Chọn một nhân vật để tự động điền thông tin chi tiết.</p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="space-y-2">
          <InputGroup 
            label="Tên nhân vật" 
            value={state.player.name} 
            onChange={(v) => dispatch({ type: 'UPDATE_PLAYER', field: 'name', value: v })}
            placeholder="Ví dụ: Tôn Ngộ Không, Harry Potter..." 
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
              placeholder="Mô tả tính cách nhân vật..."
          />
          <TextAreaGroup 
              label="Ngoại hình" 
              value={state.player.appearance} 
              onChange={(v) => dispatch({ type: 'UPDATE_PLAYER', field: 'appearance', value: v })}
              onAi={() => handleAiGenerate('appearance', 'player')}
              loading={state.isGenerating && state.generatingField === 'appearance'}
              height="h-24"
              placeholder="Mô tả ngoại hình nhân vật..."
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
              placeholder="Tiểu sử nhân vật trong thế giới đồng nhân..."
          />
          <TextAreaGroup 
              label="Kỹ năng" 
              value={state.player.skills} 
              onChange={(v) => dispatch({ type: 'UPDATE_PLAYER', field: 'skills', value: v })} 
              onAi={() => handleAiGenerate('skills', 'player')}
              loading={state.isGenerating && state.generatingField === 'skills'}
              height="h-24"
              placeholder="Các kỹ năng, phép thuật..."
          />
          <TextAreaGroup 
              label="Mục tiêu" 
              value={state.player.goal} 
              onChange={(v) => dispatch({ type: 'UPDATE_PLAYER', field: 'goal', value: v })} 
              onAi={() => handleAiGenerate('goal', 'player')}
              loading={state.isGenerating && state.generatingField === 'goal'}
              height="h-24"
              placeholder="Mục tiêu của nhân vật..."
          />
        </div>
      </div>
    </div>
  );

  const renderWorldTab = () => {
    return (
      <div className="space-y-3 animate-in fade-in slide-in-from-bottom-4 duration-500">
        <h3 className="text-base font-bold text-slate-800 dark:text-slate-200 border-b border-slate-200 dark:border-slate-700 pb-0.5 mb-1 flex items-center gap-2" style={{ fontFamily: 'Arial', lineHeight: '18px' }}>
          <Globe size={18} className="text-mystic-accent" /> Bối Cảnh Thế Giới Đồng Nhân
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <InputGroup 
              label="Tên thế giới" 
              value={state.world.worldName} 
              onChange={(v) => dispatch({ type: 'UPDATE_WORLD', field: 'worldName', value: v })} 
              onAi={() => handleAiGenerate('worldName', 'world')}
              loading={state.isGenerating && state.generatingField === 'worldName'}
              placeholder="Ví dụ: Hỏa Ảnh Thế Giới, Tam Quốc Diễn Nghĩa..."
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
          placeholder="Mô tả bối cảnh thế giới đồng nhân..."
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
                <TimeInput label="Năm" value={state.gameTime.year} onChange={(v) => dispatch({ type: 'UPDATE_GAME_TIME', field: 'year', value: v })} />
                <TimeInput label="Tháng" value={state.gameTime.month} min={1} max={12} onChange={(v) => dispatch({ type: 'UPDATE_GAME_TIME', field: 'month', value: v })} />
                <TimeInput label="Ngày" value={state.gameTime.day} min={1} max={31} onChange={(v) => dispatch({ type: 'UPDATE_GAME_TIME', field: 'day', value: v })} />
                <TimeInput label="Giờ" value={state.gameTime.hour} min={0} max={23} onChange={(v) => dispatch({ type: 'UPDATE_GAME_TIME', field: 'hour', value: v })} />
                <TimeInput label="Phút" value={state.gameTime.minute} min={0} max={59} onChange={(v) => dispatch({ type: 'UPDATE_GAME_TIME', field: 'minute', value: v })} />
            </div>
        </div>

        <TextAreaGroup 
          label="Kịch bản khởi đầu (Tùy chọn)" 
          value={state.world.startingScenario || ''} 
          onChange={(v) => dispatch({ type: 'UPDATE_WORLD', field: 'startingScenario', value: v })} 
          height="h-28"
          placeholder="Nhập hành động hoặc tình huống bắt đầu cụ thể..."
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
                            placeholder="Nhập quy tắc..."
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
             Thêm các nhân vật phụ hoặc địa danh quan trọng từ tác phẩm gốc.
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
                </div>
            ))}
            {state.entities.length === 0 && (
                <div className="col-span-full border-2 border-dashed border-slate-200 dark:border-slate-700 rounded-lg flex items-center justify-center h-32 text-slate-400 dark:text-slate-500">
                    Chưa có thực thể nào.
                </div>
            )}
       </div>
    </div>
  );

  return (
    <div className="flex flex-col h-full w-full relative overflow-hidden bg-stone-100 dark:bg-mystic-950">
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
            <h2 className="text-lg font-bold text-mystic-accent text-center" style={{ fontFamily: 'Arial', lineHeight: '18px' }}>Đồng Nhân</h2>
            <div className="w-10 sm:w-20" />
          </div>

          <div className="mb-2 space-y-2 bg-slate-100 dark:bg-mystic-800/40 p-2 rounded-lg border border-slate-200 dark:border-mystic-accent/20">
             <div className="flex gap-2 items-center">
                <Wand2 className="text-mystic-accent shrink-0" size={16} />
                <input 
                    value={originalWorkName}
                    onChange={(e) => setOriginalWorkName(e.target.value)}
                    placeholder="Tên Tác Phẩm Gốc..."
                    className="flex-1 bg-transparent border-none text-xs text-slate-800 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none"
                />
                <button 
                    onClick={() => setIsAnalysisCollapsed(!isAnalysisCollapsed)}
                    className="text-[10px] text-mystic-accent hover:underline"
                >
                    {isAnalysisCollapsed ? 'Mở' : 'Ẩn'}
                </button>
             </div>

             <AnimatePresence>
                {!isAnalysisCollapsed && (
                    <motion.div 
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                    >
                        <div className="space-y-1.5 pt-1.5 border-t border-slate-200 dark:border-slate-700/50">
                            <div className="flex justify-between items-center">
                                <label className="text-[9px] font-bold text-mystic-accent uppercase tracking-wider">Nội dung tác phẩm gốc</label>
                                <div className="flex gap-1.5">
                                    <input type="file" ref={txtFileInputRef} onChange={handleTxtFileChange} accept=".txt" className="hidden" />
                                    <button onClick={() => txtFileInputRef.current?.click()} className="text-[9px] flex items-center gap-1 px-1.5 py-0.5 rounded border border-slate-300 dark:border-slate-700 hover:border-mystic-accent transition-colors text-slate-500">
                                        <Upload size={10} /> Tải TXT
                                    </button>
                                    <button onClick={() => setOriginalContent('')} className="text-[9px] flex items-center gap-1 px-1.5 py-0.5 rounded border border-slate-300 dark:border-slate-700 hover:text-red-500 transition-colors text-slate-500">
                                        <Trash2 size={10} /> Xóa
                                    </button>
                                </div>
                            </div>
                            <textarea 
                                value={originalContent}
                                onChange={(e) => setOriginalContent(e.target.value)}
                                placeholder="Dán nội dung truyện vào đây..."
                                className="w-full h-20 bg-white/50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-700 rounded p-1.5 text-[11px] text-slate-800 dark:text-slate-200 focus:border-mystic-accent outline-none resize-none custom-scrollbar"
                            />
                            <Button 
                                variant="primary" 
                                className="w-full py-1.5 text-[10px] font-bold tracking-widest" 
                                onClick={handleAnalyze}
                                isLoading={isAnalyzing}
                                icon={<Sparkles size={12} />}
                            >
                                PHÂN TÍCH TÁC PHẨM
                            </Button>
                        </div>
                    </motion.div>
                )}
             </AnimatePresence>
          </div>

          <div className="flex border-b border-slate-200 dark:border-slate-700 overflow-x-auto no-scrollbar md:justify-start">
             {TABS.map((tab) => (
                 <button
                    key={tab.id}
                    onClick={() => dispatch({ type: 'SET_TAB', payload: tab.id })}
                    className={`flex-1 md:flex-none px-3 py-2 text-xs font-bold transition-colors whitespace-nowrap relative flex justify-center items-center gap-1.5 ${
                        state.currentTab === tab.id ? 'text-mystic-accent' : 'text-slate-400 dark:text-slate-500 hover:text-slate-900 dark:hover:text-slate-300'
                    }`}
                 >
                    <tab.icon size={16} />
                    <span className="hidden xs:inline">{tab.label}</span>
                    {state.currentTab === tab.id && (
                        <motion.div layoutId="activeTabFanfic" className="absolute bottom-0 left-0 right-0 h-0.5 bg-mystic-accent" />
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
         </AnimatePresence>
      </div>

      <div className="shrink-0 z-20 bg-stone-200/95 dark:bg-mystic-950/95 backdrop-blur-md border-t border-stone-400 dark:border-slate-800 p-0.5 shadow-lg">
         <div className="flex flex-row gap-1 md:gap-4 justify-between items-center w-full overflow-x-auto no-scrollbar">
              <div className="flex-1 min-w-[120px]">
                  <Button 
                    variant="ghost" 
                    className="w-full py-1 text-[10px] md:text-sm bg-mystic-accent/10 hover:bg-mystic-accent/20" 
                    onClick={handleAutoFillAll}
                    isLoading={state.isGenerating && !state.generatingField}
                    icon={<Wand2 size={16} />}
                  >
                    AI Khởi tạo nhanh
                  </Button>
              </div>
              <div className="flex-1 min-w-[120px]">
                  <Button 
                    variant="ghost" 
                    className="w-full py-1 text-[10px] md:text-sm bg-amber-500/10 hover:bg-amber-500/20 text-amber-600 border-amber-200" 
                    onClick={() => setShowWorkSelector(true)}
                    disabled={importedWorks.length === 0}
                    icon={<Clock size={16} />}
                  >
                    FILE ĐN
                  </Button>
              </div>
              <div className="flex-1 min-w-[80px]">
                  <Button variant="ghost" icon={<Upload size={16}/>} className="w-full py-1 text-[10px] md:text-sm" onClick={handleImportClick}>Nhập</Button>
              </div>
              <div className="flex-1 min-w-[80px]">
                  <Button variant="ghost" icon={<Download size={16}/>} className="w-full py-1 text-[10px] md:text-sm" onClick={handleExportWorld}>Xuất</Button>
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

      {showWorkSelector && (
        <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/20 backdrop-blur-[2px] animate-in fade-in duration-300">
          <div 
            className="fixed inset-0" 
            onClick={() => setShowWorkSelector(false)} 
          />
          <motion.div 
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="relative bg-white dark:bg-slate-900 w-full max-w-md h-full shadow-[-20px_0_50px_rgba(0,0,0,0.1)] border-l border-slate-200 dark:border-slate-800 flex flex-col overflow-hidden"
          >
            {/* Header - Minimal & Clean */}
            <div className="px-6 py-5 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-mystic-accent/10 rounded-lg flex items-center justify-center">
                  <Sparkles className="text-mystic-accent" size={16} />
                </div>
                <h3 className="font-bold text-slate-900 dark:text-white">Thư viện tác phẩm</h3>
              </div>
              
              <div className="flex items-center gap-1">
                <button 
                  onClick={() => {
                    setImportedWorks([]);
                    setShowWorkSelector(false);
                  }}
                  className="p-2 text-slate-400 hover:text-red-500 transition-colors"
                  title="Xóa tất cả"
                >
                  <Trash2 size={16} />
                </button>
                <button 
                  onClick={() => setShowWorkSelector(false)}
                  className="p-2 text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors"
                >
                  <X size={20} />
                </button>
              </div>
            </div>

            {/* Search & Filters - Compact List Style */}
            <div className="p-4 space-y-3 bg-slate-50/50 dark:bg-slate-950/50 border-b border-slate-100 dark:border-slate-800">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={14} />
                <input
                  type="text"
                  placeholder="Tìm kiếm tác phẩm..."
                  value={workSearchTerm}
                  onChange={(e) => setWorkSearchTerm(e.target.value)}
                  className="w-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg pl-9 pr-4 py-2 text-xs outline-none focus:ring-1 focus:ring-mystic-accent transition-all"
                />
              </div>
              <div className="grid grid-cols-3 gap-2">
                <select
                  value={workFilterCountry}
                  onChange={(e) => setWorkFilterCountry(e.target.value)}
                  className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-md px-2 py-1.5 text-[10px] font-bold outline-none"
                >
                  <option value="All">Quốc gia</option>
                  {Array.from(new Set(importedWorks.map(w => w.country).filter(Boolean))).sort().map(c => (
                    <option key={c as string} value={c as string}>{c as string}</option>
                  ))}
                </select>
                <select
                  value={workFilterGenre}
                  onChange={(e) => setWorkFilterGenre(e.target.value)}
                  className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-md px-2 py-1.5 text-[10px] font-bold outline-none"
                >
                  <option value="All">Thể loại</option>
                  {Array.from(new Set(importedWorks.map(w => w.genre).filter(Boolean))).sort().map(g => (
                    <option key={g as string} value={g as string}>{g as string}</option>
                  ))}
                </select>
                <select
                  value={workSortBy}
                  onChange={(e) => setWorkSortBy(e.target.value as 'title' | 'chars_desc' | 'chars_asc')}
                  className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-md px-2 py-1.5 text-[10px] font-bold outline-none"
                >
                  <option value="title">Tên A-Z</option>
                  <option value="chars_desc">Nhiều NV</option>
                  <option value="chars_asc">Ít NV</option>
                </select>
              </div>
            </div>
            
            {/* List Content - Clean & Dense */}
            <div className="flex-1 overflow-y-auto custom-scrollbar">
              {importedWorks
                .filter(work => {
                  const matchesSearch = 
                    (work.title?.toLowerCase() || "").includes(workSearchTerm.toLowerCase()) || 
                    (work.description?.toLowerCase() || "").includes(workSearchTerm.toLowerCase());
                  const matchesCountry = workFilterCountry === 'All' || work.country === workFilterCountry;
                  const matchesGenre = workFilterGenre === 'All' || work.genre === workFilterGenre;
                  return matchesSearch && matchesCountry && matchesGenre;
                })
                .sort((a, b) => {
                  if (workSortBy === 'title') return (a.title || "").localeCompare(b.title || "");
                  if (workSortBy === 'chars_desc') return (b.characters?.length || 0) - (a.characters?.length || 0);
                  if (workSortBy === 'chars_asc') return (a.characters?.length || 0) - (b.characters?.length || 0);
                  return 0;
                })
                .map((work, idx) => (
                <button
                  key={work.id || idx}
                  onClick={() => applyWorkData(work)}
                  className="w-full flex flex-col p-5 border-b border-slate-50 dark:border-slate-800/50 hover:bg-mystic-accent/5 transition-all text-left group"
                >
                  <div className="flex justify-between items-start mb-1">
                    <h4 className="font-bold text-slate-800 dark:text-slate-200 group-hover:text-mystic-accent transition-colors line-clamp-1 text-sm">
                      {work.title}
                    </h4>
                    <ChevronRight size={14} className="text-slate-300 group-hover:text-mystic-accent group-hover:translate-x-1 transition-all" />
                  </div>
                  
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 line-clamp-2 mb-3 leading-relaxed">
                    {work.description || work.plot || "Không có mô tả."}
                  </p>
                  
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1 text-[9px] font-bold text-slate-400 uppercase tracking-wider">
                      <Users size={10} className="text-mystic-accent" />
                      <span>{work.characters?.length || 0} Nhân vật</span>
                    </div>
                    {work.genre && (
                      <span className="text-[9px] font-bold text-mystic-accent bg-mystic-accent/10 px-1.5 py-0.5 rounded">
                        {work.genre}
                      </span>
                    )}
                    {work.country && (
                      <span className="text-[9px] font-bold text-slate-400 bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded">
                        {work.country}
                      </span>
                    )}
                  </div>
                </button>
              ))}
              
              {/* Empty State */}
              {importedWorks.length > 0 && importedWorks.filter(work => {
                  const matchesSearch = 
                    (work.title?.toLowerCase() || "").includes(workSearchTerm.toLowerCase());
                  const matchesCountry = workFilterCountry === 'All' || work.country === workFilterCountry;
                  const matchesGenre = workFilterGenre === 'All' || work.genre === workFilterGenre;
                  return matchesSearch && matchesCountry && matchesGenre;
                }).length === 0 && (
                <div className="flex flex-col items-center justify-center py-20 text-slate-400 px-10 text-center">
                  <Search size={32} className="mb-4 opacity-10" />
                  <p className="text-sm font-medium">Không tìm thấy tác phẩm phù hợp</p>
                  <button 
                    onClick={() => { setWorkSearchTerm(''); setWorkFilterCountry('All'); setWorkFilterGenre('All'); }}
                    className="mt-4 text-[11px] font-bold text-mystic-accent hover:underline uppercase tracking-widest"
                  >
                    Xóa bộ lọc
                  </button>
                </div>
              )}
            </div>

            {/* Footer - Subtle Stats */}
            <div className="px-6 py-4 bg-slate-50 dark:bg-slate-950/50 border-t border-slate-100 dark:border-slate-800">
              <p className="text-[10px] text-slate-400 font-medium text-center uppercase tracking-[0.2em]">
                Chọn một tác phẩm để áp dụng dữ liệu
              </p>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
};

const InputGroup = ({ label, value, onChange, placeholder }: { label: string, value: string, onChange: (v: string) => void, placeholder?: string }) => (
    <div className="mb-1">
        <label className="block text-xs font-medium text-mystic-accent mb-0.5">{label}</label>
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

export default FanficScreen;
