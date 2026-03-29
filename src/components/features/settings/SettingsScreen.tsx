
import React, { useEffect, useState } from 'react';
import { NavigationProps, GameState, AppSettings, ThinkingBudgetLevel, ThinkingLevel, NarrativePerspective } from '../../../types';
import SafetySettings from './SafetySettings';
import { dbService, DEFAULT_SETTINGS } from '../../../services/db/indexedDB';
import Button from '../../ui/Button';
import { useTheme } from '../../../context/ThemeContext';
import { DIFFICULTY_LEVELS, OUTPUT_LENGTHS } from '../../../constants/promptTemplates';

interface SettingsScreenProps extends NavigationProps {
  fromGame?: boolean;
}

const SettingsScreen: React.FC<SettingsScreenProps> = ({ onNavigate, fromGame }) => {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [activeTab, setActiveTab] = useState<'general' | 'api'>('general');
  const { setTheme, setFontFamily, setFontSize, setVisualEffects } = useTheme();
  const [localFontSize, setLocalFontSize] = useState<string>('');

  useEffect(() => {
    const load = async () => {
      const s = await dbService.getSettings();
      setSettings(s);
      if (s) {
        setLocalFontSize(s.fontSize.toString());
      }
    };
    load();
  }, []);

  const handleChange = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    setSettings(prev => {
      if (!prev) return null;
      const newSettings = { ...prev, [key]: value };
      
      // Side effects should be triggered after state update
      setTimeout(() => {
        dbService.saveSettings(newSettings);
        if (key === 'theme') setTheme(value as 'light' | 'dark');
        if (key === 'systemFont') setFontFamily(value as string);
        if (key === 'fontSize') {
          setFontSize(value as number);
          setLocalFontSize(value.toString());
        }
        if (key === 'visualEffects') setVisualEffects(value as boolean);
      }, 0);
      
      return newSettings;
    });
  };

  const handleMultipleChanges = (changes: Partial<AppSettings>) => {
    setSettings(prev => {
      if (!prev) return null;
      const newSettings = { ...prev, ...changes };
      
      // Side effects should be triggered after state update
      setTimeout(() => {
        dbService.saveSettings(newSettings);
        Object.entries(changes).forEach(([key, value]) => {
          if (key === 'theme') setTheme(value as 'light' | 'dark');
          if (key === 'systemFont') setFontFamily(value as string);
          if (key === 'fontSize') {
            setFontSize(value as number);
            setLocalFontSize(value.toString());
          }
          if (key === 'visualEffects') setVisualEffects(value as boolean);
        });
      }, 0);
      
      return newSettings;
    });
  };

  const handleGlobalUpdate = (newSettings: AppSettings) => {
    setSettings(newSettings);
    dbService.saveSettings(newSettings);
  };

  const handleSave = async () => {
    if (!settings) return;
    setIsSaving(true);
    await dbService.saveSettings(settings);
    setIsSaving(false);
    onNavigate(fromGame ? GameState.PLAYING : GameState.MENU);
  };

  const handleResetFactory = async () => {
      setSettings(DEFAULT_SETTINGS);
      await dbService.saveSettings(DEFAULT_SETTINGS);
  };

  const handleLoadModels = async () => {
    if (!settings?.proxyUrl && !settings?.proxyUrl2) {
      return;
    }
    
    setIsSaving(true);
    let updatedSettings = { ...settings };

    const loadFromProxy = async (url: string, key: string, currentModel: string) => {
      try {
        const response = await fetch(`${url}/models`, {
          headers: {
            'Authorization': key ? `Bearer ${key}` : '',
            'x-goog-api-key': key || ''
          }
        });
        
        if (!response.ok) {
          const errorText = await response.text().catch(() => "");
          throw new Error(`Lỗi Proxy (${response.status}): ${errorText || "Không thể tải danh sách model"}`);
        }
        
        const data = await response.json();
        let modelList: string[] = [];
        
        if (Array.isArray(data.data)) {
          modelList = data.data.map((m: { id: string }) => m.id);
        } else if (Array.isArray(data.models)) {
          modelList = data.models.map((m: { name: string }) => m.name.replace('models/', ''));
        }
        
        if (modelList.length > 0) {
          return {
            models: modelList,
            model: currentModel || modelList[0]
          };
        } else {
          throw new Error("Proxy không trả về danh sách model hợp lệ");
        }
      } catch (err: unknown) {
        console.error("Proxy Error:", err);
        throw err;
      }
    };

    try {
      if (settings.proxyUrl) {
        try {
          const result = await loadFromProxy(settings.proxyUrl, settings.proxyKey || '', settings.proxyModel || '');
          updatedSettings = {
            ...updatedSettings,
            proxyModels: result.models,
            proxyModel: result.model
          };
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : String(err);
          console.error(`Proxy 1: ${message}`);
        }
      }

      if (settings.proxyUrl2) {
        try {
          const result = await loadFromProxy(settings.proxyUrl2, settings.proxyKey2 || '', settings.proxyModel2 || '');
          updatedSettings = {
            ...updatedSettings,
            proxyModels2: result.models,
            proxyModel2: result.model
          };
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : String(err);
          console.error(`Proxy 2: ${message}`);
        }
      }
      
      setSettings(updatedSettings);
      await dbService.saveSettings(updatedSettings);
    } catch (err: unknown) {
      console.error("General Proxy Error:", err);
    } finally {
      setIsSaving(false);
    }
  };
  const handleResetApiTab = () => {
    if (settings) {
      setSettings({
        ...settings,
        geminiApiKey: [],
        proxyUrl: '',
        proxyKey: '',
        proxyModel: '',
        proxyModels: [],
        proxyName: '',
        proxyUrl2: '',
        proxyKey2: '',
        proxyModel2: '',
        proxyModels2: [],
        proxyName2: '',
        useGeminiApi: true,
        useProxy: true
      });
    }
  };

  const handleImportTxt = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      
      // Try JSON first
      try {
        const parsed = JSON.parse(content);
        if (settings) {
            setSettings({
                ...settings,
                proxyName: parsed.proxyName || parsed.name || settings.proxyName,
                proxyUrl: parsed.proxyUrl || parsed.url || settings.proxyUrl,
                proxyKey: parsed.proxyKey || parsed.key || settings.proxyKey,
                geminiApiKey: Array.isArray(parsed.geminiApiKey) 
                    ? parsed.geminiApiKey 
                    : (parsed.geminiApiKey ? [parsed.geminiApiKey] : settings.geminiApiKey)
            });
            return;
        }
      } catch {
        // Not JSON, continue to TXT parsing
      }

      // TXT Parsing logic
      const lines = content.split('\n').map(l => l.trim()).filter(l => l !== '');
      const newGeminiKeys: string[] = [...(settings?.geminiApiKey || [])];
      let newProxyUrl = settings?.proxyUrl || '';
      let newProxyKey = settings?.proxyKey || '';
      let newProxyName = settings?.proxyName || '';

      const geminiKeyRegex = /^AIzaSy[A-Za-z0-9_-]{33}$/;

      lines.forEach(line => {
        if (geminiKeyRegex.test(line)) {
          if (!newGeminiKeys.includes(line)) {
            newGeminiKeys.push(line);
          }
        } else if (line.startsWith('http')) {
          newProxyUrl = line;
        } else if (line.toLowerCase().includes('proxy_name:') || line.toLowerCase().includes('name:')) {
          newProxyName = line.split(':')[1]?.trim() || newProxyName;
        } else if (line.toLowerCase().includes('proxy_key:') || line.toLowerCase().includes('key:')) {
          newProxyKey = line.split(':')[1]?.trim() || newProxyKey;
        } else {
          // Heuristic: if it's a long string but doesn't match Gemini regex, it might be a proxy key
          if (line.length > 20 && !newProxyKey) {
            newProxyKey = line;
          }
        }
      });

      if (settings) {
        setSettings({
          ...settings,
          geminiApiKey: newGeminiKeys,
          proxyUrl: newProxyUrl,
          proxyKey: newProxyKey,
          proxyName: newProxyName
        });
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  if (!settings) return <div className="flex items-center justify-center h-full text-slate-400">Đang tải cấu hình...</div>;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-[0.5vh]">
      <div className="flex flex-col h-[99vh] w-[99vw] bg-stone-50 dark:bg-mystic-950 text-slate-900 dark:text-slate-200 overflow-hidden rounded-xl border border-slate-200 dark:border-slate-800 shadow-2xl">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-800 bg-stone-50/95 dark:bg-mystic-950/95 backdrop-blur z-20 flex justify-between items-center shrink-0">
            <div className="flex items-center gap-4">
                <div className={`p-2 rounded-lg ${fromGame ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400' : 'bg-mystic-accent/10 text-mystic-accent'}`}>
                </div>
                <div>
                    <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 leading-tight">
                        {fromGame ? 'Cài đặt Trò chơi' : 'Cài đặt Hệ thống'}
                    </h2>
                    <p className="text-[10px] text-slate-500 dark:text-slate-400 uppercase font-bold tracking-widest">
                        {fromGame ? 'Đang trong phiên chơi' : 'Cấu hình ứng dụng'}
                    </p>
                </div>
            </div>

            <div className="flex items-center gap-8">
                {/* Tab Switcher Integrated in Header */}
                <div className="flex bg-stone-300 dark:bg-mystic-900/50 rounded-lg p-1 border border-stone-400 dark:border-slate-800">
                    <button 
                        onClick={() => setActiveTab('general')}
                        className={`px-6 py-2 text-sm font-medium transition-all rounded-md ${activeTab === 'general' ? 'text-white bg-mystic-accent shadow-lg' : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'}`}
                    >
                        Chung
                    </button>
                    <button 
                        onClick={() => setActiveTab('api')}
                        className={`px-6 py-2 text-sm font-medium transition-all rounded-md ${activeTab === 'api' ? 'text-white bg-mystic-accent shadow-lg' : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'}`}
                    >
                        API & Proxy
                    </button>
                </div>
            </div>
            
            <div className="w-48 hidden md:block"></div> {/* Spacer for balance */}
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 md:p-10 w-full space-y-10 pb-24">
            
            {activeTab === 'general' ? (
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-10">
                    <div className="space-y-10">
                        {/* AI Model Selection */}
                        <section className="space-y-4">
                            <h3 className="text-lg font-bold text-slate-900 dark:text-slate-200 pb-2 border-b border-slate-200 dark:border-slate-800">
                                Mô Hình AI Gemini (Dùng cho API Key cá nhân)
                            </h3>
                            <div className="space-y-2">
                                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Chọn Model</label>
                                <select 
                                    value={settings.aiModel}
                                    onChange={(e) => handleChange('aiModel', e.target.value)}
                                    className="w-full bg-stone-100 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 rounded p-3 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none transition-colors"
                                >
                                    <option value="gemini-3.1-pro-preview">Gemini 3.1 Pro</option>
                                    <option value="gemini-3-flash-preview">Gemini 3 Flash (Mặc định)</option>
                                    <option value="gemini-2.5-pro-preview">Gemini 2.5 Pro</option>
                                    <option value="gemini-2.5-flash-preview">Gemini 2.5 Flash</option>
                                </select>
                                <p className="text-[10px] text-slate-500 italic">
                                    {settings.aiModel.includes('pro') ? 'Model Pro: Tư duy sâu, phù hợp cốt truyện phức tạp.' : 'Model Flash: Tốc độ nhanh, phản hồi tức thì.'}
                                </p>
                            </div>
                        </section>

                        {/* Full Screen Mode */}
                        <section className="space-y-4">
                            <h3 className="text-lg font-bold text-slate-900 dark:text-slate-200 pb-2 border-b border-slate-200 dark:border-slate-800">
                                Full màn hình
                            </h3>
                            <div className="flex items-center gap-3 p-2 bg-slate-100 dark:bg-slate-800/30 rounded border border-slate-200 dark:border-slate-700 transition-colors">
                                <button
                                    onClick={() => {
                                        handleChange('fullScreenMode', !settings.fullScreenMode);
                                        if (!settings.fullScreenMode) {
                                            document.documentElement.requestFullscreen().catch(() => {});
                                        } else {
                                            if (document.fullscreenElement) {
                                                document.exitFullscreen().catch(() => {});
                                            }
                                        }
                                    }}
                                    className={`w-12 h-6 rounded-full p-1 transition-colors flex items-center ${settings.fullScreenMode ? 'bg-mystic-accent justify-end' : 'bg-stone-400 dark:bg-slate-700 justify-start'}`}
                                >
                                    <div className="w-4 h-4 bg-white rounded-full shadow-md" />
                                </button>
                                <span className="text-xs text-slate-500 dark:text-slate-400">{settings.fullScreenMode ? 'Đang bật' : 'Đang tắt'}</span>
                            </div>
                        </section>
                    </div>

                        <div className="space-y-10">
                            {/* System & Display Settings */}
                            <section className="space-y-6">
                                <h3 className="text-lg font-bold text-slate-900 dark:text-slate-200 pb-2 border-b border-slate-200 dark:border-slate-800">
                                    Hệ Thống & Hiển Thị
                                </h3>
                                
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
                                    {/* System Font */}
                                    <div className="space-y-2">
                                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Phông chữ hệ thống</label>
                                        <select 
                                            value={settings.systemFont}
                                            onChange={(e) => handleChange('systemFont', e.target.value)}
                                            className="w-full bg-stone-100 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none transition-colors"
                                        >
                                            <option value="Inter">Inter</option>
                                            <option value="Roboto">Roboto</option>
                                            <option value="Open Sans">Open Sans</option>
                                            <option value="Montserrat">Montserrat</option>
                                            <option value="Oswald">Oswald</option>
                                            <option value="Playfair Display">Playfair Display</option>
                                            <option value="Lora">Lora</option>
                                            <option value="Noto Sans Vietnamese">Noto Sans Vietnamese</option>
                                            <option value="Be Vietnam Pro">Be Vietnam Pro</option>
                                            <option value="JetBrains Mono">JetBrains Mono</option>
                                        </select>
                                    </div>

                                    {/* Font Size */}
                                    <div className="space-y-2">
                                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Cỡ chữ hệ thống (px)</label>
                                        <input 
                                            type="number"
                                            value={localFontSize}
                                            onChange={(e) => {
                                                const val = e.target.value;
                                                setLocalFontSize(val);
                                                const num = parseInt(val);
                                                if (!isNaN(num) && num >= 10 && num <= 40) {
                                                    handleChange('fontSize', num);
                                                }
                                            }}
                                            onBlur={() => {
                                                if (localFontSize === '' || isNaN(parseInt(localFontSize))) {
                                                    setLocalFontSize(settings.fontSize.toString());
                                                } else {
                                                    const num = parseInt(localFontSize);
                                                    if (num < 10) {
                                                        handleChange('fontSize', 10);
                                                        setLocalFontSize('10');
                                                    } else if (num > 40) {
                                                        handleChange('fontSize', 40);
                                                        setLocalFontSize('40');
                                                    }
                                                }
                                            }}
                                            className="w-full bg-slate-50 dark:bg-mystic-900 border border-slate-300 dark:border-slate-700 rounded p-2 text-sm text-slate-900 dark:text-slate-200 focus:border-mystic-accent outline-none transition-colors"
                                            min="10"
                                            max="40"
                                            placeholder="16"
                                        />
                                    </div>

                                    {/* Reality Difficulty */}
                                    <div className="space-y-2">
                                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Độ Khó Thực Tại</label>
                                        <select 
                                            value={settings.realityDifficulty}
                                            onChange={(e) => handleChange('realityDifficulty', e.target.value)}
                                            className="w-full bg-stone-100 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none transition-colors"
                                        >
                                            <option value="Easy">Dễ (Hỗ trợ nhiều)</option>
                                            <option value="Normal">Bình thường</option>
                                            <option value="Hard">Khó (Khắc nghiệt)</option>
                                            <option value="Nightmare">Ác mộng</option>
                                        </select>
                                    </div>

                                    {/* Theme (Light Mode Toggle) */}
                                    <div className="space-y-2">
                                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Chế độ Nền sáng</label>
                                        <div className="flex items-center gap-3 p-2 bg-slate-100 dark:bg-slate-800/30 rounded border border-slate-200 dark:border-slate-700 transition-colors">
                                            <button
                                                onClick={() => handleChange('theme', settings.theme === 'light' ? 'dark' : 'light')}
                                                className={`w-12 h-6 rounded-full p-1 transition-colors flex items-center ${settings.theme === 'light' ? 'bg-mystic-accent justify-end' : 'bg-slate-300 dark:bg-slate-700 justify-start'}`}
                                            >
                                                <div className="w-4 h-4 bg-white rounded-full shadow-md" />
                                            </button>
                                            <span className="text-xs text-slate-500 dark:text-slate-400">{settings.theme === 'light' ? 'Đang bật' : 'Đang tắt'}</span>
                                        </div>
                                    </div>

                                    {/* Content Beautify */}
                                    <div className="space-y-2">
                                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Làm đẹp cho nội dung</label>
                                        <div className="flex items-center gap-3 p-2 bg-slate-100 dark:bg-slate-800/30 rounded border border-slate-200 dark:border-slate-700 transition-colors">
                                            <button
                                                onClick={() => handleChange('contentBeautify', !settings.contentBeautify)}
                                                className={`w-12 h-6 rounded-full p-1 transition-colors flex items-center ${settings.contentBeautify ? 'bg-mystic-accent justify-end' : 'bg-stone-400 dark:bg-slate-700 justify-start'}`}
                                            >
                                                <div className="w-4 h-4 bg-white rounded-full shadow-md" />
                                            </button>
                                            <span className="text-xs text-slate-500 dark:text-slate-400">{settings.contentBeautify ? 'Đang bật' : 'Đang tắt'}</span>
                                        </div>
                                    </div>

                                    {/* Visual Effects */}
                                    <div className="space-y-2">
                                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Hiệu ứng Hình ảnh</label>
                                        <div className="flex items-center gap-3 p-2 bg-slate-100 dark:bg-slate-800/30 rounded border border-slate-200 dark:border-slate-700 transition-colors">
                                            <button
                                                onClick={() => handleChange('visualEffects', !settings.visualEffects)}
                                                className={`w-12 h-6 rounded-full p-1 transition-colors flex items-center ${settings.visualEffects ? 'bg-mystic-accent justify-end' : 'bg-stone-400 dark:bg-slate-700 justify-start'}`}
                                            >
                                                <div className="w-4 h-4 bg-white rounded-full shadow-md" />
                                            </button>
                                            <span className="text-xs text-slate-500 dark:text-slate-400">{settings.visualEffects ? 'Đang bật' : 'Đang tắt'}</span>
                                        </div>
                                    </div>
                                </div>
                            </section>

                            {/* Game Configuration Section */}
                            <section className="space-y-6">
                                <h3 className="text-lg font-bold text-slate-900 dark:text-slate-200 pb-2 border-b border-slate-200 dark:border-slate-800">
                                    Cấu Hình Trò Chơi
                                </h3>
                                
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
                                    {/* Narrative Perspective */}
                                    <div className="space-y-2">
                                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                                            Góc nhìn kể chuyện (POV)
                                        </label>
                                        <select 
                                            value={settings.perspective}
                                            onChange={(e) => handleChange('perspective', e.target.value as NarrativePerspective)}
                                            className="w-full bg-stone-100 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none transition-colors"
                                        >
                                            <option value="third">Ngôi thứ 3 (Anh ấy/Cô ấy/Tên)</option>
                                            <option value="first">Ngôi thứ 1 (Tôi)</option>
                                            <option value="second">Ngôi thứ 2 (Bạn/Ngươi)</option>
                                        </select>
                                    </div>

                                    {/* Difficulty */}
                                    <div className="space-y-2">
                                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                                            Độ khó cốt truyện
                                        </label>
                                        <select 
                                            value={settings.difficulty.id}
                                            onChange={(e) => {
                                                const diff = DIFFICULTY_LEVELS.find(d => d.id === e.target.value);
                                                if (diff) handleChange('difficulty', diff);
                                            }}
                                            className="w-full bg-stone-100 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none transition-colors"
                                        >
                                            {DIFFICULTY_LEVELS.map(d => (
                                                <option key={d.id} value={d.id}>{d.label}</option>
                                            ))}
                                        </select>
                                    </div>

                                    {/* Output Length */}
                                    <div className="space-y-2">
                                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                                            Độ dài phản hồi
                                        </label>
                                        <select 
                                            value={settings.outputLength.id}
                                            onChange={(e) => {
                                                const len = OUTPUT_LENGTHS.find(o => o.id === e.target.value);
                                                if (len) handleChange('outputLength', len);
                                            }}
                                            className="w-full bg-stone-100 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none transition-colors"
                                        >
                                            {OUTPUT_LENGTHS.map(o => (
                                                <option key={o.id} value={o.id}>{o.label}</option>
                                            ))}
                                        </select>
                                    </div>

                                    {/* Custom Word Count */}
                                    {settings.outputLength.id === 'custom' && (
                                        <div className="col-span-1 md:col-span-2 grid grid-cols-2 gap-4 animate-in fade-in slide-in-from-top-2 duration-300">
                                            <div className="space-y-2">
                                                <label className="block text-xs font-medium text-slate-500 dark:text-slate-400">Tối thiểu (Min words)</label>
                                                <input 
                                                    type="number"
                                                    value={settings.customMinWords}
                                                    onChange={(e) => handleChange('customMinWords', parseInt(e.target.value))}
                                                    className="w-full bg-stone-100 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none"
                                                />
                                            </div>
                                            <div className="space-y-2">
                                                <label className="block text-xs font-medium text-slate-500 dark:text-slate-400">Tối đa (Max words)</label>
                                                <input 
                                                    type="number"
                                                    value={settings.customMaxWords}
                                                    onChange={(e) => handleChange('customMaxWords', parseInt(e.target.value))}
                                                    className="w-full bg-stone-100 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none"
                                                />
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </section>

                            {/* Safety */}
                            <section className="space-y-4">
                                <SafetySettings 
                                    settings={settings}
                                    onUpdate={handleGlobalUpdate}
                                />
                            </section>

                            {/* Advanced Generation Params */}
                            <section className="space-y-6">
                                <h3 className="text-lg font-bold text-slate-900 dark:text-slate-200 pb-2 border-b border-slate-200 dark:border-slate-800">
                                    Tham số Sinh (Generation)
                                </h3>
                                
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
                                    {/* Temperature */}
                                    <div className="space-y-2">
                                        <div className="flex justify-between text-sm items-center">
                                            <label className="font-medium text-slate-700 dark:text-slate-300">Temperature</label>
                                            <input 
                                                type="number" 
                                                min="0" max="2" step="0.01" 
                                                value={settings.temperature}
                                                onChange={(e) => handleChange('temperature', parseFloat(e.target.value))}
                                                className="bg-stone-200 dark:bg-slate-900 border border-stone-400 dark:border-slate-700 rounded px-2 py-1 text-xs w-16 text-center text-mystic-accent outline-none focus:border-mystic-accent transition-colors"
                                            />
                                        </div>
                                        <input 
                                            type="range" min="0" max="2" step="0.01" 
                                            value={settings.temperature}
                                            onChange={(e) => handleChange('temperature', parseFloat(e.target.value))}
                                            className="w-full accent-mystic-accent bg-slate-300 dark:bg-slate-700 h-1 rounded-lg appearance-none cursor-pointer"
                                        />
                                    </div>

                                    {/* Thinking Budget */}
                                    {settings.aiModel.includes('pro') && !settings.aiModel.includes('gemini-3') && (
                                        <div className="space-y-2 animate-in fade-in duration-300">
                                            <div className="flex justify-between text-sm">
                                                <label className="font-medium text-slate-700 dark:text-slate-300">
                                                    Thinking Budget
                                                </label>
                                                <span className="text-purple-600 dark:text-purple-400 font-mono uppercase text-xs border border-purple-200 dark:border-purple-900/50 bg-purple-100 dark:bg-purple-900/20 px-2 py-0.5 rounded">{settings.thinkingBudgetLevel}</span>
                                            </div>
                                            <select 
                                                value={settings.thinkingBudgetLevel}
                                                onChange={(e) => {
                                                    handleMultipleChanges({
                                                        thinkingBudgetLevel: e.target.value as ThinkingBudgetLevel,
                                                        thinkingMode: 'budget'
                                                    });
                                                }}
                                                className="w-full bg-stone-100 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none transition-colors"
                                            >
                                                <option value="auto">Auto (0 tokens)</option>
                                                <option value="low">Low (4,096 tokens)</option>
                                                <option value="medium">Medium (16,384 tokens)</option>
                                                <option value="high">High (32,768 tokens)</option>
                                            </select>
                                        </div>
                                    )}

                                    {/* Thinking Level (Gemini 3 Only) */}
                                    {settings.aiModel.includes('gemini-3') && (
                                        <div className="space-y-2 animate-in fade-in duration-300">
                                            <div className="flex justify-between text-sm">
                                                <label className="font-medium text-slate-700 dark:text-slate-300">
                                                    Thinking Level
                                                </label>
                                                <span className="text-emerald-600 dark:text-emerald-400 font-mono uppercase text-xs border border-emerald-200 dark:border-emerald-900/50 bg-emerald-100 dark:bg-emerald-900/20 px-2 py-0.5 rounded">{settings.thinkingLevel}</span>
                                            </div>
                                            <select 
                                                value={settings.thinkingLevel}
                                                onChange={(e) => {
                                                    handleMultipleChanges({
                                                        thinkingLevel: e.target.value as ThinkingLevel,
                                                        thinkingMode: 'level'
                                                    });
                                                }}
                                                className="w-full bg-stone-100 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none transition-colors"
                                            >
                                                <option value="OFF">OFF (Tắt tư duy)</option>
                                                <option value="LOW">LOW (Tư duy cơ bản)</option>
                                                <option value="MEDIUM">MEDIUM (Tư duy trung bình)</option>
                                                <option value="HIGH">HIGH (Tư duy tối đa)</option>
                                            </select>
                                        </div>
                                    )}

                                    {/* Top K */}
                                    <div className="space-y-2">
                                        <div className="flex justify-between text-sm items-center">
                                            <label className="font-medium text-slate-700 dark:text-slate-300">Top K</label>
                                            <input 
                                                type="number" 
                                                min="1" max="500" step="1" 
                                                value={settings.topK}
                                                onChange={(e) => handleChange('topK', parseInt(e.target.value))}
                                                className="bg-stone-200 dark:bg-slate-900 border border-stone-400 dark:border-slate-700 rounded px-2 py-1 text-xs w-16 text-center text-mystic-accent outline-none focus:border-mystic-accent transition-colors"
                                            />
                                        </div>
                                        <input 
                                            type="range" min="1" max="500" step="1" 
                                            value={settings.topK}
                                            onChange={(e) => handleChange('topK', parseInt(e.target.value))}
                                            className="w-full accent-mystic-accent bg-slate-300 dark:bg-slate-700 h-1 rounded-lg appearance-none cursor-pointer"
                                        />
                                    </div>

                                    {/* Top P */}
                                    <div className="space-y-2">
                                        <div className="flex justify-between text-sm items-center">
                                            <label className="font-medium text-slate-700 dark:text-slate-300">Top P</label>
                                            <input 
                                                type="number" 
                                                min="0" max="1" step="0.01" 
                                                value={settings.topP}
                                                onChange={(e) => handleChange('topP', parseFloat(e.target.value))}
                                                className="bg-stone-200 dark:bg-slate-900 border border-stone-400 dark:border-slate-700 rounded px-2 py-1 text-xs w-16 text-center text-mystic-accent outline-none focus:border-mystic-accent transition-colors"
                                            />
                                        </div>
                                        <input 
                                            type="range" min="0" max="1" step="0.01" 
                                            value={settings.topP}
                                            onChange={(e) => handleChange('topP', parseFloat(e.target.value))}
                                            className="w-full accent-mystic-accent bg-slate-300 dark:bg-slate-700 h-1 rounded-lg appearance-none cursor-pointer"
                                        />
                                    </div>

                                    {/* Context Size */}
                                    <div className="space-y-2">
                                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Context Size</label>
                                        <input 
                                            type="number" 
                                            min="1000"
                                            max="2000000"
                                            value={settings.contextSize}
                                            onChange={(e) => handleChange('contextSize', parseInt(e.target.value))}
                                            className="w-full bg-stone-100 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none font-mono transition-colors"
                                        />
                                    </div>

                                    {/* Max Output */}
                                    <div className="space-y-2">
                                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Max Output</label>
                                        <input 
                                            type="number" 
                                            value={settings.maxOutputTokens}
                                            onChange={(e) => handleChange('maxOutputTokens', parseInt(e.target.value))}
                                            className="w-full bg-stone-100 dark:bg-slate-800 border border-stone-400 dark:border-slate-600 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none font-mono transition-colors"
                                        />
                                    </div>
                                </div>
                            </section>
                        </div>
                </div>
            ) : (
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-10 animate-in fade-in slide-in-from-bottom-4 duration-300">
                    {/* Column 1: Gemini API Key */}
                    <div className="space-y-8">
                        <section className="space-y-4">
                            <div className="flex justify-between items-center pb-2 border-b border-slate-200 dark:border-slate-800">
                                <div className="flex items-center">
                                    <h3 className="text-lg font-bold text-slate-900 dark:text-slate-200">
                                        Gemini API Key (Cá nhân)
                                    </h3>
                                    <button 
                                        onClick={() => handleChange('useGeminiApi', !settings.useGeminiApi)}
                                        className={`w-10 h-5 rounded-full transition-colors relative ${settings.useGeminiApi ? 'bg-mystic-accent' : 'bg-slate-400'}`}
                                        title={settings.useGeminiApi ? "Đang bật" : "Đang tắt"}
                                    >
                                        <div className={`absolute top-1 w-3 h-3 rounded-full bg-white transition-all ${settings.useGeminiApi ? 'left-6' : 'left-1'}`} />
                                    </button>
                                </div>
                                <div className="flex gap-2">
                                    <button 
                                        onClick={handleResetApiTab}
                                        className="text-xs text-red-500 dark:text-red-400 hover:text-red-600 dark:hover:text-red-300 px-2 py-1 rounded hover:bg-red-400/10 transition-colors"
                                    >
                                        Reset tab API
                                    </button>
                                    <label className={`text-xs text-mystic-accent hover:text-mystic-accent/80 font-medium cursor-pointer px-2 py-1 rounded hover:bg-mystic-accent/10 transition-colors ${!settings.useGeminiApi ? 'opacity-50 cursor-not-allowed' : ''}`}>
                                        Nhập txt
                                        <input 
                                            type="file" 
                                            accept=".txt,.json" 
                                            className="hidden" 
                                            onChange={handleImportTxt}
                                            disabled={!settings.useGeminiApi}
                                        />
                                    </label>
                                </div>
                            </div>
                            <div className={`space-y-4 bg-stone-100 dark:bg-slate-800/50 p-6 rounded-lg border border-stone-300 dark:border-slate-700 transition-all ${!settings.useGeminiApi ? 'opacity-50 grayscale pointer-events-none' : ''}`}>
                                <div className="space-y-4">
                                    <div className="space-y-2">
                                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Thêm API Key mới</label>
                                        <div className="flex flex-col gap-2">
                                            <textarea 
                                                placeholder="Dán API Key vào đây (mỗi dòng 1 key)..."
                                                className="w-full bg-stone-200 dark:bg-slate-900 border border-stone-400 dark:border-slate-700 rounded p-3 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none font-mono min-h-[80px] transition-colors"
                                                disabled={!settings.useGeminiApi}
                                                onKeyDown={(e) => {
                                                    if (e.key === 'Enter' && e.ctrlKey) {
                                                        const target = e.target as HTMLTextAreaElement;
                                                        const newKeys = target.value.split('\n').map(k => k.trim()).filter(k => k !== '');
                                                        if (newKeys.length > 0) {
                                                            const currentKeys = settings.geminiApiKey || [];
                                                            const updatedKeys = [...currentKeys];
                                                            newKeys.forEach(nk => {
                                                                if (!updatedKeys.includes(nk)) updatedKeys.push(nk);
                                                            });
                                                            handleChange('geminiApiKey', updatedKeys);
                                                            target.value = '';
                                                        }
                                                    }
                                                }}
                                                onBlur={(e) => {
                                                    const target = e.target as HTMLTextAreaElement;
                                                    const newKeys = target.value.split('\n').map(k => k.trim()).filter(k => k !== '');
                                                    if (newKeys.length > 0) {
                                                        const currentKeys = settings.geminiApiKey || [];
                                                        const updatedKeys = [...currentKeys];
                                                        newKeys.forEach(nk => {
                                                            if (!updatedKeys.includes(nk)) updatedKeys.push(nk);
                                                        });
                                                        handleChange('geminiApiKey', updatedKeys);
                                                        target.value = '';
                                                    }
                                                }}
                                            />
                                            <p className="text-[10px] text-slate-500 italic">Mẹo: Nhấn Ctrl + Enter hoặc click ra ngoài để thêm nhanh.</p>
                                        </div>
                                    </div>

                                    <div className="space-y-2">
                                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Danh sách Key ({settings.geminiApiKey?.length || 0})</label>
                                        <div className="max-h-[200px] overflow-y-auto space-y-2 pr-2 custom-scrollbar">
                                            {settings.geminiApiKey && settings.geminiApiKey.length > 0 ? (
                                                settings.geminiApiKey.map((key, index) => (
                                                    <div key={index} className="flex items-center justify-between bg-stone-200 dark:bg-slate-900 border border-stone-400 dark:border-slate-700 rounded p-2 group transition-colors">
                                                        <div className="flex items-center overflow-hidden">
                                                            <span className="flex-shrink-0 w-5 h-5 flex items-center justify-center bg-mystic-accent/10 text-mystic-accent text-[10px] font-bold rounded-full border border-mystic-accent/20">
                                                                {index + 1}
                                                            </span>
                                                            <span className="text-xs font-mono text-slate-500 dark:text-slate-400 truncate">
                                                                {key.substring(0, 8)}...{key.substring(key.length - 4)}
                                                            </span>
                                                        </div>
                                                        <button 
                                                            onClick={() => {
                                                                const updated = settings.geminiApiKey?.filter((_, i) => i !== index);
                                                                handleChange('geminiApiKey', updated || []);
                                                            }}
                                                            className="text-slate-400 hover:text-red-500 transition-colors p-1"
                                                            disabled={!settings.useGeminiApi}
                                                        >
                                                            Xóa
                                                        </button>
                                                    </div>
                                                ))
                                            ) : (
                                                <div className="text-xs text-slate-500 italic p-4 text-center border border-dashed border-slate-300 dark:border-slate-700 rounded">
                                                    Chưa có API Key nào được thêm.
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                    
                                    <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed pt-2 border-t border-slate-200 dark:border-slate-700/50">
                                        Hệ thống sẽ tự động luân phiên (rotate) các Key này theo thứ tự từ 1 đến hết để tối ưu hóa giới hạn gọi API.
                                    </p>
                                </div>
                            </div>
                        </section>

                        {/* Security Notice Removed */}
                    </div>

                    {/* Column 2: Reverse Proxy */}
                    <div className="space-y-8">
                        <section className="space-y-4">
                            <div className="flex justify-between items-center pb-2 border-b border-slate-200 dark:border-slate-800">
                                <div className="flex items-center">
                                    <h3 className="text-lg font-bold text-slate-900 dark:text-slate-200">
                                        Reverse Proxy
                                    </h3>
                                    <button 
                                        onClick={() => handleChange('useProxy', !settings.useProxy)}
                                        className={`w-10 h-5 rounded-full transition-colors relative ${settings.useProxy ? 'bg-mystic-accent' : 'bg-slate-400'}`}
                                        title={settings.useProxy ? "Đang bật" : "Đang tắt"}
                                    >
                                        <div className={`absolute top-1 w-3 h-3 rounded-full bg-white transition-all ${settings.useProxy ? 'left-6' : 'left-1'}`} />
                                    </button>
                                </div>
                                <Button 
                                    variant="ghost"
                                    className="text-[10px] h-7 px-3 border-mystic-accent/30 text-mystic-accent hover:bg-mystic-accent/10"
                                    onClick={handleLoadModels}
                                    disabled={isSaving || !settings.useProxy}
                                >
                                    {isSaving ? 'Đang tải...' : 'Load All Models'}
                                </Button>
                            </div>
                            
                            <div className={`space-y-6 transition-all ${!settings.useProxy ? 'opacity-50 grayscale pointer-events-none' : ''}`}>
                                {/* Proxy 1 Table */}
                                <div className="space-y-6 bg-stone-100 dark:bg-slate-800/50 p-6 rounded-lg border border-stone-300 dark:border-slate-700 transition-colors">
                                    <div className="flex justify-between items-center">
                                        <h4 className="text-xs font-bold text-mystic-accent uppercase tracking-widest">Proxy 1</h4>
                                        {settings.proxyUrl ? (
                                            <span className="px-2 py-0.5 rounded bg-sky-500/10 text-sky-500 text-[10px] font-bold border border-sky-500/20">
                                                ACTIVE
                                            </span>
                                        ) : (
                                            <span className="px-2 py-0.5 rounded bg-slate-500/10 text-slate-500 text-[10px] font-bold border border-slate-500/20">
                                                OFF
                                            </span>
                                        )}
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        <div className="space-y-2">
                                            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300">Tên Proxy (Tùy chọn)</label>
                                            <input 
                                                type="text" 
                                                placeholder="Ví dụ: Proxy 1"
                                                value={settings.proxyName || ''}
                                                onChange={(e) => handleChange('proxyName', e.target.value)}
                                                className="w-full bg-stone-200 dark:bg-slate-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none transition-colors"
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300">URL Proxy</label>
                                            <input 
                                                type="text" 
                                                placeholder="https://proxy.example.com/v1beta"
                                                value={settings.proxyUrl || ''}
                                                onChange={(e) => handleChange('proxyUrl', e.target.value)}
                                                className="w-full bg-stone-200 dark:bg-slate-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none font-mono transition-colors"
                                            />
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        <div className="space-y-2">
                                            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300">Password / Key</label>
                                            <input 
                                                type="password" 
                                                placeholder="Nhập Key hoặc Password..."
                                                value={settings.proxyKey || ''}
                                                onChange={(e) => handleChange('proxyKey', e.target.value)}
                                                className="w-full bg-stone-200 dark:bg-slate-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none font-mono transition-colors"
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300">Chọn Model</label>
                                            <select 
                                                value={settings.proxyModel || ''}
                                                onChange={(e) => handleChange('proxyModel', e.target.value)}
                                                className="w-full bg-stone-200 dark:bg-slate-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none font-mono transition-colors"
                                            >
                                                <option value="">-- Chọn Model --</option>
                                                {settings.proxyModels?.map(m => (
                                                    <option key={m} value={m}>{m}</option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>

                                    <div className="pt-2">
                                        <Button 
                                            variant="ghost"
                                            className="w-full border-mystic-accent/30 text-mystic-accent hover:bg-mystic-accent/10 py-2 text-xs"
                                            onClick={async () => {
                                                if (!settings.proxyUrl) return;
                                                setIsSaving(true);
                                                try {
                                                    const res = await fetch(`${settings.proxyUrl}/models`, {
                                                        method: 'GET',
                                                        headers: {
                                                            'Authorization': settings.proxyKey ? `Bearer ${settings.proxyKey}` : '',
                                                            'x-goog-api-key': settings.proxyKey || ''
                                                        }
                                                    });
                                                    if (res.ok) {
                                                        handleLoadModels();
                                                    } else {
                                                        const text = await res.text().catch(() => "");
                                                        console.error(`Lỗi Proxy 1 (${res.status}): ${text || "Không xác định"}`);
                                                    }
                                                } catch (err: unknown) {
                                                    const message = err instanceof Error ? err.message : String(err);
                                                    console.error(`Lỗi Proxy 1: ${message}`);
                                                } finally {
                                                    setIsSaving(false);
                                                }
                                            }}
                                        >
                                            Test Connection Proxy 1
                                        </Button>
                                    </div>
                                </div>

                                {/* Proxy 2 Table */}
                                <div className="space-y-6 bg-stone-100 dark:bg-slate-800/50 p-6 rounded-lg border border-stone-300 dark:border-slate-700 transition-colors">
                                    <div className="flex justify-between items-center">
                                        <h4 className="text-xs font-bold text-mystic-accent uppercase tracking-widest">Proxy 2</h4>
                                        {settings.proxyUrl2 ? (
                                            <span className="px-2 py-0.5 rounded bg-sky-500/10 text-sky-500 text-[10px] font-bold border border-sky-500/20">
                                                ACTIVE
                                            </span>
                                        ) : (
                                            <span className="px-2 py-0.5 rounded bg-slate-500/10 text-slate-500 text-[10px] font-bold border border-slate-500/20">
                                                OFF
                                            </span>
                                        )}
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        <div className="space-y-2">
                                            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300">Tên Proxy (Tùy chọn)</label>
                                            <input 
                                                type="text" 
                                                placeholder="Ví dụ: Proxy 2"
                                                value={settings.proxyName2 || ''}
                                                onChange={(e) => handleChange('proxyName2', e.target.value)}
                                                className="w-full bg-stone-200 dark:bg-slate-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none transition-colors"
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300">URL Proxy</label>
                                            <input 
                                                type="text" 
                                                placeholder="https://proxy2.example.com/v1beta"
                                                value={settings.proxyUrl2 || ''}
                                                onChange={(e) => handleChange('proxyUrl2', e.target.value)}
                                                className="w-full bg-stone-200 dark:bg-slate-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none font-mono transition-colors"
                                            />
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        <div className="space-y-2">
                                            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300">Password / Key</label>
                                            <input 
                                                type="password" 
                                                placeholder="Nhập Key hoặc Password..."
                                                value={settings.proxyKey2 || ''}
                                                onChange={(e) => handleChange('proxyKey2', e.target.value)}
                                                className="w-full bg-stone-200 dark:bg-slate-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none font-mono transition-colors"
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300">Chọn Model</label>
                                            <select 
                                                value={settings.proxyModel2 || ''}
                                                onChange={(e) => handleChange('proxyModel2', e.target.value)}
                                                className="w-full bg-stone-200 dark:bg-slate-900 border border-stone-400 dark:border-slate-700 rounded p-2 text-sm text-stone-900 dark:text-slate-200 focus:border-mystic-accent outline-none font-mono transition-colors"
                                            >
                                                <option value="">-- Chọn Model --</option>
                                                {settings.proxyModels2?.map(m => (
                                                    <option key={m} value={m}>{m}</option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>

                                    <div className="pt-2">
                                        <Button 
                                            variant="ghost"
                                            className="w-full border-mystic-accent/30 text-mystic-accent hover:bg-mystic-accent/10 py-2 text-xs"
                                            onClick={async () => {
                                                if (!settings.proxyUrl2) return;
                                                setIsSaving(true);
                                                try {
                                                    const res = await fetch(`${settings.proxyUrl2}/models`, {
                                                        method: 'GET',
                                                        headers: {
                                                            'Authorization': settings.proxyKey2 ? `Bearer ${settings.proxyKey2}` : '',
                                                            'x-goog-api-key': settings.proxyKey2 || ''
                                                        }
                                                    });
                                                    if (res.ok) {
                                                        handleLoadModels();
                                                    } else {
                                                        const text = await res.text().catch(() => "");
                                                        console.error(`Lỗi Proxy 2 (${res.status}): ${text || "Không xác định"}`);
                                                    }
                                                } catch (err: unknown) {
                                                    const message = err instanceof Error ? err.message : String(err);
                                                    console.error(`Lỗi Proxy 2: ${message}`);
                                                } finally {
                                                    setIsSaving(false);
                                                }
                                            }}
                                        >
                                            Test Connection Proxy 2
                                        </Button>
                                    </div>
                                </div>
                            </div>
                            <p className="text-xs text-slate-500 italic leading-relaxed">
                                Khi Proxy được cấu hình, hệ thống sẽ ưu tiên sử dụng tài nguyên từ các Proxy này. Nút "Load All Models" sẽ tải danh sách model từ cả hai Proxy.
                            </p>
                        </section>
                    </div>
                </div>
            )}

            <div className="h-10"></div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-stone-400 dark:border-slate-800 bg-stone-200/95 dark:bg-mystic-900/95 backdrop-blur flex justify-center items-center gap-6 shrink-0 transition-colors">
            <Button 
                variant="ghost" 
                onClick={handleResetFactory}
                className="text-red-600 dark:text-red-500 hover:text-white border-red-500/50 hover:bg-red-600 font-bold px-6 py-3 transition-all shadow-lg shadow-red-900/10 dark:shadow-red-900/20"
            >
                Khôi phục cài đặt gốc
            </Button>

            <div className="flex flex-col items-center gap-1">
                <Button 
                    variant="primary" 
                    onClick={handleSave}
                    disabled={isSaving}
                    className="px-10 flex items-center gap-2 shadow-lg shadow-mystic-accent/20"
                >
                    {isSaving ? 'Đang lưu...' : (fromGame ? 'Lưu & Quay Lại' : 'Lưu & Về Sảnh')}
                </Button>
                <p className="text-[10px] text-slate-500 dark:text-slate-500 italic">
                    {fromGame ? 'Lưu cấu hình và tiếp tục cuộc hành trình' : 'Lưu cấu hình và quay lại màn hình chính'}
                </p>
            </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsScreen;
