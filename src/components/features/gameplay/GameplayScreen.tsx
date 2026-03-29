
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Menu, Shield, Zap, User, Save, BrainCircuit, Globe, Brain, Loader2, ChevronLeft, ChevronRight, ChevronDown, ChevronsDown, ChevronsUp, BookOpen, Trash2, Edit2, Check, X, Database, ToggleLeft, ToggleRight, RefreshCw, Settings, Clock, Copy, Maximize2, Minimize2, MapPin, Box, History, LogOut, Image as ImageIcon } from 'lucide-react';
import { NavigationProps, GameState, ChatMessage, AppSettings, SaveFile, WorldData, TawaPresetConfig, GameTime, Entity, ImageMetadata } from '../../../types';
import { gameplayAiService } from '../../../services/ai/gameplay/service';
import { dbService } from '../../../services/db/indexedDB';
import Button from '../../ui/Button';
import { TAWA_REGEX, extractTagContent, cleanRawText, parseChoices } from '../../../utils/regex';
import { DEFAULT_PRESET_CONFIG } from '../../../constants/tawa_modules'; 
import TawaPresetManager from './components/TawaPresetManager';
import ImageLibraryModal from './components/ImageLibraryModal';
import { LsrParser, LsrTableDefinition } from '../../../services/lsr/LsrParser';
import { LSR_REGEX } from '../../../data/lsr_config';
import { vectorService } from '../../../services/ai/vectorService';
import { INITIAL_GAME_TIME, formatGameTime, advanceTime } from '../../../utils/timeUtils';
import { useResponsive } from '../../../hooks/useResponsive';
import { MarkdownRenderer } from '../../common/MarkdownRenderer';

// Constants for Pagination
const MESSAGES_PER_PAGE = 10;

// Helper component to render Tawa's structured output (UPDATED WITH ENTITY HIGHLIGHTING)
interface TawaMessageRendererProps {
    text: string | { text: string };
    onUpdate: (newText: string) => void;
    isStreaming?: boolean;
    settings?: AppSettings | null;
    entities?: Entity[];
    onEntityClick?: (entityId: string) => void;
    turnNumber?: number;
    userAction?: string;
    playerName?: string;
    playerAvatar?: string;
}

const TawaMessageRenderer: React.FC<TawaMessageRendererProps> = ({ 
    text, 
    onUpdate, 
    isStreaming, 
    settings, 
    entities, 
    onEntityClick,
    turnNumber,
    userAction,
    playerName,
    playerAvatar
}) => {
    // Safety check: ensure text is a string to prevent React Error #31
    const displayText = typeof text === 'string' ? text : (text && typeof text === 'object' && 'text' in text ? (text as { text: string }).text : JSON.stringify(text));
    
    const [showThinking, setShowThinking] = useState(false);
    const [isEditing, setIsEditing] = useState(false);
    const [editedText, setEditedText] = useState(displayText);
    const containerRef = useRef<HTMLDivElement>(null);

    const contentBeautify = settings?.contentBeautify ?? false;

    // Handle clicks on entities
    useEffect(() => {
        const handleClick = (e: MouseEvent) => {
            const target = e.target as HTMLElement;
            const entityId = target.getAttribute('data-entity-id');
            if (entityId && onEntityClick) {
                onEntityClick(entityId);
            }
        };

        const el = containerRef.current;
        if (el) {
            el.addEventListener('click', handleClick);
        }
        return () => {
            if (el) {
                el.removeEventListener('click', handleClick);
            }
        };
    }, [onEntityClick]);

    // Sync editedText when prop text changes
    useEffect(() => {
        setEditedText(displayText);
    }, [displayText]);

    // Task 2: Simple Render for Streaming to prevent Markdown breakages
    if (isStreaming) {
        // Làm sạch văn bản ngay cả khi đang streaming để tránh rò rỉ thẻ hệ thống
        const cleanedStreamingText = cleanRawText(displayText);
        
        return (
            <div className="w-full flex flex-col gap-2">
                 {turnNumber !== undefined && (
                    <div className="mb-2 pb-2 border-b border-stone-400/30 dark:border-slate-800/50">
                        <div className="text-[10px] font-black uppercase tracking-[0.2em] text-mystic-accent mb-1 opacity-80 flex items-center gap-2">
                            {turnNumber === 0 ? 'Khởi Đầu (Lượt 0)' : `Lượt ${turnNumber}`}
                            <Loader2 size={10} className="animate-spin text-mystic-accent"/>
                        </div>
                        {userAction && (
                            <div className="text-sm font-serif italic text-stone-500 dark:text-slate-400 opacity-70">
                                &ldquo;{userAction}&rdquo;
                            </div>
                        )}
                    </div>
                 )}
                 
                 {!turnNumber && turnNumber !== 0 && (
                    <div className="text-xs font-bold text-mystic-accent uppercase mb-1 flex items-center gap-1">
                        <Loader2 size={12} className="animate-spin"/> Streaming...
                    </div>
                 )}

                 {/* Render plain text with whitespace preservation during stream */}
                 <div className="whitespace-pre-wrap font-mono text-base text-stone-700 dark:text-slate-300 leading-relaxed opacity-90 animate-pulse">
                    {cleanedStreamingText || "..."}
                 </div>
            </div>
        );
    }

    // Extract Thinking content
    const thinkingContent = extractTagContent(displayText, 'thinking');
    
    // Get Clean Main Content (Strip all system tags)
    const mainContent = cleanRawText(displayText);

    // --- DIALOGUE & FORMATTING LOGIC ---
    const renderContentBlocks = (rawContent: string) => {
        if (!rawContent) return null;
        
        let content = rawContent;

        // 0. Clean Artifacts (System text, status checks)
        if (TAWA_REGEX.ARTIFACTS_REMOVAL) {
             TAWA_REGEX.ARTIFACTS_REMOVAL.forEach(regex => {
                 if (contentBeautify && regex.source.includes('Hệ thống')) {
                     return; 
                 }
                 content = content.replace(regex, '');
             });
        }

        // Apply LSR Regex cleaning
        LSR_REGEX.forEach(rule => {
            content = content.replace(rule.regex, '');
        });

        // 1. Remove Asterisks (*) completely for novel style
        if (TAWA_REGEX.ASTERISK_REMOVAL) {
            content = content.replace(TAWA_REGEX.ASTERISK_REMOVAL, '');
        }

        // 2. Split into blocks by lines
        const lines = content.split(/\n+/);
        const blocks: React.ReactNode[] = [];

        // Helper to highlight entities in a string
        const highlightEntities = (text: string) => {
            if (!entities || entities.length === 0) return text;
            
            const formatted = text;
            const sortedEntities = [...entities].sort((a, b) => b.name.length - a.name.length);
            const placeholders: Record<string, string> = {};
            let tempFormatted = formatted;

            sortedEntities.forEach((entity, index) => {
                if (!entity.name) return;
                const escapedName = entity.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                const regex = new RegExp('(^|[^a-zA-Z0-9\\u00C0-\\u1EF9])(' + escapedName + ')(?![a-zA-Z0-9\\u00C0-\\u1EF9])', 'g');
                const placeholder = `__ENTITY_REF_${index}__`;
                const colorClass = entity.type === 'NPC' ? 'text-sky-500' : 
                                   entity.type === 'LOCATION' ? 'text-emerald-500' : 
                                   entity.type === 'FACTION' ? 'text-rose-500' :
                                   entity.type === 'ITEM' ? 'text-amber-500' :
                                   'text-stone-500';

                placeholders[placeholder] = `<span class="${colorClass} font-bold cursor-pointer hover:underline decoration-dotted" data-entity-id="${entity.id}">${entity.name}</span>`;
                tempFormatted = tempFormatted.replace(regex, `$1${placeholder}`);
            });

            Object.keys(placeholders).forEach(placeholder => {
                tempFormatted = tempFormatted.replace(new RegExp(placeholder, 'g'), placeholders[placeholder]);
            });
            
            return tempFormatted;
        };

        lines.forEach((line, idx) => {
            const trimmedLine = line.trim();
            if (!trimmedLine) return;

            // Check for Scene Separator ---
            if (trimmedLine === '---') {
                blocks.push(
                    <div key={`sep-${idx}`} className="my-8 flex items-center justify-center gap-4 opacity-30">
                        <div className="h-[1px] flex-1 bg-gradient-to-r from-transparent to-stone-400 dark:to-slate-600"></div>
                        <div className="flex gap-1">
                            <div className="w-1 h-1 rounded-full bg-stone-400 dark:bg-slate-600"></div>
                            <div className="w-1.5 h-1.5 rounded-full bg-stone-500 dark:bg-slate-500"></div>
                            <div className="w-1 h-1 rounded-full bg-stone-400 dark:bg-slate-600"></div>
                        </div>
                        <div className="h-[1px] flex-1 bg-gradient-to-l from-transparent to-stone-400 dark:to-slate-600"></div>
                    </div>
                );
                return;
            }

            // Skip garbage arrows
            if (trimmedLine === '->' || trimmedLine === '=>') return;

            // Check for Notification [Hệ thống: ...]
            const notificationMatch = trimmedLine.match(/^\[(Hệ thống|Thông báo|System|Notification):?\s*([^\]]+)\]/i);
            if (notificationMatch && contentBeautify) {
                blocks.push(
                    <div key={`notif-${idx}`} className="my-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg flex items-start gap-3 animate-in fade-in slide-in-from-left-2 duration-500">
                        <div className="mt-0.5 text-blue-500">
                            <Shield size={16} />
                        </div>
                        <div className="flex-1">
                            <span className="text-[10px] uppercase font-black tracking-widest text-blue-500/70 block mb-0.5">{notificationMatch[1]}</span>
                            <span className="text-sm font-medium text-blue-700 dark:text-blue-300">{notificationMatch[2]}</span>
                        </div>
                    </div>
                );
                return;
            }

            // Check for Dialogue: [Name]: "..." or [Name] nói: "..."
            // Improved regex to be more precise about dialogue blocks
            const dialogueMatch = trimmedLine.match(/^\s*(?:([^:<>\n]+?)(?:\s+nói)?:\s*)?(["“「][^"”」]*["”」])\s*$/i);
            
            if (dialogueMatch && contentBeautify) {
                const speakerNameRaw = dialogueMatch[1] ? dialogueMatch[1].trim() : "";
                const dialogueText = dialogueMatch[2];
                
                const isPC = playerName && speakerNameRaw.toLowerCase().includes(playerName.toLowerCase());
                const entity = entities?.find(e => speakerNameRaw.toLowerCase().includes(e.name.toLowerCase()));
                const isNPC = !!entity;
                
                const finalName = isPC ? playerName : (entity ? entity.name : (speakerNameRaw || "Người dẫn chuyện"));
                const initial = finalName.charAt(0).toUpperCase();
                const avatarUrl = isPC ? playerAvatar : (entity?.avatar);
                
                const avatarColor = isPC ? 'bg-sky-500' : (isNPC ? 'bg-amber-500' : 'bg-stone-500');
                const bubbleBg = isPC 
                    ? 'bg-sky-50 dark:bg-sky-900/20 border-sky-200/50 dark:border-sky-800/50' 
                    : (isNPC 
                        ? 'bg-white dark:bg-slate-800/40 border-stone-200 dark:border-slate-700' 
                        : 'bg-stone-100 dark:bg-stone-900/30 border-stone-200 dark:border-stone-800');
                const textColor = isPC 
                    ? 'text-sky-900 dark:text-sky-100' 
                    : (isNPC 
                        ? 'text-stone-900 dark:text-slate-200' 
                        : 'text-stone-800 dark:text-slate-300');
                const nameColor = isPC 
                    ? 'text-sky-600 dark:text-sky-400' 
                    : (isNPC 
                        ? 'text-amber-600 dark:text-amber-400' 
                        : 'text-stone-500 dark:text-slate-500');

                blocks.push(
                    <div key={`dialogue-${idx}`} className={`flex items-start gap-3 my-6 animate-in fade-in slide-in-from-left-2 duration-300 ${isPC ? 'flex-row-reverse' : ''}`}>
                        <div className={`w-9 h-9 rounded-xl ${avatarColor} text-white flex items-center justify-center text-sm font-bold flex-shrink-0 shadow-sm border-2 border-white dark:border-slate-800 overflow-hidden`}>
                            {avatarUrl ? (
                                <img src={avatarUrl} alt={finalName} className="w-full h-full object-cover" referrerPolicy="no-referrer" />
                            ) : (
                                initial
                            )}
                        </div>
                        <div className={`flex flex-col max-w-[80%] ${isPC ? 'items-end' : 'items-start'}`}>
                            <div className="flex items-center gap-2 mb-1 px-1">
                                <span className={`text-[11px] font-black ${nameColor} uppercase tracking-[0.1em]`}>{finalName}</span>
                                {isNPC && <span className="w-1 h-1 rounded-full bg-amber-500/50"></span>}
                                {isNPC && <span className="text-[9px] text-amber-600/70 dark:text-amber-500/50 font-bold uppercase tracking-tighter">NPC</span>}
                            </div>
                            <div className={`${bubbleBg} p-4 rounded-2xl ${isPC ? 'rounded-tr-none' : 'rounded-tl-none'} border shadow-[0_2px_10px_rgba(0,0,0,0.05)] dark:shadow-none relative group`}>
                                {/* Bubble Tail */}
                                <div className={`absolute top-0 ${isPC ? '-right-1.5' : '-left-1.5'} w-3 h-3 ${bubbleBg} border-t ${isPC ? 'border-r' : 'border-l'} transform ${isPC ? 'rotate-45' : '-rotate-45'} translate-y-3`}></div>
                                
                                <MarkdownRenderer 
                                    className={`text-base md:text-lg leading-relaxed ${textColor}`}
                                    content={highlightEntities(dialogueText)}
                                />
                            </div>
                        </div>
                    </div>
                );
                return;
            }

            // Default: Narrative block
            let narrativeHtml = highlightEntities(trimmedLine);
            
            // Handle internal thoughts (Tiếng lòng) - Thường trong ngoặc đơn (...)
            narrativeHtml = narrativeHtml.replace(
                /\(([^)]{10,})\)/g,
                '<span class="text-stone-400 dark:text-slate-500 italic font-serif opacity-90">($1)</span>'
            );

            // Handle Action Highlights [Hành động: ...]
            narrativeHtml = narrativeHtml.replace(
                /\[(Hành động|Action):?\s*([^\]]+)\]/gi,
                '<span class="px-1.5 py-0.5 bg-emerald-500/10 border border-emerald-500/30 rounded text-[10px] font-black text-emerald-600 dark:text-emerald-400 uppercase tracking-wider mx-1">$2</span>'
            );

            blocks.push(
                <MarkdownRenderer 
                    key={`narrative-${idx}`} 
                    className="mb-4 text-stone-800 dark:text-slate-300 leading-relaxed text-base md:text-lg font-normal opacity-95"
                    content={narrativeHtml}
                />
            );
        });

        return blocks;
    };

    const handleSaveEdit = () => {
        onUpdate(editedText);
        setIsEditing(false);
    };

    const handleCancelEdit = () => {
        setEditedText(text);
        setIsEditing(false);
    };

    if (isEditing) {
        return (
            <div className="w-full flex flex-col gap-2 animate-in fade-in duration-200 border border-mystic-accent/30 p-2 rounded bg-stone-200 dark:bg-mystic-900/80">
                <div className="flex justify-between items-center mb-1">
                    <span className="text-xs font-bold text-mystic-accent uppercase flex items-center gap-1">
                        <Edit2 size={12} /> Editing Raw Context
                    </span>
                    <button onClick={handleCancelEdit} className="text-stone-500 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white">
                        <X size={14} />
                    </button>
                </div>
                <textarea
                    value={editedText}
                    onChange={(e) => setEditedText(e.target.value)}
                    className="w-full h-96 bg-stone-200 dark:bg-mystic-950 border border-stone-400 dark:border-slate-800 rounded p-3 text-xs font-mono text-stone-800 dark:text-slate-300 focus:border-mystic-accent outline-none resize-y custom-scrollbar leading-relaxed"
                    placeholder="Nhập nội dung raw (bao gồm cả thẻ <thinking>, <content>...)"
                    spellCheck={false}
                />
                 <div className="flex justify-end gap-2 mt-1">
                    <button 
                        onClick={handleCancelEdit}
                        className="px-3 py-1.5 text-xs text-stone-500 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white bg-stone-300 dark:bg-slate-800 rounded border border-stone-400 dark:border-slate-700 hover:bg-stone-400 dark:hover:bg-slate-700 font-medium"
                    >
                        Hủy bỏ
                    </button>
                    <button 
                        onClick={handleSaveEdit}
                        className="px-3 py-1.5 text-xs text-mystic-900 bg-mystic-accent hover:bg-sky-400 rounded font-bold shadow-[0_0_10px_rgba(56,189,248,0.3)] flex items-center gap-1"
                    >
                        <Save size={12} /> Lưu thay đổi
                    </button>
                </div>
            </div>
        );
    }

    // Header Row: Thinking Toggle + Edit Button - Only show if needed
    const isUserMessage = turnNumber === undefined;
    
    if (thinkingContent || isEditing || isUserMessage) {
        return (
            <div className="flex flex-col gap-1 w-full group" ref={containerRef}>
                <div className="flex items-center gap-3 w-full min-h-[16px]">
                    {thinkingContent ? (
                        <button 
                            onClick={() => setShowThinking(!showThinking)}
                            className={`flex items-center gap-2 text-[10px] uppercase font-bold text-stone-500 hover:text-mystic-accent transition-all duration-200 ${showThinking ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}
                        >
                            <BrainCircuit size={12} />
                            {showThinking ? 'Ẩn dòng tư duy' : 'Hiện dòng tư duy'}
                        </button>
                    ) : (
                        <div className="flex-1"></div>
                    )}
                    
                    <button
                        onClick={() => setIsEditing(true)}
                        className={`flex items-center gap-1 text-[10px] font-bold text-stone-600 hover:text-sky-400 uppercase transition-all ${thinkingContent ? 'opacity-40 hover:opacity-100' : 'opacity-100'}`}
                        title={isUserMessage ? "Chỉnh sửa hành động của bạn" : "Chỉnh sửa Context gốc (Raw)"}
                    >
                        <Edit2 size={10} /> {isUserMessage ? 'Chỉnh sửa hành động' : 'Edit Raw'}
                    </button>
                </div>
                
                {thinkingContent && (
                    <AnimatePresence>
                        {showThinking && (
                            <motion.div 
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                className={`${
                                    contentBeautify 
                                    ? "bg-purple-500/5 dark:bg-purple-900/10 border-l-4 border-purple-400/50 p-4 rounded-r-xl shadow-inner mb-4" 
                                    : "bg-stone-200 dark:bg-slate-900/50 border-l-2 border-stone-400 dark:border-slate-600 p-3 rounded-r mb-2"
                                } text-xs text-stone-500 dark:text-slate-400 font-mono overflow-hidden whitespace-pre-wrap relative`}
                            >
                                {contentBeautify && (
                                    <div className="absolute top-2 right-2 opacity-20">
                                        <Brain size={24} />
                                    </div>
                                )}
                                <div className={contentBeautify ? "opacity-80 leading-relaxed" : ""}>
                                    {thinkingContent}
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                )}
                
                <div 
                  className="font-sans text-base md:text-lg text-stone-800 dark:text-slate-300 [&>p]:mb-3 last:[&>p]:mb-0 leading-[20px] font-normal not-italic"
                >
                    {turnNumber !== undefined && (
                        <div id={`turn-${turnNumber}`} className="mb-4 pb-2 border-b border-stone-400/30 dark:border-slate-800/50 scroll-mt-20">
                            <div className="text-[10px] font-black uppercase tracking-[0.2em] text-mystic-accent mb-1 opacity-80">
                                {turnNumber === 0 ? 'Khởi Đầu (Lượt 0)' : `Lượt ${turnNumber}`}
                            </div>
                            {userAction && (
                                <div className="text-sm font-serif italic text-stone-500 dark:text-slate-400">
                                    &ldquo;{userAction}&rdquo;
                                </div>
                            )}
                        </div>
                    )}
                    {renderContentBlocks(mainContent) || (isStreaming ? null : (
                        <div className="p-4 bg-stone-300/50 dark:bg-slate-900/50 border border-dashed border-stone-400 dark:border-slate-700 rounded-xl">
                            <div className="flex items-center gap-2 text-red-500 mb-2">
                                <Shield size={16} />
                                <span className="text-xs font-bold uppercase tracking-wider">Lỗi Hiển Thị Nội Dung</span>
                            </div>
                            <p className="text-stone-600 dark:text-slate-400 italic text-sm mb-3">
                                [Nội dung truyện trống hoặc đã bị bộ lọc hệ thống loại bỏ hoàn toàn. AI có thể đã gặp lỗi logic, vi phạm chính sách an toàn, hoặc chỉ phản hồi các thẻ kỹ thuật mà không có văn bản truyện.]
                            </p>
                            <div className="flex flex-wrap gap-2">
                                <button 
                                    onClick={() => setIsEditing(true)}
                                    className="px-3 py-1.5 bg-mystic-accent/10 hover:bg-mystic-accent/20 border border-mystic-accent/30 rounded text-[10px] font-bold text-mystic-accent uppercase transition-colors flex items-center gap-1"
                                >
                                    <Edit2 size={10} /> Kiểm tra dữ liệu gốc (Edit Raw)
                                </button>
                                <button 
                                    onClick={() => window.location.reload()} 
                                    className="px-3 py-1.5 bg-stone-400/20 hover:bg-stone-400/30 border border-stone-400/30 rounded text-[10px] font-bold text-stone-500 uppercase transition-colors flex items-center gap-1"
                                >
                                    <RefreshCw size={10} /> Tải lại trang
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    // Default return to prevent React Error #31
    return (
        <div className="font-sans text-base md:text-lg text-stone-800 dark:text-slate-300 [&>p]:mb-3 last:[&>p]:mb-0 leading-[20px] font-normal not-italic" ref={containerRef}>
            {turnNumber !== undefined && (
                <div id={`turn-${turnNumber}`} className="mb-4 pb-2 border-b border-stone-400/30 dark:border-slate-800/50 scroll-mt-20">
                    <div className="text-[10px] font-black uppercase tracking-[0.2em] text-mystic-accent mb-1 opacity-80">
                        {turnNumber === 0 ? 'Khởi Đầu (Lượt 0)' : `Lượt ${turnNumber}`}
                    </div>
                    {userAction && (
                        <div className="text-sm font-serif italic text-stone-500 dark:text-slate-400">
                            &ldquo;{userAction}&rdquo;
                        </div>
                    )}
                </div>
            )}
            {renderContentBlocks(mainContent) || (isStreaming ? null : <div className="text-stone-500 italic text-sm">[Nội dung truyện trống - AI có thể đã gặp lỗi hoặc chỉ phản hồi các thẻ hệ thống. Bạn có thể thử 'Regenerate' hoặc kiểm tra 'Edit Raw' để xem dữ liệu gốc]</div>)}
        </div>
    );
};

// --- RULES MANAGER COMPONENT ---
const RulesManager: React.FC<{
    rules: string[];
    onUpdate: (newRules: string[]) => void;
}> = ({ rules, onUpdate }) => {
    // ... (No changes to RulesManager)
    const [isOpen, setIsOpen] = useState(false);
    const [newRule, setNewRule] = useState("");
    const [editIdx, setEditIdx] = useState<number | null>(null);
    const [editValue, setEditValue] = useState("");

    const handleAdd = () => {
        if (!newRule.trim()) return;
        onUpdate([...rules, newRule.trim()]);
        setNewRule("");
    };

    const handleDelete = (idx: number) => {
        onUpdate(rules.filter((_, i) => i !== idx));
    };

    const startEdit = (idx: number, val: string) => {
        setEditIdx(idx);
        setEditValue(val);
    };

    const saveEdit = (idx: number) => {
        const updated = [...rules];
        updated[idx] = editValue;
        onUpdate(updated);
        setEditIdx(null);
    };

    return (
        <>
            <button 
                onClick={() => setIsOpen(true)}
                className="w-full p-3 flex justify-between items-center text-left hover:bg-stone-400 dark:hover:bg-slate-700/50 transition-colors bg-stone-300 dark:bg-slate-800/30 rounded-lg border border-stone-400 dark:border-slate-700 mb-3"
            >
                <div className="flex items-center gap-2 text-sm font-bold text-red-600 dark:text-red-400 uppercase">
                    <BookOpen size={16} />
                    Luật Lệ & Ràng Buộc
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-[10px] bg-stone-400 dark:bg-slate-800 px-2 py-0.5 rounded text-stone-500 dark:text-slate-400">{rules.length} Active</span>
                </div>
            </button>

            <AnimatePresence>
                {isOpen && (
                    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
                        <motion.div 
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            className="bg-stone-200 dark:bg-mystic-950 border border-stone-400 dark:border-slate-800 w-[99vw] h-[99vh] rounded-xl shadow-2xl flex flex-col overflow-hidden"
                        >
                            {/* Modal Header */}
                            <div className="p-4 border-b border-stone-400 dark:border-slate-800 flex justify-between items-center bg-stone-300 dark:bg-mystic-900/50 shrink-0">
                                <h2 className="text-lg font-bold text-red-600 dark:text-red-400 flex items-center gap-2">
                                    <BookOpen size={20}/> Luật Bất Khả Kháng
                                </h2>
                                <button onClick={() => setIsOpen(false)} className="text-stone-500 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white p-1 rounded hover:bg-stone-400 dark:hover:bg-mystic-900 transition-colors">
                                    <X size={24} />
                                </button>
                            </div>

                            {/* Modal Body */}
                            <div className="p-4 flex-1 overflow-y-auto custom-scrollbar bg-stone-200 dark:bg-mystic-950">
                                <p className="text-xs text-stone-500 mb-4 italic border-l-2 border-red-200 dark:border-red-900/50 pl-2">
                                    Các quy tắc này có quyền lực tối thượng, ép buộc AI phải tuân theo bất kể ngữ cảnh hay logic thông thường.
                                </p>

                                <div className="space-y-2 mb-4">
                                    {/* Rule 0 - System Mandatory Rule */}
                                    <div className="flex gap-2 items-start bg-red-500/10 dark:bg-red-900/20 p-3 rounded border border-red-500/30 transition-colors">
                                        <span className="text-red-600 dark:text-red-500 font-bold text-xs mt-0.5">0.</span>
                                        <p className="flex-1 text-sm text-stone-800 dark:text-slate-200 font-bold leading-relaxed">
                                            Cấm AI viết các nội dung như chửi nhau 'Mày' 'Tao'.
                                        </p>
                                        <div className="px-1.5 py-0.5 bg-red-500 text-white text-[8px] font-black uppercase rounded">Hệ Thống</div>
                                    </div>

                                    {rules.length === 0 && rules.length === 0 && <p className="text-xs text-center text-stone-400 dark:text-slate-600 py-8 border border-dashed border-stone-400 dark:border-slate-700 rounded-lg">Chưa có luật lệ bổ sung.</p>}
                                    {rules.map((rule, idx) => (
                                        <div key={idx} className="flex gap-2 items-start bg-stone-300 dark:bg-slate-800/50 p-3 rounded border border-stone-400 dark:border-slate-700/50 group hover:border-red-300 dark:hover:border-red-900/50 transition-colors">
                                            <span className="text-red-600 dark:text-red-500 font-bold text-xs mt-0.5">{idx + 1}.</span>
                                            {editIdx === idx ? (
                                                <div className="flex-1 flex gap-2">
                                                    <input 
                                                        autoFocus
                                                        value={editValue}
                                                        onChange={(e) => setEditValue(e.target.value)}
                                                        className="flex-1 bg-stone-200 dark:bg-slate-900 text-sm text-stone-800 dark:text-slate-200 border border-stone-400 dark:border-slate-600 rounded px-2 py-1 outline-none focus:border-red-400"
                                                    />
                                                    <button onClick={() => saveEdit(idx)} className="text-green-600 dark:text-green-400 p-1 hover:bg-green-100 dark:hover:bg-green-900/20 rounded"><Check size={16}/></button>
                                                </div>
                                            ) : (
                                                <p className="flex-1 text-sm text-stone-700 dark:text-slate-300 leading-relaxed">{rule}</p>
                                            )}
                                            
                                            {editIdx !== idx && (
                                                <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                                    <button onClick={() => startEdit(idx, rule)} className="p-1 text-stone-400 dark:text-stone-500 hover:text-mystic-accent hover:bg-stone-400 dark:hover:bg-slate-700 rounded"><Edit2 size={14}/></button>
                                                    <button onClick={() => handleDelete(idx)} className="p-1 text-stone-400 dark:text-stone-500 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-100 dark:hover:bg-red-900/20 rounded"><Trash2 size={14}/></button>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>

                                {/* Add New Rule Input */}
                                <div className="flex gap-2 pt-2 border-t border-stone-400 dark:border-slate-800">
                                    <input 
                                        value={newRule}
                                        onChange={(e) => setNewRule(e.target.value)}
                                        placeholder="Thêm luật mới (VD: Cấm giết NPC này, Tiền tệ là Vàng)..."
                                        className="flex-1 bg-stone-300 dark:bg-slate-900 border border-stone-400 dark:border-slate-600 rounded px-3 py-2 text-sm text-stone-800 dark:text-slate-200 focus:border-red-400 outline-none"
                                        onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
                                    />
                                    <button 
                                        onClick={handleAdd}
                                        className="px-4 bg-red-100 dark:bg-red-900/30 border border-red-200 dark:border-red-900/50 text-red-600 dark:text-red-400 rounded hover:bg-red-200 dark:hover:bg-red-900/50 font-bold text-sm"
                                    >
                                        Thêm
                                    </button>
                                </div>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </>
    );
};


const EntityDetailModal: React.FC<{
    entity: Entity | null;
    onClose: () => void;
    onUpdateAvatar: (entityId: string) => void;
}> = ({ entity, onClose, onUpdateAvatar }) => {
    if (!entity) return null;

    const getIcon = () => {
        switch (entity.type) {
            case 'NPC': return <User size={20} className="text-sky-500" />;
            case 'LOCATION': return <MapPin size={20} className="text-emerald-500" />;
            case 'FACTION': return <Shield size={20} className="text-rose-500" />;
            case 'ITEM': return <Box size={20} className="text-amber-500" />;
            default: return <Box size={20} className="text-stone-500" />;
        }
    };

    const getTitle = () => {
        switch (entity.type) {
            case 'NPC': return 'Thông tin Nhân vật';
            case 'LOCATION': return 'Thông tin Địa danh';
            case 'FACTION': return 'Thông tin Phe phái';
            case 'ITEM': return 'Thông tin Vật phẩm';
            default: return 'Thông tin Thực thể';
        }
    };

    return (
        <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
            <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-stone-200 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 w-full max-w-lg rounded-xl shadow-2xl flex flex-col max-h-[80vh] overflow-hidden"
            >
                <div className="p-4 border-b border-stone-400 dark:border-slate-800 flex justify-between items-center bg-stone-300 dark:bg-slate-900/50">
                    <h2 className="text-lg font-bold text-stone-800 dark:text-slate-200 flex items-center gap-2">
                        {getIcon()} {getTitle()}
                    </h2>
                    <button onClick={onClose} className="text-stone-500 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white p-1 rounded hover:bg-stone-400 dark:hover:bg-slate-800 transition-colors">
                        <X size={24} />
                    </button>
                </div>

                <div className="p-6 overflow-y-auto custom-scrollbar space-y-4 bg-stone-200 dark:bg-mystic-900">
                    <div className="flex items-start gap-4">
                        <button 
                            onClick={() => onUpdateAvatar(entity.id)}
                            className="w-20 h-20 rounded-xl bg-stone-300 dark:bg-slate-800 border-2 border-mystic-accent flex items-center justify-center shrink-0 shadow-lg overflow-hidden group relative"
                        >
                            {entity.avatar ? (
                                <img src={entity.avatar} alt={entity.name} className="w-full h-full object-cover" referrerPolicy="no-referrer" />
                            ) : (
                                <div className="text-mystic-accent opacity-50 group-hover:opacity-100 transition-opacity flex flex-col items-center">
                                    {getIcon()}
                                    <span className="text-[8px] mt-1 font-bold">Thêm ảnh</span>
                                </div>
                            )}
                            <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                <Edit2 size={16} className="text-white" />
                            </div>
                        </button>
                        <div>
                            <h3 className="text-xl font-bold text-stone-900 dark:text-white mb-1">{entity.name}</h3>
                            <span className="text-xs font-bold text-mystic-accent uppercase tracking-widest opacity-70">
                                {entity.type === 'CUSTOM' ? entity.customType || 'Vật phẩm' : entity.type}
                            </span>
                        </div>
                    </div>

                    <div className="space-y-4">
                        {entity.personality && (
                            <div className="space-y-1">
                                <h4 className="text-xs font-bold text-mystic-accent uppercase tracking-wider">Tính cách</h4>
                                <MarkdownRenderer 
                                    className="text-sm text-stone-700 dark:text-slate-300 bg-stone-300 dark:bg-slate-800/50 p-3 rounded border border-stone-400 dark:border-slate-700/50 leading-relaxed"
                                    content={entity.personality}
                                />
                            </div>
                        )}

                        {entity.type === 'ITEM' && (entity.rarity || entity.price) && (
                            <div className="grid grid-cols-2 gap-3">
                                {entity.rarity && (
                                    <div className="space-y-1">
                                        <h4 className="text-xs font-bold text-mystic-accent uppercase tracking-wider">Độ hiếm</h4>
                                        <div className="text-sm font-bold text-stone-700 dark:text-slate-300 bg-stone-300 dark:bg-slate-800/50 px-3 py-2 rounded border border-stone-400 dark:border-slate-700/50">
                                            {entity.rarity}
                                        </div>
                                    </div>
                                )}
                                {entity.price && (
                                    <div className="space-y-1">
                                        <h4 className="text-xs font-bold text-mystic-accent uppercase tracking-wider">Giá cả</h4>
                                        <div className="text-sm font-bold text-stone-700 dark:text-slate-300 bg-stone-300 dark:bg-slate-800/50 px-3 py-2 rounded border border-stone-400 dark:border-slate-700/50">
                                            {entity.price}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        <div className="space-y-1">
                            <h4 className="text-xs font-bold text-mystic-accent uppercase tracking-wider">Mô tả</h4>
                            <MarkdownRenderer 
                                className="text-sm text-stone-700 dark:text-slate-300 bg-stone-300 dark:bg-slate-800/50 p-3 rounded border border-stone-400 dark:border-slate-700/50 leading-relaxed"
                                content={entity.description || "Không có mô tả chi tiết."}
                            />
                        </div>
                    </div>
                </div>
            </motion.div>
        </div>
    );
};

const GameplayScreen: React.FC<NavigationProps> = ({ onNavigate, activeWorld, onUpdateWorld }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState<ChatMessage[]>(activeWorld?.savedState?.history || []);
  const [inputValue, setInputValue] = useState('');
  const [lastAction, setLastAction] = useState('');
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [tempInputValue, setTempInputValue] = useState('');
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [turnCount, setTurnCount] = useState(activeWorld?.savedState?.turnCount || 0);
  const [showMobileSidebar, setShowMobileSidebar] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  
  // AI Monitor States
  const [tokenHistory, setTokenHistory] = useState<{tokens: number, words: number, timestamp: number}[]>(activeWorld?.savedState?.aiMonitor?.tokenHistory || []);
  const [totalTokens, setTotalTokens] = useState<number>(activeWorld?.savedState?.aiMonitor?.totalTokens || 0);
  const processingStartTimeRef = useRef<number | null>(null);
  const [currentProcessingTime, setCurrentProcessingTime] = useState<number>(0);
  const [lastTurnTotalTime, setLastTurnTotalTime] = useState<number>(activeWorld?.savedState?.aiMonitor?.lastTurnTotalTime || 0);

  // Image Library States
  const [showImageLibrary, setShowImageLibrary] = useState(false);
  const [selectingAvatarFor, setSelectingAvatarFor] = useState<{ type: 'player' | 'entity', id?: string } | null>(null);

  // UI States
  const [showTokenDetails, setShowTokenDetails] = useState(true);
  const [showStatsDetails, setShowStatsDetails] = useState(true);
  const [isInputCollapsed, setIsInputCollapsed] = useState(false);
  const [showCharModal, setShowCharModal] = useState(false);
  const [showGlobalModal, setShowGlobalModal] = useState(false);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [autosaveList, setAutosaveList] = useState<SaveFile[]>([]);
  const [manualSaveList, setManualSaveList] = useState<SaveFile[]>([]);
  const [initialSaveList, setInitialSaveList] = useState<SaveFile[]>([]);
  const [activeSaveTab, setActiveSaveTab] = useState<'manual' | 'autosave' | 'history' | 'initial'>('manual');
  const [showContextModal, setShowContextModal] = useState(false);
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null);

  // Pagination State
  const [currentPage, setCurrentPage] = useState(1);
  const [lastNavigatedTurn, setLastNavigatedTurn] = useState<number | null>(null);
  const pendingScrollTurnRef = useRef<number | null>(null);

  // Dynamic States (Rules & Tawa)
  const [tawaPresetConfig, setTawaPresetConfig] = useState<TawaPresetConfig>(DEFAULT_PRESET_CONFIG);
  const [dynamicRules, setDynamicRules] = useState<string[]>(activeWorld?.config?.rules || []);
  const [gameTime, setGameTime] = useState<GameTime>(activeWorld?.gameTime || INITIAL_GAME_TIME);

  const handleAvatarSelect = async (image: ImageMetadata) => {
    if (!selectingAvatarFor) return;

    if (selectingAvatarFor.type === 'player') {
        const updatedWorld = {
            ...activeWorld,
            player: { ...activeWorld.player, avatar: image.data }
        };
        onUpdateWorld(updatedWorld);
    } else if (selectingAvatarFor.type === 'entity' && selectingAvatarFor.id) {
        const updatedEntities = activeWorld.entities.map(e => 
            e.id === selectingAvatarFor.id ? { ...e, avatar: image.data } : e
        );
        const updatedWorld = {
            ...activeWorld,
            entities: updatedEntities
        };
        onUpdateWorld(updatedWorld);
    }
    
    setSelectingAvatarFor(null);
    setShowImageLibrary(false);
  };

  // LSR State
  const [lsrTables, setLsrTables] = useState<LsrTableDefinition[]>([]);
  const [lsrRuntimeData, setLsrRuntimeData] = useState<Record<string, unknown[]>>(activeWorld?.lsrData || {});
  const [activeLsrTableId, setActiveLsrTableId] = useState<string | null>(null);
  const { isMobile } = useResponsive();

  // --- DERIVED STATE (MUST BE BEFORE HOOKS THAT USE THEM) ---
  const totalPages = history.length <= 11 ? 1 : 1 + Math.ceil((history.length - 11) / MESSAGES_PER_PAGE);
  const startIndex = currentPage === 1 ? 0 : 11 + (currentPage - 2) * MESSAGES_PER_PAGE;
  const endIndex = currentPage === 1 ? 11 : startIndex + MESSAGES_PER_PAGE;
  const displayedMessages = history.slice(startIndex, endIndex);

  const initializedRef = useRef(false);
  const isReadyRef = useRef(false); // Guard for syncing state - ALWAYS false initially
  const lastWorldRef = useRef<WorldData | null>(null);
  const initialStartedRef = useRef(false); // Guard for initial generation
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Refs for state data to ensure up-to-date values in callbacks without triggering re-renders or loops
  const tokenHistoryRef = useRef<{tokens: number, words: number, timestamp: number}[]>(activeWorld?.savedState?.aiMonitor?.tokenHistory || []);
  const totalTokensRef = useRef<number>(activeWorld?.savedState?.aiMonitor?.totalTokens || 0);
  const lastTurnTotalTimeRef = useRef<number>(activeWorld?.savedState?.aiMonitor?.lastTurnTotalTime || 0);
  const historyRef = useRef<ChatMessage[]>(activeWorld?.savedState?.history || []);
  const turnCountRef = useRef<number>(activeWorld?.savedState?.turnCount || 0);
  const gameTimeRef = useRef<GameTime>(activeWorld?.gameTime || INITIAL_GAME_TIME);
  const lsrRuntimeDataRef = useRef<Record<string, unknown[]>>(activeWorld?.lsrData || {});
  const activeWorldSummaryRef = useRef<string | undefined>(activeWorld?.summary);
  const lastActionRef = useRef<string | undefined>(lastAction);
  const activeWorldRef = useRef<WorldData | null>(activeWorld);
  const dynamicRulesRef = useRef<string[]>(activeWorld?.config?.rules || []);
  const tawaPresetConfigRef = useRef<TawaPresetConfig>(tawaPresetConfig);

  // Sync refs with state
  useEffect(() => { tokenHistoryRef.current = tokenHistory; }, [tokenHistory]);
  useEffect(() => { totalTokensRef.current = totalTokens; }, [totalTokens]);
  useEffect(() => { lastTurnTotalTimeRef.current = lastTurnTotalTime; }, [lastTurnTotalTime]);
  useEffect(() => { historyRef.current = history; }, [history]);
  useEffect(() => { turnCountRef.current = turnCount; }, [turnCount]);
  useEffect(() => { gameTimeRef.current = gameTime; }, [gameTime]);
  useEffect(() => { lsrRuntimeDataRef.current = lsrRuntimeData; }, [lsrRuntimeData]);
  useEffect(() => { activeWorldSummaryRef.current = activeWorld?.summary; }, [activeWorld?.summary]);
  useEffect(() => { lastActionRef.current = lastAction; }, [lastAction]);
  useEffect(() => { activeWorldRef.current = activeWorld; }, [activeWorld]);
  useEffect(() => { dynamicRulesRef.current = dynamicRules; }, [dynamicRules]);
  useEffect(() => { tawaPresetConfigRef.current = tawaPresetConfig; }, [tawaPresetConfig]);

  // Helper to sync state back to App.tsx
  const syncWorldState = useCallback((currentHistory?: ChatMessage[], currentTurn?: number, currentTime?: GameTime, currentLsrData?: Record<string, unknown[]>, currentSummary?: string) => {
    if (!isReadyRef.current) return; // Guard against stale sync during init
    // console.log(`GameplayScreen: Syncing state to App.tsx (Turn ${currentTurn !== undefined ? currentTurn : turnCountRef.current})...`);
    if (onUpdateWorld) {
      onUpdateWorld({
        summary: currentSummary || activeWorldSummaryRef.current,
        lsrData: currentLsrData || lsrRuntimeDataRef.current,
        savedState: {
          history: currentHistory || historyRef.current,
          turnCount: currentTurn !== undefined ? currentTurn : turnCountRef.current,
          gameTime: currentTime || gameTimeRef.current,
          aiMonitor: {
            tokenHistory: tokenHistoryRef.current,
            totalTokens: totalTokensRef.current,
            lastTurnTotalTime: lastTurnTotalTimeRef.current
          }
        }
      });
    }
  }, [onUpdateWorld]);

  // Timer Logic
  useEffect(() => {
      if (isLoading) {
          processingStartTimeRef.current = Date.now();
          setCurrentProcessingTime(0);
          timerRef.current = setInterval(() => {
              setCurrentProcessingTime(prev => prev + 100);
          }, 100);
      } else {
          if (timerRef.current) {
              clearInterval(timerRef.current);
              timerRef.current = null;
          }
          if (processingStartTimeRef.current) {
              const total = Date.now() - processingStartTimeRef.current;
              setLastTurnTotalTime(total);
              lastTurnTotalTimeRef.current = total; // Update ref immediately
              processingStartTimeRef.current = null;
              // Sync one last time to capture the final processing time
              syncWorldState();
          }
      }
      return () => {
          if (timerRef.current) clearInterval(timerRef.current);
      };
  }, [isLoading, syncWorldState]);

  // Initialize active LSR table
  useEffect(() => {
    if (lsrTables.length > 0 && !activeLsrTableId) {
      setActiveLsrTableId(lsrTables[0].id);
    }
  }, [lsrTables, activeLsrTableId]);

  const formatTime = (ms: number) => {
      const totalSeconds = Math.floor(ms / 1000);
      const minutes = Math.floor(totalSeconds / 60);
      const seconds = totalSeconds % 60;
      const milliseconds = Math.floor((ms % 1000) / 100);
      return `${minutes}:${seconds.toString().padStart(2, '0')}.${milliseconds}`;
  };

  const updateTokenHistory = (tokens: number, text?: string) => {
      if (!tokens) return;
      const words = text ? text.trim().split(/\s+/).length : 0;
      const newEntry = { tokens, words, timestamp: Date.now() };
      
      // Update state for UI
      setTokenHistory(prev => {
          const updated = [newEntry, ...prev].slice(0, 5);
          tokenHistoryRef.current = updated; // Update ref
          return updated;
      });
      setTotalTokens(prev => {
          const updated = prev + tokens;
          totalTokensRef.current = updated; // Update ref
          return updated;
      });
  };

  // Sync AI Monitor data whenever it changes to persist across navigation
  // REMOVED to prevent infinite loop. Syncing is now handled manually after AI responses.


  const formatNumber = (num: number) => {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
  };

  const AIMonitor = () => {
    const isProxy = !!settings?.proxyUrl;
    const activeModel = (isProxy && settings?.proxyModel) ? settings?.proxyModel : settings?.aiModel;
    
    return (
        <div className="p-3 bg-stone-300 dark:bg-slate-900/80 rounded-lg border border-stone-400 dark:border-slate-700 space-y-2 font-mono text-[10px] mt-2">
            <div className="flex justify-between items-center border-b border-stone-400 dark:border-slate-800 pb-1.5">
                <span className="text-stone-500 dark:text-slate-500 uppercase font-bold">AI Monitor</span>
                <div className="flex items-center gap-2">
                    {isLoading && <div className="w-1.5 h-1.5 rounded-full bg-mystic-accent animate-pulse" />}
                    <span className={isLoading ? "text-mystic-accent" : "text-stone-400 dark:text-slate-600"}>
                        {isLoading ? "PROCESSING" : "IDLE"}
                    </span>
                </div>
            </div>

            <div className="space-y-2">
                <div className="flex flex-col">
                    <span className="text-stone-500 dark:text-slate-500 uppercase">Connection</span>
                    <span className={isProxy ? "text-sky-500" : "text-emerald-500"}>
                        {isProxy ? "REVERSE PROXY" : "DIRECT API"}
                    </span>
                </div>

                <div className="flex flex-col">
                    <span className="text-stone-500 dark:text-slate-500 uppercase">Active Model</span>
                    <span className="text-stone-700 dark:text-slate-300 truncate" title={activeModel}>
                        {activeModel}
                    </span>
                </div>

                <div className="flex flex-col">
                    <span className="text-stone-500 dark:text-slate-500 uppercase">Timer</span>
                    <span className="text-amber-500 font-bold">
                        {formatTime(isLoading ? currentProcessingTime : lastTurnTotalTime)}
                    </span>
                </div>

                <div className="flex flex-col">
                    <div className="flex justify-between items-center cursor-pointer hover:text-mystic-accent transition-colors" onClick={() => setShowTokenDetails(!showTokenDetails)}>
                        <span className="text-stone-500 dark:text-slate-500 uppercase">Tokens (Last 5)</span>
                        <span className="text-[8px] opacity-50">{showTokenDetails ? "Ẩn" : "Hiện"} chi tiết</span>
                    </div>
                    <div className="flex gap-1 items-end h-4 mt-1">
                        {tokenHistory.length > 0 ? tokenHistory.map((entry, i) => (
                            <div 
                                key={i} 
                                className="bg-mystic-accent/30 border-t border-mystic-accent w-2" 
                                style={{ height: `${Math.min(100, (entry.tokens / 4000) * 100)}%` }}
                                title={`${entry.tokens} tokens, ${entry.words} words`}
                            />
                        )) : <span className="text-stone-400">No data</span>}
                    </div>
                    
                    {showTokenDetails && tokenHistory.length > 0 && (
                        <div className="mt-2 space-y-1 border-t border-stone-400/30 dark:border-slate-800/50 pt-1">
                            {tokenHistory.map((entry, i) => (
                                <div key={i} className="flex justify-between text-[9px] text-stone-600 dark:text-slate-400">
                                    <span>#{tokenHistory.length - i}</span>
                                    <span>{formatNumber(entry.tokens)} tkn</span>
                                    <span>{formatNumber(entry.words)} chữ</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                <div className="flex flex-col border-t border-stone-400 dark:border-slate-800 pt-1">
                    <div className="flex justify-between items-center cursor-pointer hover:text-mystic-accent transition-colors" onClick={() => setShowStatsDetails(!showStatsDetails)}>
                        <span className="text-stone-500 dark:text-slate-500 uppercase">Thống kê</span>
                        <span className="text-[8px] opacity-50">{showStatsDetails ? "Ẩn" : "Hiện"} chi tiết</span>
                    </div>
                    
                    {showStatsDetails && (
                        <div className="flex flex-col gap-0.5 mt-1 animate-in fade-in slide-in-from-top-1 duration-200">
                            <div className="flex justify-between">
                                <span>Gần nhất:</span>
                                <span className="text-stone-700 dark:text-slate-300">{tokenHistory.length > 0 ? formatNumber(tokenHistory[0].tokens) : 0} tkn</span>
                            </div>
                            <div className="flex justify-between">
                                <span>Tổng cộng:</span>
                                <span className="text-mystic-accent font-bold">{formatNumber(totalTokens)} tkn</span>
                            </div>
                            <div className="flex justify-between">
                                <span>Trung bình:</span>
                                <span className="text-stone-700 dark:text-slate-300">
                                    {tokenHistory.length > 0 
                                        ? formatNumber(Math.round(tokenHistory.reduce((a, b) => a + b.tokens, 0) / tokenHistory.length)) 
                                        : 0} tkn
                                </span>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
  };
  // Refs for auto-scrolling
  const chatEndRef = useRef<HTMLDivElement>(null);
  const scrollViewportRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);

  // --- Handlers (Defined as functions to avoid ReferenceError) ---

  // --- Auto-Scroll & Pagination Logic ---
  useEffect(() => {
    const totalPages = history.length <= 11 ? 1 : 1 + Math.ceil((history.length - 11) / MESSAGES_PER_PAGE);
    // Auto switch to last page when new message arrives
    if (history.length > 0) {
        setCurrentPage(totalPages);
    }
  }, [history.length]);

  // Scroll handler to detect if user is at the bottom
  const handleScroll = () => {
    if (scrollViewportRef.current) {
        const { scrollTop, scrollHeight, clientHeight } = scrollViewportRef.current;
        // Check if user is near bottom (e.g. 50px tolerance)
        const isAtBottom = scrollHeight - (scrollTop + clientHeight) < 50;
        shouldAutoScrollRef.current = isAtBottom;
    }
  };

  useEffect(() => {
    if (pendingScrollTurnRef.current !== null) {
      const turnNumber = pendingScrollTurnRef.current;
      // Đợi một chút để DOM cập nhật sau khi chuyển trang
      const timer = setTimeout(() => {
        const element = document.getElementById(`turn-${turnNumber}`);
        if (element) {
          element.scrollIntoView({ behavior: 'smooth' });
          pendingScrollTurnRef.current = null;
        }
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [currentPage, displayedMessages]);

  useEffect(() => {
      if (shouldAutoScrollRef.current) {
          // Khi AI vừa kết thúc lượt, ưu tiên cuộn đến đầu dòng "Lượt" mới nhất
          if (!isLoading && history.length > 0) {
              const lastMsg = history[history.length - 1];
              if (lastMsg.role === 'model' && lastMsg.turnNumber !== undefined) {
                  // Đợi DOM render xong ID
                  const scrollTimeout = setTimeout(() => {
                      const element = document.getElementById(`turn-${lastMsg.turnNumber}`);
                      if (element) {
                          element.scrollIntoView({ behavior: 'smooth' });
                          setLastNavigatedTurn(lastMsg.turnNumber);
                      } else if (chatEndRef.current) {
                          chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
                      }
                  }, 150);
                  return () => clearTimeout(scrollTimeout);
              }
          }

          if (chatEndRef.current) {
              chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
          }
      }
  }, [history, isLoading]);
  
  // Force scroll when page changes (navigating history)
  useEffect(() => {
      if (chatEndRef.current && pendingScrollTurnRef.current === null) {
          chatEndRef.current.scrollIntoView({ behavior: 'auto' });
          shouldAutoScrollRef.current = true; 
      }
  }, [currentPage]);


  // --- Handlers ---
  const triggerInitialSave = useCallback(async (world: WorldData, time: GameTime) => {
      if (!isReadyRef.current) return;
      try {
          // console.log("GameplayScreen: Triggering Initial Save (Bản lưu lượt 0)...");
          const worldData: WorldData = {
              ...world,
              lsrData: lsrRuntimeDataRef.current,
              savedState: {
                  history: [],
                  turnCount: 0,
                  gameTime: time,
                  aiMonitor: {
                      tokenHistory: tokenHistoryRef.current,
                      totalTokens: totalTokensRef.current,
                      lastTurnTotalTime: lastTurnTotalTimeRef.current
                  }
              }
          };
          const worldName = world.world.worldName || 'Unknown_World';
          const slotId = `initial-${worldName.replace(/\s+/g, '_')}-start`;

          await dbService.saveAutosave({
              id: slotId,
              name: `${worldName} - Bản lưu lượt 0`,
              createdAt: Date.now(),
              updatedAt: Date.now(),
              data: worldData
          });
      } catch (err) {
          console.error("Initial save failed", err);
      }
  }, []);

  const triggerAutosave = useCallback(async (currentHistory: ChatMessage[], currentTurn: number, currentTime: GameTime, currentLsrData?: Record<string, unknown[]>) => {
      if (!activeWorldRef.current || !isReadyRef.current) return;
      try {
          // console.log(`GameplayScreen: Triggering Autosave for Turn ${currentTurn}...`);
          const worldData: WorldData = {
              ...activeWorldRef.current,
              lsrData: currentLsrData || lsrRuntimeDataRef.current, // Use provided data or ref
              savedState: {
                  history: currentHistory,
                  turnCount: currentTurn,
                  gameTime: currentTime,
                  aiMonitor: {
                      tokenHistory: tokenHistoryRef.current,
                      totalTokens: totalTokensRef.current,
                      lastTurnTotalTime: lastTurnTotalTimeRef.current
                  }
              }
          };
          
          const worldName = activeWorldRef.current.world.worldName || 'Unknown_World';
          const slotId = `autosave-${worldName.replace(/\s+/g, '_')}-${currentTurn}`;

          await dbService.saveAutosave({
              id: slotId,
              name: `${worldName} - Lượt ${currentTurn} (Tự động)`,
              createdAt: Date.now(),
              updatedAt: Date.now(),
              data: worldData
          });
      } catch (err) {
          console.error("Autosave failed", err);
      }
  }, []);

  const handleSend = async (overrideValue?: string) => {
    const textToSend = (overrideValue || inputValue).trim();
    if (!textToSend || isLoading || !activeWorld || !settings) return;

    setLastAction(textToSend);
    setHistoryIndex(-1);
    setTempInputValue('');
    setLastNavigatedTurn(null);

    // Thời gian sẽ được AI quyết định trong phản hồi tiếp theo
    const userMsg: ChatMessage = { 
        role: 'user', 
        text: textToSend, 
        timestamp: Date.now(),
        gameTime: gameTime, // Giữ nguyên thời gian hiện tại cho tin nhắn người dùng
        turnNumber: turnCount + 1 // User action starts the new turn
    };
    const newHistory = [...history, userMsg];
    setHistory(newHistory);
    syncWorldState(newHistory, turnCount, gameTime);
    setInputValue('');
    
    // Force auto-scroll on send
    shouldAutoScrollRef.current = true;
    
    if (settings.streamResponse) {
        await runStreamGeneration(userMsg.text, newHistory, settings, undefined, activeWorld, gameTime);
    } else {
        setIsLoading(true);
        const effectiveWorldData: WorldData = {
            ...activeWorld,
            lsrData: lsrRuntimeDataRef.current, // Sử dụng dữ liệu LSR hiện tại từ Ref để tránh stale
            gameTime: gameTime,
            savedState: {
                history: newHistory,
                turnCount: turnCount,
                gameTime: gameTime
            },
            config: {
                ...activeWorld.config,
                rules: dynamicRules
            }
        };

        const result = await gameplayAiService.generateStoryTurn(
            userMsg.text,
            newHistory, 
            effectiveWorldData,
            settings,
            tawaPresetConfig,
            gameTime
        );
        if (result.usage?.totalTokenCount) {
            updateTokenHistory(result.usage.totalTokenCount, result.text);
        }
        processAIResponse(result.text, false, gameTime);
        
        // Trigger Memory Archiving check
    }
  };

  // Fetch saves when history modal opens
  useEffect(() => {
    if (showHistoryModal) {
      const fetchSaves = async () => {
        try {
          const saves = await dbService.getAllSaves();
          const autosaves = saves.filter(s => s.id.startsWith('autosave-')).sort((a, b) => b.updatedAt - a.updatedAt);
          const manuals = saves.filter(s => s.id.startsWith('manual-')).sort((a, b) => b.updatedAt - a.updatedAt);
          const initials = saves.filter(s => s.id.startsWith('initial-')).sort((a, b) => b.updatedAt - a.updatedAt);
          setAutosaveList(autosaves);
          setManualSaveList(manuals);
          setInitialSaveList(initials);
        } catch (err) {
          console.error("Failed to fetch saves", err);
        }
      };
      fetchSaves();
    }
  }, [showHistoryModal]);

  const handleLoadSave = (save: SaveFile) => {
    if (!save.data) return;
    const worldData = save.data as WorldData;
    if (!worldData.savedState) {
        return;
    }
    
    // Restore state
    const time = worldData.savedState.gameTime || INITIAL_GAME_TIME;
    setGameTime(time);
    gameTimeRef.current = time;
    
    setTurnCount(worldData.savedState.turnCount);
    turnCountRef.current = worldData.savedState.turnCount;
    
    setHistory(worldData.savedState.history);
    historyRef.current = worldData.savedState.history;
    
    setLsrRuntimeData(worldData.lsrData || {});
    lsrRuntimeDataRef.current = worldData.lsrData || {};
    
    // Mark as ready
    isReadyRef.current = true;
    
    // Update activeWorld in parent
    if (onUpdateWorld) {
        onUpdateWorld(worldData);
    }
    
    setShowHistoryModal(false);
  };

  const handleDeleteSave = async (id: string) => {
      try {
          await dbService.deleteSave(id);
          // Refresh lists
          const saves = await dbService.getAllSaves();
          setAutosaveList(saves.filter(s => s.id.startsWith('autosave-')).sort((a, b) => b.updatedAt - a.updatedAt));
          setManualSaveList(saves.filter(s => s.id.startsWith('manual-')).sort((a, b) => b.updatedAt - a.updatedAt));
          setInitialSaveList(saves.filter(s => s.id.startsWith('initial-')).sort((a, b) => b.updatedAt - a.updatedAt));
      } catch (err) {
          console.error("Failed to delete save", err);
      }
  };

    const handleRegenerate = async (msgIndex: number) => {
        if (isLoading || !activeWorld || !settings) return;
        
        // Determine context: history up to msgIndex - 1 (the user message triggering this)
        const prevHistory = history.slice(0, msgIndex);
        const userTriggerMsg = history[msgIndex - 1];
        
        // CRITICAL FIX: Use the gameTime from the user message as the starting point for regeneration
        // This prevents time from "stacking" or advancing incorrectly when retrying.
        const startTime = userTriggerMsg?.gameTime || gameTime;

        // Handle Turn 0 correctly: if msgIndex is 0, it's the opening narrative
        const userInput = msgIndex === 0 ? "Hãy bắt đầu câu chuyện." : (userTriggerMsg?.text || "Continue");

        // Force auto-scroll on regenerate
        shouldAutoScrollRef.current = true;

        if (settings.streamResponse) {
            // Pass the correct startTime to runStreamGeneration
            await runStreamGeneration(userInput, history, settings, msgIndex, activeWorld, startTime);
        } else {
            setIsLoading(true);
            const effectiveWorldData: WorldData = {
                ...activeWorld,
                lsrData: lsrRuntimeDataRef.current,
                config: {
                    ...activeWorld.config,
                    rules: dynamicRules
                }
            };

            const result = await gameplayAiService.generateStoryTurn(
                userInput,
                prevHistory,
                effectiveWorldData,
                settings,
                tawaPresetConfig,
                startTime // Use startTime here too
            );
            
            if (result.usage?.totalTokenCount) {
                updateTokenHistory(result.usage.totalTokenCount, result.text);
            }
            
            // updateMessageSwipes needs to handle time sync correctly
            updateMessageSwipes(msgIndex, result.text, startTime);
            setIsLoading(false);
        }
    };

  const runStreamGeneration = useCallback(async (
      userInput: string, 
      currentHistory: ChatMessage[], 
      currentSettings: AppSettings,
      regenerateIndex?: number,
      world?: WorldData,
      time?: GameTime
  ) => {
      setIsLoading(true);
      const effectiveWorldData: WorldData = {
          ...(world || activeWorldRef.current!),
          lsrData: lsrRuntimeDataRef.current, // Sử dụng dữ liệu LSR hiện tại từ state
          gameTime: time || gameTimeRef.current,
          savedState: {
              history: currentHistory,
              turnCount: turnCountRef.current,
              gameTime: time || gameTimeRef.current
          },
          config: {
              ...(world || activeWorldRef.current!).config,
              rules: dynamicRulesRef.current
          }
      };

        const workingHistory = regenerateIndex !== undefined 
            ? [...currentHistory.slice(0, regenerateIndex + 1)] 
            : [...currentHistory];
        let targetIndex = regenerateIndex;

        // If NOT regenerating, create a placeholder message first
        if (targetIndex === undefined) {
            const placeholderMsg: ChatMessage = {
                role: 'model',
                text: "",
                timestamp: Date.now(),
                gameTime: time || gameTimeRef.current,
                swipes: [""],
                swipeIndex: 0,
                choices: [],
                turnNumber: currentHistory.length === 0 ? 0 : turnCountRef.current + 1,
                userAction: currentHistory.length === 0 ? undefined : userInput
            };
            workingHistory.push(placeholderMsg);
            targetIndex = workingHistory.length - 1;
            
            // Update state with placeholder
            setHistory([...workingHistory]);
        } else {
            // If regenerating, prepare the new swipe slot
            // workingHistory[targetIndex] should now exist because we passed the full history
            const msg = { ...(workingHistory[targetIndex] || {}) } as ChatMessage;
            
            // Ensure role and basic properties are present
            if (!msg.role) msg.role = 'model';
            
            const newSwipes = [...(msg.swipes || [msg.text || ""]), ""]; // Add empty slot
            msg.swipes = newSwipes;
            msg.swipeIndex = newSwipes.length - 1;
            msg.text = ""; // Clear current text for streaming visual
            
            // Ensure turn info is present even for legacy messages being regenerated
            if (msg.turnNumber === undefined) {
                msg.turnNumber = targetIndex === 0 ? 0 : turnCountRef.current;
            }
            if (msg.userAction === undefined && targetIndex > 0) {
                msg.userAction = userInput;
            }
            
            workingHistory[targetIndex] = msg;
            // Update state for visual feedback
            setHistory([...workingHistory]);
        }

        // Small delay to ensure state update (optional but safe)
        await new Promise(r => setTimeout(r, 0));

        const stream = gameplayAiService.generateStoryTurnStream(
            userInput,
            regenerateIndex !== undefined ? currentHistory.slice(0, regenerateIndex) : currentHistory,
            effectiveWorldData,
            currentSettings,
            tawaPresetConfigRef.current,
            time || gameTimeRef.current
        );

      let accumulatedText = "";
      let lastTokenCount = 0;
      let lastUIUpdateTime = 0;
      const UI_UPDATE_INTERVAL = 150; // Tần suất cập nhật UI (ms) - Giúp ổn định khi chuyển tab

      for await (const chunk of stream) {
          if (typeof chunk === 'string') {
              accumulatedText += chunk;
          } else {
              if (chunk.text) accumulatedText += chunk.text;
              if (chunk.usageMetadata?.totalTokenCount) {
                  lastTokenCount = chunk.usageMetadata.totalTokenCount;
              }
          }
          
          const now = Date.now();
          // Chỉ cập nhật UI nếu đã qua khoảng thời gian chỉ định hoặc là chunk cuối (thông qua việc kết thúc loop)
          if (now - lastUIUpdateTime > UI_UPDATE_INTERVAL) {
              if (targetIndex !== undefined && workingHistory[targetIndex]) {
                  const msg = { ...workingHistory[targetIndex] };
                  
                  const swipes = [...(msg.swipes || [""])];
                  const currentSwipeIdx = msg.swipeIndex || 0;
                  swipes[currentSwipeIdx] = accumulatedText;
                  
                  const branchesContent = extractTagContent(accumulatedText, 'branches') || 
                                          extractTagContent(accumulatedText, 'choices') || 
                                          extractTagContent(accumulatedText, 'actions');
                  const choicesList = parseChoices(branchesContent);

                  msg.swipes = swipes;
                  msg.text = accumulatedText;
                  msg.choices = choicesList;
                  
                  workingHistory[targetIndex] = msg;
                  setHistory([...workingHistory]);
                  lastUIUpdateTime = now;
              }
          }
      }

      // ĐẢM BẢO CẬP NHẬT LẦN CUỐI CÙNG KHI KẾT THÚC LUỒNG
      if (targetIndex !== undefined && workingHistory[targetIndex]) {
          const msg = { ...workingHistory[targetIndex] };
          const swipes = [...(msg.swipes || [""])];
          const currentSwipeIdx = msg.swipeIndex || 0;
          swipes[currentSwipeIdx] = accumulatedText;
          
          const branchesContent = extractTagContent(accumulatedText, 'branches') || 
                                  extractTagContent(accumulatedText, 'choices') || 
                                  extractTagContent(accumulatedText, 'actions');
          const choicesList = parseChoices(branchesContent);

          msg.swipes = swipes;
          msg.text = accumulatedText;
          msg.choices = choicesList;
          
          workingHistory[targetIndex] = msg;
          setHistory([...workingHistory]);
      }

      // Update token history once after stream completes
      if (lastTokenCount > 0) {
          updateTokenHistory(lastTokenCount, accumulatedText);
      }

      // Finalize parsing (Branches/Choices and Time)
      let finalTime = time || gameTimeRef.current;

      // Trích xuất thời gian tiêu tốn hoặc thiết lập lại thời gian từ AI
      const setTimeStr = extractTagContent(accumulatedText, 'set_time');
      if (setTimeStr) {
          const parts = setTimeStr.split('|').map(p => parseInt(p.trim(), 10));
          if (parts.length === 5 && !parts.some(isNaN)) {
              finalTime = {
                  year: parts[0],
                  month: parts[1],
                  day: parts[2],
                  hour: parts[3],
                  minute: parts[4]
              };
          }
      } else {
          const timeCostStr = extractTagContent(accumulatedText, 'time_cost');
          let timeCost = parseInt(timeCostStr || '1', 10); // Mặc định 1 phút nếu không có thẻ
          if (isNaN(timeCost) || timeCost < 1) timeCost = 1; // Đảm bảo tối thiểu 1 phút
          finalTime = advanceTime(finalTime, timeCost);
      }

      setGameTime(finalTime);
      
      // Extract incrementalSummary
      const incrementalSummary = extractTagContent(accumulatedText, 'incrementalSummary');
      
      // Task: Parse LSR Data for immediate sync after stream
      const tableStored = extractTagContent(accumulatedText, 'table_stored');
      let nextLsrData = lsrRuntimeDataRef.current;
      if (tableStored) {
          // console.log("GameplayScreen: Detected <table_stored> tag. Parsing...");
          const parsedData = LsrParser.parseLsrString(tableStored);
          // Guard: Only overwrite if we actually found some data in the tag
          if (Object.keys(parsedData).length > 0) {
              nextLsrData = parsedData;
              // console.log("GameplayScreen: Parsed LSR Data from <table_stored>:", nextLsrData);
              setLsrRuntimeData(nextLsrData);
          } else {
              console.warn("GameplayScreen: <table_stored> was present but empty or unparseable. Keeping current data.");
          }
      } else {
          const tableEdit = extractTagContent(accumulatedText, 'tableEdit');
          if (tableEdit) {
              // console.log("GameplayScreen: Detected <tableEdit> tag. Merging edits...");
              const parsedEdits = LsrParser.parseLsrString(tableEdit);
              if (Object.keys(parsedEdits).length > 0) {
                  // console.log("GameplayScreen: Parsed LSR Edits:", parsedEdits);
                  nextLsrData = LsrParser.mergeLsrData(lsrRuntimeDataRef.current, parsedEdits);
                  setLsrRuntimeData(nextLsrData);
              }
          }
      }
      
      // CRITICAL: Calculate finalHistory EXPLICITLY from workingHistory
      if (targetIndex !== undefined && workingHistory[targetIndex]) {
          const msg = { ...workingHistory[targetIndex] };
          
          const branchesContent = extractTagContent(accumulatedText, 'branches') || 
                                  extractTagContent(accumulatedText, 'choices') || 
                                  extractTagContent(accumulatedText, 'actions');
          const choicesList = parseChoices(branchesContent);
          
          msg.choices = choicesList;
          msg.gameTime = finalTime;
          msg.incrementalSummary = incrementalSummary;
          msg.text = accumulatedText; // Ensure text is fully captured
          
          workingHistory[targetIndex] = msg;
      }
      
      const finalHistory = [...workingHistory];
      setHistory(finalHistory);

      if (targetIndex !== undefined && regenerateIndex === undefined) {
          // Only increment turnCount if it's NOT the initial message (Turn 0)
          const isInitial = currentHistory.length === 0;
          const newTurnCount = isInitial ? turnCountRef.current : turnCountRef.current + 1;
          
          if (!isInitial) {
              setTurnCount(newTurnCount);
          }
          
          // console.log(`GameplayScreen: Finalizing Turn ${newTurnCount}. History length: ${finalHistory.length}`);
          
          // Trigger Autosave after stream completes
          triggerAutosave(finalHistory, newTurnCount, finalTime, nextLsrData);
          // Sync state after stream completes
          syncWorldState(finalHistory, newTurnCount, finalTime, nextLsrData, incrementalSummary);
      } else {
          // If regenerating or no target, just sync current state
          syncWorldState(finalHistory, turnCountRef.current, finalTime, nextLsrData, incrementalSummary);
      }
      setIsLoading(false);
      
      // Trigger Memory Archiving check after stream
  }, [syncWorldState, triggerAutosave]);

  const processAIResponse = useCallback((responseText: string, initial = false, time?: GameTime) => {
    const branchesContent = extractTagContent(responseText, 'branches') || 
                            extractTagContent(responseText, 'choices') || 
                            extractTagContent(responseText, 'actions');
    const choicesList = parseChoices(branchesContent);

    // Trích xuất thời gian tiêu tốn hoặc thiết lập lại thời gian từ AI
    const setTimeStr = extractTagContent(responseText, 'set_time');
    let updatedTime = time || gameTimeRef.current;

    if (setTimeStr) {
        const parts = setTimeStr.split('|').map(p => parseInt(p.trim(), 10));
        if (parts.length === 5 && !parts.some(isNaN)) {
            updatedTime = {
                year: parts[0],
                month: parts[1],
                day: parts[2],
                hour: parts[3],
                minute: parts[4]
            };
        }
    } else {
        const timeCostStr = extractTagContent(responseText, 'time_cost');
        let timeCost = parseInt(timeCostStr || (initial ? '0' : '1'), 10);
        if (!initial && (isNaN(timeCost) || timeCost < 1)) timeCost = 1; // Đảm bảo tối thiểu 1 phút cho hành động (trừ mở đầu)
        
        if (timeCost > 0 || initial) {
            updatedTime = advanceTime(updatedTime, timeCost);
        }
    }
    
    setGameTime(updatedTime);

    // Extract incrementalSummary
    const incrementalSummary = extractTagContent(responseText, 'incrementalSummary');
    
    const modelMsg: ChatMessage = { 
        role: 'model', 
        text: responseText, 
        timestamp: Date.now(),
        gameTime: updatedTime,
        choices: choicesList,
        swipes: [responseText],
        swipeIndex: 0,
        turnNumber: initial ? 0 : turnCountRef.current + 1,
        userAction: initial ? undefined : lastActionRef.current,
        incrementalSummary: incrementalSummary
    };
    
    const newHistory = [...historyRef.current, modelMsg];
    // Task: Parse LSR Data for immediate sync
    const tableStored = extractTagContent(responseText, 'table_stored');
    let nextLsrData = lsrRuntimeDataRef.current;
    if (tableStored) {
        // console.log("GameplayScreen (processAIResponse): Detected <table_stored>. Parsing...");
        const parsedData = LsrParser.parseLsrString(tableStored);
        if (Object.keys(parsedData).length > 0) {
            nextLsrData = parsedData;
            // console.log("GameplayScreen (processAIResponse): Parsed LSR Data:", nextLsrData);
            setLsrRuntimeData(nextLsrData);
        } else {
            console.warn("GameplayScreen (processAIResponse): <table_stored> was empty or unparseable.");
        }
    } else {
        const tableEdit = extractTagContent(responseText, 'tableEdit');
        if (tableEdit) {
            // console.log("GameplayScreen (processAIResponse): Detected <tableEdit>. Merging edits...");
            const parsedEdits = LsrParser.parseLsrString(tableEdit);
            if (Object.keys(parsedEdits).length > 0) {
                // console.log("GameplayScreen (processAIResponse): Parsed LSR Edits:", parsedEdits);
                nextLsrData = LsrParser.mergeLsrData(lsrRuntimeDataRef.current, parsedEdits);
                setLsrRuntimeData(nextLsrData);
            }
        }
    }

    setHistory(newHistory);
    
    if (!initial) {
        const newTurnCount = turnCountRef.current + 1;
        setTurnCount(newTurnCount);
        // Sync & Autosave
        syncWorldState(newHistory, newTurnCount, updatedTime, nextLsrData, incrementalSummary);
        triggerAutosave(newHistory, newTurnCount, updatedTime, nextLsrData);
    } else {
        // Initial message (Opening)
        syncWorldState(newHistory, turnCountRef.current, updatedTime, nextLsrData, incrementalSummary);
        triggerAutosave(newHistory, turnCountRef.current, updatedTime, nextLsrData);
    }
    setIsLoading(false);
  }, [syncWorldState, triggerAutosave]);

  const handleSendInitial = useCallback(async (currentSettings: AppSettings, world: WorldData, time: GameTime) => {
    setIsLoading(true);
    // Enable auto scroll for initial load
    shouldAutoScrollRef.current = true;
    
    if (currentSettings.streamResponse) {
        await runStreamGeneration("Hãy bắt đầu câu chuyện.", [], currentSettings, undefined, world, time);
    } else {
        const opening = await gameplayAiService.generateStoryTurn(
             "Hãy bắt đầu câu chuyện.", 
             [], 
             world, 
             currentSettings,
             tawaPresetConfig,
             time
         );
         processAIResponse(opening.text, true, time);
    }
 }, [runStreamGeneration, processAIResponse, tawaPresetConfig]);

  // --- Initial Load ---
  useEffect(() => {
    // If world object reference changes, allow re-initialization
    // Use a unique property or the object itself to detect a "new session"
    if (activeWorld && activeWorld !== lastWorldRef.current) {
        // console.log("GameplayScreen: New world detected, resetting initialization.");
        initializedRef.current = false;
        lastWorldRef.current = activeWorld;
        initialStartedRef.current = false;
        isReadyRef.current = false; // Reset ready state for new world
    }

    if (initializedRef.current) return;
    initializedRef.current = true; 
    
    const init = async () => {
      // console.log("GameplayScreen: Starting initialization...");
      const s = await dbService.getSettings();
      setSettings(s);

      // Load LSR Definitions
      setLsrTables(LsrParser.parseDefinitions());

      if (activeWorld) {
          // Sync initial rules from world config
          setDynamicRules(activeWorld.config.rules || []);
          
          // Load LSR Data
          if (activeWorld.lsrData) {
              setLsrRuntimeData(activeWorld.lsrData);
          }

          const worldDataWithState = activeWorld as WorldData;
          if (worldDataWithState.savedState && worldDataWithState.savedState.history.length > 0) {
              // console.log("GameplayScreen: Loading saved state, history length:", worldDataWithState.savedState.history.length);
              setHistory(worldDataWithState.savedState.history);
              historyRef.current = worldDataWithState.savedState.history;
              
              setTurnCount(worldDataWithState.savedState.turnCount);
              turnCountRef.current = worldDataWithState.savedState.turnCount;
              
              if (worldDataWithState.savedState.gameTime) {
                  setGameTime(worldDataWithState.savedState.gameTime);
                  gameTimeRef.current = worldDataWithState.savedState.gameTime;
              }
              
              // Restore AI Monitor state
              if (worldDataWithState.savedState.aiMonitor) {
                  setTokenHistory(worldDataWithState.savedState.aiMonitor.tokenHistory);
                  setTotalTokens(worldDataWithState.savedState.aiMonitor.totalTokens);
                  setLastTurnTotalTime(worldDataWithState.savedState.aiMonitor.lastTurnTotalTime);
              } else {
                  setTokenHistory([]);
                  setTotalTokens(0);
                  setLastTurnTotalTime(0);
              }
              
              setTimeout(() => {
                  vectorService.vectorizeAllHistory(worldDataWithState.savedState!.history, settings);
              }, 1000);

              // Mark as ready AFTER state is set
              isReadyRef.current = true;
              // console.log("GameplayScreen: Initialization complete (Loaded Save).");
          } else {
            // console.log("GameplayScreen: No saved state found, starting new game.");
            // New world or empty history: Reset AI Monitor
            setTokenHistory([]);
            setTotalTokens(0);
            setLastTurnTotalTime(0);

            if (s && !initialStartedRef.current) {
                // Initial Start: Generate opening
                initialStartedRef.current = true;
                const initialTime = worldDataWithState.gameTime || INITIAL_GAME_TIME;
                setGameTime(initialTime);
                
                // Mark as ready BEFORE initial generation so it can sync
                isReadyRef.current = true;
                
                // Trigger Initial Save (Bản lưu lượt 0) BEFORE opening generation
                await triggerInitialSave(worldDataWithState, initialTime);
                
                handleSendInitial(s, worldDataWithState, initialTime);
                // console.log("GameplayScreen: Initialization complete (New Game).");
            }
          }
      }
    };
    init();
  }, [activeWorld, handleSendInitial, settings, triggerInitialSave]);

  const updateMessageSwipes = (index: number, newText: string, overrideTime?: GameTime) => {
       // Task: Parse LSR Data for immediate sync during swipe/regen
       const tableStored = extractTagContent(newText, 'table_stored');
       let nextLsrData = lsrRuntimeDataRef.current;
       if (tableStored) {
           // console.log("GameplayScreen (updateMessageSwipes): Detected <table_stored>. Parsing...");
           nextLsrData = LsrParser.parseLsrString(tableStored);
           // console.log("GameplayScreen (updateMessageSwipes): Parsed LSR Data:", nextLsrData);
           setLsrRuntimeData(nextLsrData);
       } else {
           const tableEdit = extractTagContent(newText, 'tableEdit');
           if (tableEdit) {
               // console.log("GameplayScreen (updateMessageSwipes): Detected <tableEdit>. Parsing edits...");
               const parsedEdits = LsrParser.parseLsrString(tableEdit);
               // console.log("GameplayScreen (updateMessageSwipes): Parsed LSR Edits:", parsedEdits);
               nextLsrData = LsrParser.mergeLsrData(lsrRuntimeDataRef.current, parsedEdits);
               setLsrRuntimeData(nextLsrData);
           }
       }

       setHistory(prev => {
            // Truncate history to the current index to ensure story divergence
            const updated = prev.slice(0, index + 1);
            const msg = { ...(updated[index] || {}) } as ChatMessage;
            
            // Ensure role is present
            if (!msg.role) msg.role = 'model';
            
            const branchesContent = extractTagContent(newText, 'branches') || 
                                    extractTagContent(newText, 'choices') || 
                                    extractTagContent(newText, 'actions');
            const choicesList = parseChoices(branchesContent);

            const currentSwipes = msg.swipes || [msg.text];
            const newSwipes = [...currentSwipes, newText];
            
            msg.swipes = newSwipes;
            msg.swipeIndex = newSwipes.length - 1;
            msg.text = newText;
            msg.choices = choicesList; // Update choices to latest generation

            // Calculate final time for this swipe
            let finalTime = overrideTime || gameTime;
            const setTimeStr = extractTagContent(newText, 'set_time');
            if (setTimeStr) {
                const parts = setTimeStr.split('|').map(p => parseInt(p.trim(), 10));
                if (parts.length === 5 && !parts.some(isNaN)) {
                    finalTime = { year: parts[0], month: parts[1], day: parts[2], hour: parts[3], minute: parts[4] };
                }
            } else {
                const timeCostStr = extractTagContent(newText, 'time_cost');
                let timeCost = parseInt(timeCostStr || '1', 10);
                if (isNaN(timeCost) || timeCost < 1) timeCost = 1;
                finalTime = advanceTime(finalTime, timeCost);
            }
            
            msg.gameTime = finalTime;
            setGameTime(finalTime);
            
            // Extract incrementalSummary
            const incrementalSummary = extractTagContent(newText, 'incrementalSummary');
            msg.incrementalSummary = incrementalSummary;

            // Ensure turn info is present even for legacy messages being regenerated
            if (msg.turnNumber === undefined) {
                msg.turnNumber = index === 0 ? 0 : turnCount;
            }
            if (msg.userAction === undefined && index > 0) {
                msg.userAction = updated[index - 1].text;
            }

            updated[index] = msg;
            
            // Sync state back to parent
            syncWorldState(updated, turnCount, finalTime, nextLsrData, incrementalSummary);
            return updated;
       });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    } else if (e.key === 'ArrowUp') {
      // Chỉ kích hoạt nếu con trỏ đang ở đầu textarea (vị trí 0)
      const target = e.target as HTMLTextAreaElement;
      if (target.selectionStart === 0) {
        // Điều hướng ngược lại lịch sử (lên trên)
        const userMessages = history.filter(m => m.role === 'user').map(m => m.text).reverse();
        if (userMessages.length > 0) {
          const nextIndex = historyIndex + 1;
          if (nextIndex < userMessages.length) {
            e.preventDefault();
            if (historyIndex === -1) {
              setTempInputValue(inputValue);
            }
            setHistoryIndex(nextIndex);
            setInputValue(userMessages[nextIndex]);
          }
        }
      }
    } else if (e.key === 'ArrowDown') {
      // Chỉ kích hoạt nếu con trỏ đang ở cuối textarea
      const target = e.target as HTMLTextAreaElement;
      if (target.selectionStart === inputValue.length && historyIndex >= 0) {
        // Điều hướng xuôi theo lịch sử (xuống dưới)
        const userMessages = history.filter(m => m.role === 'user').map(m => m.text).reverse();
        e.preventDefault();
        const nextIndex = historyIndex - 1;
        setHistoryIndex(nextIndex);
        if (nextIndex === -1) {
          setInputValue(tempInputValue);
        } else {
          setInputValue(userMessages[nextIndex]);
        }
      }
    }
  };

  const handleMessageUpdate = (index: number, newText: string) => {
    setHistory(prev => {
        const newHistory = [...prev];
        if (newHistory[index]) {
            // Update raw text
            const msg = { ...newHistory[index] };
            msg.text = newText;

            // Also update the specific swipe if it exists
            if (msg.swipes && msg.swipeIndex !== undefined) {
                const newSwipes = [...msg.swipes];
                newSwipes[msg.swipeIndex] = newText;
                msg.swipes = newSwipes;
            }

            if (msg.role === 'model') {
                const branchesContent = extractTagContent(newText, 'branches') || 
                                        extractTagContent(newText, 'choices') || 
                                        extractTagContent(newText, 'actions');
                msg.choices = parseChoices(branchesContent);
            }
            newHistory[index] = msg;
        }
        return newHistory;
    });
  };

  const handleSwipe = (msgIndex: number, direction: 'prev' | 'next') => {
      const msg = history[msgIndex];
      if (!msg.swipes || msg.swipes.length === 0) return;
      
      const currentIndex = msg.swipeIndex || 0;
      let newIndex = currentIndex;

      if (direction === 'prev') {
          if (currentIndex > 0) newIndex--;
      } else {
          if (currentIndex < msg.swipes.length - 1) {
              newIndex++;
          } else {
              // Trigger Regenerate if at the end
              handleRegenerate(msgIndex);
              return;
          }
      }

      const newText = msg.swipes[newIndex];
      
      // Task: Parse LSR Data for immediate sync during swipe
      const tableStored = extractTagContent(newText, 'table_stored');
      let nextLsrData = lsrRuntimeDataRef.current;
      if (tableStored) {
          // console.log("GameplayScreen (handleSwipe): Detected <table_stored>. Parsing...");
          const parsedData = LsrParser.parseLsrString(tableStored);
          if (Object.keys(parsedData).length > 0) {
              nextLsrData = parsedData;
              setLsrRuntimeData(nextLsrData);
          }
      } else {
          const tableEdit = extractTagContent(newText, 'tableEdit');
          if (tableEdit) {
              // console.log("GameplayScreen (handleSwipe): Detected <tableEdit>. Merging edits...");
              const parsedEdits = LsrParser.parseLsrString(tableEdit);
              if (Object.keys(parsedEdits).length > 0) {
                  nextLsrData = LsrParser.mergeLsrData(lsrRuntimeDataRef.current, parsedEdits);
                  setLsrRuntimeData(nextLsrData);
              }
          }
      }

      // Extract incrementalSummary
      const incrementalSummary = extractTagContent(newText, 'incrementalSummary');

      setHistory(prev => {
          const updated = [...prev];
          const updatedMsg = { ...updated[msgIndex] };
          
          updatedMsg.swipeIndex = newIndex;
          updatedMsg.text = newText;
          
          // Re-parse choices for this specific swipe version
          const branchesContent = extractTagContent(newText, 'branches') || 
                                  extractTagContent(newText, 'choices') || 
                                  extractTagContent(newText, 'actions');
          updatedMsg.choices = parseChoices(branchesContent);
          updatedMsg.incrementalSummary = incrementalSummary;

          updated[msgIndex] = updatedMsg;

          // Sync state back to parent
          syncWorldState(updated, turnCount, gameTime, nextLsrData, incrementalSummary);

          return updated;
      });
  };

  const handleExit = () => {
    onNavigate(GameState.MENU);
  };

  const scrollToTurn = (turnNumber: number) => {
    // Tìm index của tin nhắn đầu tiên thuộc lượt này
    const msgIndex = history.findIndex(m => m.turnNumber === turnNumber);
    if (msgIndex === -1) return;

    // Tính toán trang chứa tin nhắn này
    const targetPage = msgIndex < 11 ? 1 : 1 + Math.ceil((msgIndex - 11) / MESSAGES_PER_PAGE);

    if (targetPage !== currentPage) {
      setCurrentPage(targetPage);
      pendingScrollTurnRef.current = turnNumber;
    } else {
      const element = document.getElementById(`turn-${turnNumber}`);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }
    setLastNavigatedTurn(turnNumber);
    shouldAutoScrollRef.current = false;
  };

  const findCurrentTurnInView = () => {
    if (!scrollViewportRef.current) return null;
    const viewport = scrollViewportRef.current;
    const viewportRect = viewport.getBoundingClientRect();
    const midPoint = viewportRect.top + viewportRect.height / 3; // Check top third

    // Find all turn elements in the current viewport
    const turnElements = Array.from(viewport.querySelectorAll('[id^="turn-"]'));
    let closestTurn = null;
    let minDistance = Infinity;

    for (const el of turnElements) {
      const rect = el.getBoundingClientRect();
      // If the element is visible in the viewport
      if (rect.bottom > viewportRect.top && rect.top < viewportRect.bottom) {
        const distance = Math.abs(rect.top - midPoint);
        if (distance < minDistance) {
          minDistance = distance;
          const turnId = el.id.replace('turn-', '');
          closestTurn = parseInt(turnId, 10);
        }
      }
    }
    return closestTurn;
  };

  const scrollToTop = () => {
    // Lấy danh sách tất cả các số lượt hiện có, sắp xếp tăng dần
    const allTurns = Array.from(new Set(history
        .filter(m => m.turnNumber !== undefined)
        .map(m => m.turnNumber as number)
    )).sort((a, b) => a - b);

    if (allTurns.length === 0) {
      if (scrollViewportRef.current) {
        scrollViewportRef.current.scrollTo({ top: 0, behavior: 'smooth' });
      }
      return;
    }

    // Ưu tiên sử dụng lượt đang hiển thị nếu có
    const currentTurn = findCurrentTurnInView() ?? lastNavigatedTurn;

    let targetTurn: number;
    if (currentTurn === null) {
      // Nếu không xác định được, nhảy đến lượt cuối cùng
      targetTurn = allTurns[allTurns.length - 1];
    } else {
      const currentIndex = allTurns.indexOf(currentTurn);
      if (currentIndex > 0) {
        targetTurn = allTurns[currentIndex - 1];
      } else {
        // Nếu đã ở lượt đầu, quay lại lượt cuối
        targetTurn = allTurns[allTurns.length - 1];
      }
    }

    scrollToTurn(targetTurn);
  };

  const scrollToBottom = () => {
    // Lấy danh sách tất cả các số lượt hiện có, sắp xếp tăng dần
    const allTurns = Array.from(new Set(history
        .filter(m => m.turnNumber !== undefined)
        .map(m => m.turnNumber as number)
    )).sort((a, b) => a - b);

    if (allTurns.length === 0) {
      if (chatEndRef.current) {
        chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
      }
      return;
    }

    // Ưu tiên sử dụng lượt đang hiển thị nếu có
    const currentTurn = findCurrentTurnInView() ?? lastNavigatedTurn;

    let targetTurn: number;
    if (currentTurn === null) {
      // Nếu không xác định được, nhảy đến lượt đầu tiên
      targetTurn = allTurns[0];
    } else {
      const currentIndex = allTurns.indexOf(currentTurn);
      if (currentIndex !== -1 && currentIndex < allTurns.length - 1) {
        targetTurn = allTurns[currentIndex + 1];
      } else {
        // Nếu đã ở lượt cuối, cuộn xuống cuối hẳn
        if (chatEndRef.current) {
          chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
        setLastNavigatedTurn(null);
        shouldAutoScrollRef.current = true;
        return;
      }
    }

    scrollToTurn(targetTurn);
  };

  const handleManualSave = async () => {
    if (!activeWorld) return;
    setIsSaving(true);
    
    const worldName = activeWorld?.world?.worldName || 'Unknown_World';
    const playerName = activeWorld?.player?.name || 'Unknown_Hero';
    const turnCountValue = history.filter(m => m.role === 'user').length;
    
    // 1. Prepare Save Data
    const saveData: WorldData = {
        ...activeWorld,
        lsrData: lsrRuntimeData, // CRITICAL: Include latest LSR data
        savedState: {
            history: history,
            turnCount: turnCountValue,
            gameTime: gameTime,
            aiMonitor: {
                tokenHistory: tokenHistoryRef.current,
                totalTokens: totalTokensRef.current,
                lastTurnTotalTime: lastTurnTotalTimeRef.current
            }
        },
        config: {
            ...activeWorld.config,
            rules: dynamicRules
        }
    };
    
    // 2. Save to Database (Internal)
    // Use deterministic ID for manual save: manual-[worldName]-[turnCount]
    const saveId = `manual-${worldName.replace(/\s+/g, '_')}-${turnCount}`;
    const saveFile: SaveFile = {
        id: saveId,
        name: `${worldName} - Turn ${turnCount} (Manual)`,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        data: saveData
    };

    try {
        await dbService.saveGameState(saveFile);
        
        // 3. Download to Computer
        // Format: ARK_save_[world_name]_[player_name]_[turn_number].json
        const fileName = `ARK_save_${worldName.replace(/\s+/g, '_')}_${playerName.replace(/\s+/g, '_')}_${turnCountValue}.json`;
        
        const blob = new Blob([JSON.stringify(saveData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error("Failed to save game:", error);
    } finally {
        setIsSaving(false);
    }
  };
  
  const toggleStreamResponse = async () => {
      if (!settings) return;
      const newSetting = !settings.streamResponse;
      setSettings({ ...settings, streamResponse: newSetting });
      await dbService.saveSettings({ ...settings, streamResponse: newSetting });
  };

  const handleGoToSettings = () => {
    syncWorldState();
    onNavigate(GameState.SETTINGS);
  };

  const handleUpdateContextConfig = (newConfig: ContextWindowConfig) => {
    if (onUpdateWorld && activeWorld) {
      onUpdateWorld({
        config: {
          ...activeWorld.config,
          contextConfig: newConfig
        }
      });
    }
  };

  const ContextWindowModal = () => {
    if (!activeWorld || !showContextModal) return null;
    
    // Default config if not exists
    const defaultConfig = {
      items: {
        playerProfile: true,
        worldInfo: true,
        longTermMemory: true,
        relevantMemories: true,
        entities: true,
        npcRegistry: true,
        timeSystem: true,
        reinforcement: true
      },
      maxEntities: 10,
      recentHistoryCount: 100
    };

    const config = activeWorld.config.contextConfig || defaultConfig;

    const toggleItem = (key: keyof typeof config.items) => {
      const newConfig = {
        ...config,
        items: {
          ...config.items,
          [key]: !config.items[key]
        }
      };
      handleUpdateContextConfig(newConfig);
    };

    const updateMaxEntities = (val: number) => {
      handleUpdateContextConfig({ ...config, maxEntities: val });
    };

    const updateHistoryCount = (val: number) => {
      handleUpdateContextConfig({ ...config, recentHistoryCount: val });
    };

    return (
      <AnimatePresence>
        {showContextModal && (
          <div className="fixed inset-0 z-[150] flex items-center justify-center bg-black/90 backdrop-blur-md p-2">
            <motion.div 
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              className="bg-stone-200 dark:bg-mystic-950 border border-stone-400 dark:border-slate-800 w-[99vw] h-[99vh] rounded-xl shadow-2xl flex flex-col overflow-hidden"
            >
              {/* Header */}
              <div className="p-4 border-b border-stone-400 dark:border-slate-800 flex justify-between items-center bg-stone-300 dark:bg-mystic-900/80 shrink-0">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-mystic-accent/20 rounded-lg text-mystic-accent">
                    <Database size={24} />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-stone-800 dark:text-slate-100">Cấu hình Cửa sổ Ngữ cảnh</h2>
                    <p className="text-xs text-stone-500 dark:text-slate-500 uppercase tracking-widest font-bold">Kiểm soát dữ liệu gửi cho AI Tawa</p>
                  </div>
                </div>
                <button 
                  onClick={() => setShowContextModal(false)}
                  className="p-2 hover:bg-red-500/20 text-stone-500 hover:text-red-500 rounded-full transition-all"
                >
                  <X size={32} />
                </button>
              </div>

              {/* Body */}
              <div className="flex-1 overflow-y-auto custom-scrollbar p-6 grid grid-cols-1 md:grid-cols-2 gap-8 bg-stone-100 dark:bg-mystic-950">
                {/* Left Column: Toggles */}
                <div className="space-y-6">
                  <h3 className="text-sm font-black text-mystic-accent uppercase tracking-[0.2em] border-b border-mystic-accent/30 pb-2">Thành phần Ngữ cảnh</h3>
                  
                  <div className="space-y-3">
                    {[
                      { key: 'playerProfile', label: 'Hồ sơ nhân vật', desc: 'Thông tin chi tiết về nhân vật của bạn' },
                      { key: 'worldInfo', label: 'Thông tin thế giới', desc: 'Bối cảnh, thể loại và cốt truyện chung' },
                      { key: 'longTermMemory', label: 'Trí nhớ dài hạn (Summary)', desc: 'Bản tóm tắt các sự kiện đã qua' },
                      { key: 'relevantMemories', label: 'Ký ức liên quan (RAG)', desc: 'Các đoạn hội thoại cũ được tìm thấy qua Vector Search' },
                      { key: 'entities', label: 'Thực thể (NPCs/Items)', desc: 'Thông tin về các nhân vật và vật phẩm trong thế giới' },
                      { key: 'npcRegistry', label: 'Danh sách tổng NPC (Registry)', desc: 'Danh sách rút gọn tất cả NPC để AI tham chiếu ID' },
                      { key: 'timeSystem', label: 'Hệ thống thời gian', desc: 'Ngày, tháng, năm và lượt chơi hiện tại' },
                      { key: 'reinforcement', label: 'Chỉ thị củng cố (Reinforcement)', desc: 'Các lệnh ép AI duy trì chất lượng văn phong' },
                    ].map((item) => (
                      <div key={item.key} className="flex items-center justify-between p-4 bg-stone-200 dark:bg-slate-900/50 rounded-xl border border-stone-300 dark:border-slate-800 hover:border-mystic-accent/50 transition-all group">
                        <div className="flex-1">
                          <h4 className="text-sm font-bold text-stone-800 dark:text-slate-200">{item.label}</h4>
                          <p className="text-[10px] text-stone-500 dark:text-slate-500 mt-0.5">{item.desc}</p>
                        </div>
                        <button 
                          onClick={() => toggleItem(item.key as keyof ContextWindowConfig['items'])}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${config.items[item.key as keyof typeof config.items] ? 'bg-mystic-accent' : 'bg-stone-400 dark:bg-slate-700'}`}
                        >
                          <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${config.items[item.key as keyof typeof config.items] ? 'translate-x-6' : 'translate-x-1'}`} />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Right Column: Numeric Limits */}
                <div className="space-y-6">
                  <h3 className="text-sm font-black text-mystic-accent uppercase tracking-[0.2em] border-b border-mystic-accent/30 pb-2">Giới hạn Số lượng</h3>
                  
                  <div className="space-y-6">
                    {/* Max Entities */}
                    <div className="p-5 bg-stone-200 dark:bg-slate-900/50 rounded-xl border border-stone-300 dark:border-slate-800">
                      <div className="flex justify-between items-center mb-4">
                        <div>
                          <h4 className="text-sm font-bold text-stone-800 dark:text-slate-200">Số lượng Thực thể tối đa (NPCs)</h4>
                          <p className="text-[10px] text-stone-500 dark:text-slate-500 mt-0.5">Giới hạn số lượng NPC/Vật phẩm gửi cho AI mỗi lượt</p>
                        </div>
                        <div className="text-2xl font-black text-mystic-accent">{config.maxEntities}</div>
                      </div>
                      <input 
                        type="range" 
                        min="1" 
                        max="50" 
                        value={config.maxEntities} 
                        onChange={(e) => updateMaxEntities(parseInt(e.target.value))}
                        className="w-full h-2 bg-stone-300 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-mystic-accent"
                      />
                      <div className="flex justify-between text-[10px] text-stone-500 mt-2 font-bold">
                        <span>1 NPC</span>
                        <span>50 NPCs</span>
                      </div>
                    </div>

                    {/* Recent History Count */}
                    <div className="p-5 bg-stone-200 dark:bg-slate-900/50 rounded-xl border border-stone-300 dark:border-slate-800">
                      <div className="flex justify-between items-center mb-4">
                        <div>
                          <h4 className="text-sm font-bold text-stone-800 dark:text-slate-200">Lịch sử gần đây (Recent History)</h4>
                          <p className="text-[10px] text-stone-500 dark:text-slate-500 mt-0.5">Số lượng tin nhắn gần nhất AI sẽ đọc trực tiếp</p>
                        </div>
                        <div className="flex items-center gap-2">
                          <input 
                            type="number" 
                            min="1" 
                            max="500" 
                            value={config.recentHistoryCount} 
                            onChange={(e) => updateHistoryCount(parseInt(e.target.value) || 1)}
                            className="w-16 bg-stone-300 dark:bg-slate-800 border border-stone-400 dark:border-slate-700 rounded px-2 py-1 text-center font-black text-mystic-accent outline-none focus:border-mystic-accent"
                          />
                          <span className="text-xs font-bold text-stone-500">tin nhắn</span>
                        </div>
                      </div>
                      <p className="text-[10px] italic text-amber-600 dark:text-amber-500/70 bg-amber-500/5 p-2 rounded border border-amber-500/20">
                        * Mặc định là 100. Số lượng càng cao AI càng nhớ tốt các sự kiện vừa xảy ra, nhưng sẽ tốn nhiều Token hơn.
                      </p>
                    </div>
                  </div>

                  {/* Info Box */}
                  <div className="p-6 bg-mystic-accent/5 border border-mystic-accent/20 rounded-2xl space-y-3">
                    <h4 className="text-xs font-black text-mystic-accent uppercase tracking-widest flex items-center gap-2">
                      <Zap size={14} /> Tại sao cần cấu hình này?
                    </h4>
                    <p className="text-xs text-stone-600 dark:text-slate-400 leading-relaxed">
                      Cửa sổ ngữ cảnh là "bộ nhớ tạm thời" của AI. Bằng cách tắt các mục không cần thiết hoặc giảm số lượng tin nhắn lịch sử, bạn có thể:
                    </p>
                    <ul className="text-xs text-stone-600 dark:text-slate-400 space-y-1 list-disc pl-4">
                      <li>Tiết kiệm Token (giảm chi phí/tăng tốc độ phản hồi).</li>
                      <li>Tránh việc AI bị "loãng" thông tin bởi quá nhiều NPC phụ.</li>
                      <li>Tập trung sự chú ý của AI vào các thành phần quan trọng nhất.</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Footer */}
              <div className="p-4 border-t border-stone-400 dark:border-slate-800 bg-stone-300 dark:bg-mystic-900/80 flex justify-center shrink-0">
                <Button 
                  onClick={() => setShowContextModal(false)}
                  className="px-12 py-3 bg-mystic-accent text-mystic-900 font-black uppercase tracking-widest hover:bg-sky-400 shadow-[0_0_20px_rgba(56,189,248,0.3)]"
                >
                  Xác nhận & Đóng
                </Button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    );
  };

  const renderSidebarContent = () => {
      const totalPages = history.length <= 11 ? 1 : 1 + Math.ceil((history.length - 11) / MESSAGES_PER_PAGE);
      return (
      <div className="h-full flex flex-col bg-stone-300 dark:bg-mystic-900 shadow-xl">
          <div className="p-1 border-b border-stone-400 dark:border-slate-800 bg-stone-400/50 dark:bg-mystic-800/50 shrink-0 space-y-1">
              <button onClick={() => setShowCharModal(true)} className="w-full flex items-center gap-2 p-1.5 bg-stone-200 dark:bg-slate-800/50 hover:bg-stone-400 dark:hover:bg-slate-700 border border-stone-400 dark:border-slate-700 rounded transition-all group">
                  <div className="w-8 h-8 rounded-full bg-stone-300 dark:bg-slate-900 border border-stone-400 dark:border-slate-600 flex items-center justify-center shrink-0 group-hover:border-mystic-accent"><User className="text-mystic-accent" size={16}/></div>
                  <div className="text-left"><h3 className="font-normal text-stone-800 dark:text-slate-200 text-xs truncate">{activeWorld.player.name}</h3></div>
              </button>
              <button onClick={() => setShowGlobalModal(true)} className="w-full flex items-center gap-2 p-1.5 bg-stone-200 dark:bg-slate-800/50 hover:bg-stone-400 dark:hover:bg-slate-700 border border-stone-400 dark:border-slate-700 rounded transition-all group">
                  <div className="w-8 h-8 rounded-full bg-stone-300 dark:bg-slate-900 border border-stone-400 dark:border-slate-600 flex items-center justify-center shrink-0 group-hover:border-green-400"><Globe className="text-green-600 dark:text-green-400" size={16}/></div>
                  <div className="text-left"><h3 className="font-normal text-stone-800 dark:text-slate-200 text-xs">Thông tin toàn cục</h3></div>
              </button>
              <button onClick={() => setShowHistoryModal(true)} className="w-full flex items-center gap-2 p-1.5 bg-stone-200 dark:bg-slate-800/50 hover:bg-stone-400 dark:hover:bg-slate-700 border border-stone-400 dark:border-slate-700 rounded transition-all group">
                  <div className="w-8 h-8 rounded-full bg-stone-300 dark:bg-slate-900 border border-stone-400 dark:border-slate-600 flex items-center justify-center shrink-0 group-hover:border-blue-400"><History className="text-blue-600 dark:text-blue-400" size={16}/></div>
                  <div className="text-left"><h3 className="font-normal text-stone-800 dark:text-slate-200 text-xs">Lịch Sử & Load Save</h3></div>
              </button>
              <button onClick={() => setShowImageLibrary(true)} className="w-full flex items-center gap-2 p-1.5 bg-stone-200 dark:bg-slate-800/50 hover:bg-stone-400 dark:hover:bg-slate-700 border border-stone-400 dark:border-slate-700 rounded transition-all group">
                  <div className="w-8 h-8 rounded-full bg-stone-300 dark:bg-slate-900 border border-stone-400 dark:border-slate-600 flex items-center justify-center shrink-0 group-hover:border-mystic-accent"><ImageIcon className="text-mystic-accent" size={16}/></div>
                  <div className="text-left"><h3 className="font-normal text-stone-800 dark:text-slate-200 text-xs">Thư Viện Ảnh</h3></div>
              </button>
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar p-1 space-y-2">
              {/* Mobile Controls Section */}
              <div className="md:hidden p-2 space-y-3 bg-stone-400/20 dark:bg-slate-800/20 rounded-lg border border-stone-400/50 dark:border-slate-700/50 mb-2">
                  <div className="text-[10px] font-bold text-stone-500 dark:text-slate-500 uppercase tracking-wider mb-1 flex items-center gap-2">
                      <Zap size={12} /> Điều khiển nhanh
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                      {/* Thử Lại Button */}
                      <Button 
                          variant="ghost" 
                          onClick={() => {
                              const lastModelIdx = [...history].reverse().findIndex(m => m.role === 'model');
                              if (lastModelIdx !== -1) {
                                  const actualIdx = history.length - 1 - lastModelIdx;
                                  handleRegenerate(actualIdx);
                              }
                          }} 
                          disabled={isLoading || !history.some(m => m.role === 'model')} 
                          className="h-10 text-[10px] font-bold uppercase tracking-tighter border border-stone-400 dark:border-slate-700 hover:border-mystic-accent/50 flex items-center justify-center gap-2"
                      >
                          <RefreshCw size={14} className={isLoading ? "animate-spin" : ""} />
                          Thử Lại
                      </Button>

                      {/* Toggle Input Button */}
                      <button 
                          onClick={() => setIsInputCollapsed(!isInputCollapsed)}
                          className={`h-10 rounded border transition-all flex items-center justify-center gap-2 shadow-sm ${
                              isInputCollapsed 
                              ? 'bg-mystic-accent/10 border-mystic-accent/30 text-mystic-accent' 
                              : 'bg-stone-200 dark:bg-slate-800 border-stone-400 dark:border-slate-700 text-stone-500'
                          }`}
                      >
                          {isInputCollapsed ? <Maximize2 size={16} /> : <Minimize2 size={16} />}
                          <span className="text-[10px] font-bold uppercase">
                              {isInputCollapsed ? 'Mở Rộng' : 'Thu Gọn'}
                          </span>
                      </button>
                  </div>

                  <div className="flex items-center justify-between gap-2">
                      {/* Pagination Group */}
                      <div className="flex items-center h-10 bg-stone-200 dark:bg-slate-800 border border-stone-400 dark:border-slate-700 rounded overflow-hidden flex-1">
                          <button 
                              onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                              disabled={currentPage === 1}
                              className="h-full px-3 text-stone-500 hover:text-mystic-accent disabled:opacity-30 transition-colors border-r border-stone-400 dark:border-slate-700"
                          >
                              <ChevronLeft size={16} />
                          </button>
                          <div className="flex-1 flex flex-col items-center justify-center">
                              <span className="text-[10px] font-bold text-stone-600 dark:text-slate-400 leading-none">
                                  {currentPage}/{totalPages}
                              </span>
                              <span className="text-[7px] uppercase opacity-50 font-bold">Trang</span>
                          </div>
                          <button 
                              onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                              disabled={currentPage === totalPages}
                              className="h-full px-3 text-stone-500 hover:text-mystic-accent disabled:opacity-30 transition-colors border-l border-stone-400 dark:border-slate-700"
                          >
                              <ChevronRight size={16} />
                          </button>
                      </div>

                      {/* Scroll Controls */}
                      <div className="flex items-center gap-1">
                          <button 
                              onClick={scrollToTop}
                              className="h-10 w-10 flex items-center justify-center rounded bg-stone-200 dark:bg-slate-800 border border-stone-400 dark:border-slate-700 text-stone-500 hover:text-mystic-accent transition-all"
                          >
                              <ChevronsUp size={18} />
                          </button>
                          <button 
                              onClick={scrollToBottom}
                              className="h-10 w-10 flex items-center justify-center rounded bg-stone-200 dark:bg-slate-800 border border-stone-400 dark:border-slate-700 text-stone-500 hover:text-mystic-accent transition-all"
                          >
                              <ChevronsDown size={18} />
                          </button>
                      </div>
                  </div>
              </div>

              {/* Stream Toggle */}
              <button 
                  onClick={toggleStreamResponse}
                  className="w-full p-2 flex justify-between items-center text-left hover:bg-stone-400 dark:hover:bg-slate-700/50 transition-colors bg-stone-200 dark:bg-slate-800/30 rounded border border-stone-400 dark:border-slate-700"
              >
                  <div className="flex items-center gap-2 text-[10px] font-bold text-stone-700 dark:text-slate-300">
                       <Zap size={14} className={settings?.streamResponse ? "text-yellow-500 dark:text-yellow-400" : "text-stone-400 dark:text-slate-500"} />
                       Streaming
                  </div>
                  <div className={settings?.streamResponse ? "text-green-600 dark:text-green-400" : "text-stone-300 dark:text-slate-600"}>
                       {settings?.streamResponse ? <ToggleRight size={20}/> : <ToggleLeft size={20}/>}
                  </div>
              </button>

              <RulesManager rules={dynamicRules} onUpdate={setDynamicRules} />
              
              <button 
                onClick={() => setShowContextModal(true)}
                className="w-full p-2 flex justify-between items-center text-left hover:bg-mystic-accent/10 dark:hover:bg-mystic-accent/5 transition-all bg-stone-200 dark:bg-slate-800/30 rounded border border-stone-400 dark:border-slate-700 group"
              >
                <div className="flex items-center gap-2 text-[10px] font-bold text-mystic-accent uppercase">
                  <Database size={14} className="group-hover:scale-110 transition-transform" />
                  Cửa sổ Ngữ cảnh
                </div>
                <div className="text-[8px] bg-mystic-accent/20 px-1.5 py-0.5 rounded text-mystic-accent font-bold">Config</div>
              </button>

              <TawaPresetManager onConfigChange={setTawaPresetConfig} />
              <AIMonitor />
          </div>
          <div className="p-1 border-t border-stone-400 dark:border-slate-800 bg-stone-200 dark:bg-mystic-900/95 flex flex-row gap-1 mt-auto shrink-0">
              <Button variant="ghost" className="flex-1 text-[12px] h-9 px-1 justify-center border border-stone-400 dark:border-slate-700 hover:bg-stone-400 dark:hover:bg-slate-800 font-mono font-bold leading-[20px]" icon={<Settings size={12}/>} onClick={handleGoToSettings} title="Cài đặt hệ thống">Cài đặt</Button>
              <Button variant="outline" className="flex-1 text-[12px] h-9 px-1 justify-center border border-stone-400 dark:border-slate-700 font-mono font-bold" icon={<Save size={12}/>} onClick={handleManualSave} isLoading={isSaving} disabled={isLoading} title="Lưu thủ công và tải file (.json)">Lưu</Button>
              <Button variant="danger" className="flex-1 text-[12px] h-9 px-1 justify-center border-red-900/30 font-mono font-bold" icon={<LogOut size={12}/>} onClick={handleExit} title="Thoát ra Menu chính">Thoát</Button>
          </div>
      </div>
  );
  };

  // --- RENDER ---
  if (!activeWorld) return null;

  const lastMessage = history[history.length - 1];
  const activeChoices = (lastMessage?.role === 'model' && lastMessage.choices) ? lastMessage.choices : [];

  return (
    <div className="flex h-full w-full bg-stone-300 dark:bg-mystic-900 font-sans overflow-hidden">
        {/* LEFT COLUMN */}
        <div className="flex-1 flex flex-col h-full relative z-10 min-w-0">
            {/* Header ... */}
            <header className="h-14 md:h-16 shrink-0 bg-stone-200 dark:bg-mystic-900 border-b border-stone-400 dark:border-slate-800 flex items-center justify-center relative px-4 z-30 shadow-sm">
                 <button 
                    className="md:hidden absolute left-4 text-stone-500 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white"
                    onClick={() => setShowMobileSidebar(true)}
                 >
                     <Menu size={20} />
                 </button>
                 <div className="flex flex-col items-center">
                     <h1 className="font-bold text-stone-800 dark:text-slate-200 text-xs md:text-sm tracking-wide leading-tight font-mono truncate max-w-[180px] md:max-w-none">
                         {activeWorld.world.worldName}
                     </h1>
                     <div className="mt-0.5 flex items-center gap-2">
                        <span className="text-[9px] md:text-[10px] font-mono font-bold text-mystic-accent bg-mystic-accent/10 px-1.5 md:px-2 py-0.5 rounded-full border border-mystic-accent/20 leading-none">
                            Lượt: {turnCount}
                        </span>
                        <span className="text-[9px] md:text-[10px] font-mono font-bold text-emerald-500 bg-emerald-500/10 px-1.5 md:px-2 py-0.5 rounded-full border border-emerald-500/20 flex items-center gap-1 leading-none">
                            <Clock size={8} className="md:size-2.5" />
                            {formatGameTime(gameTime)}
                        </span>
                     </div>
                 </div>
            </header>

            {/* Chat Area */}
            <div 
                ref={scrollViewportRef}
                onScroll={handleScroll}
                className="flex-1 overflow-y-auto custom-scrollbar p-3 md:p-5 space-y-4 w-full bg-stone-300 dark:bg-mystic-900"
            >
                    {displayedMessages.map((msg, idx) => {
                        const globalIndex = startIndex + idx;
                        const isModel = msg.role === 'model';
                        const swipes = msg.swipes || [msg.text];
                        const swipeIndex = msg.swipeIndex || 0;
                        const displayText = swipes[swipeIndex] || "";
                        
                        // Check if this message is currently being streamed
                        // It is streaming if we are loading AND it is the very last message in the entire history
                        const isStreamingMsg = isLoading && (globalIndex === history.length - 1);

                        return (
                        <motion.div 
                                key={`${currentPage}-${idx}`}
                                id={isModel && msg.turnNumber !== undefined ? `turn-${msg.turnNumber}` : undefined}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className={`flex w-full ${!isModel ? 'justify-end' : 'justify-start'}`}
                            >
                                <div className={`relative rounded-lg p-3 md:p-5 leading-relaxed shadow-md text-base flex flex-col gap-2 ${
                                    !isModel 
                                        ? 'bg-stone-200 dark:bg-slate-800 border border-stone-400 dark:border-slate-700 text-stone-800 dark:text-slate-200 rounded-tr-none max-w-[90%] md:max-w-[85%]' 
                                        : 'bg-transparent text-stone-800 dark:text-slate-300 pl-0 w-full'
                                }`}>
                                    <TawaMessageRenderer 
                                        text={displayText}
                                        onUpdate={(newText) => handleMessageUpdate(globalIndex, newText)}
                                        isStreaming={isStreamingMsg} // Pass prop
                                        settings={settings}
                                        entities={activeWorld.entities}
                                        onEntityClick={(id) => {
                                            const entity = activeWorld.entities.find(e => e.id === id);
                                            if (entity) setSelectedEntity(entity);
                                        }}
                                        turnNumber={isModel ? msg.turnNumber : undefined}
                                        userAction={isModel ? msg.userAction : undefined}
                                        playerName={activeWorld.player.name}
                                        playerAvatar={activeWorld.player.avatar}
                                    />

                                    {/* Swipe Controls for AI Messages */}
                                    {isModel && !isStreamingMsg && (
                                        <div className="flex items-center gap-2 mt-1 select-none w-full border-t border-stone-400 dark:border-slate-800/50 pt-2">
                                            <div className="flex items-center bg-stone-300 dark:bg-slate-800/50 rounded-lg p-0.5 border border-stone-400 dark:border-slate-700/50">
                                                <button 
                                                    onClick={() => handleSwipe(globalIndex, 'prev')}
                                                    disabled={swipeIndex === 0}
                                                    className="p-1 hover:bg-stone-400 dark:hover:bg-slate-700 rounded text-stone-500 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                                                    title="Phiên bản cũ hơn"
                                                >
                                                    <ChevronLeft size={14} />
                                                </button>
                                                <span className="text-[10px] font-mono text-stone-400 dark:text-slate-500 px-2 min-w-[40px] text-center">
                                                    {swipeIndex + 1}/{swipes.length}
                                                </span>
                                                <button 
                                                    onClick={() => handleSwipe(globalIndex, 'next')}
                                                    className="p-1 hover:bg-stone-400 dark:hover:bg-slate-700 rounded text-stone-500 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white disabled:opacity-30 transition-colors"
                                                    title={swipeIndex === swipes.length - 1 ? "Tạo lại (Regenerate)" : "Phiên bản mới hơn"}
                                                >
                                                    {swipeIndex === swipes.length - 1 ? (
                                                        <RefreshCw size={14} className={isLoading ? "animate-spin text-mystic-accent" : ""} />
                                                    ) : (
                                                        <ChevronRight size={14} />
                                                    )}
                                                </button>
                                            </div>
                                            {swipes.length > 1 && (
                                                <span className="text-[10px] text-stone-400 dark:text-slate-600 italic">
                                                    {swipeIndex === swipes.length - 1 ? "Lastest" : "History"}
                                                </span>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </motion.div>
                        );
                    })}
                    {isLoading && !history[history.length - 1]?.text && (
                        /* Only show loader if we are NOT streaming (if streaming, text updates live) */
                        <div className="flex flex-col items-center justify-center p-6 space-y-3 animate-fade-in w-full border-t border-stone-400 dark:border-slate-800/30">
                            <Loader2 className="w-8 h-8 text-mystic-accent animate-spin" />
                            <span className="text-sm font-medium text-stone-500 dark:text-slate-400 animate-pulse">
                                Đang kiến tạo diễn biến...
                            </span>
                        </div>
                    )}
                    <div ref={chatEndRef} />
            </div>

            {/* Input Area ... (Same as before) */}
            <div className="bg-stone-300 dark:bg-mystic-900 border-t border-stone-400 dark:border-slate-800 z-20 shrink-0 flex flex-col shadow-[0_-5px_15px_rgba(0,0,0,0.05)] dark:shadow-[0_-5px_15px_rgba(0,0,0,0.2)]">
                {/* ACTION CHOICES */}
                <AnimatePresence>
                    {activeChoices.length > 0 && !isLoading && !isInputCollapsed && (
                        <motion.div 
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="bg-stone-200/95 dark:bg-slate-900/95 backdrop-blur border-b border-stone-400 dark:border-slate-800 max-h-[35vh] md:max-h-[60vh] overflow-y-auto custom-scrollbar"
                        >
                            <div className="p-2 flex flex-col gap-2">
                                <div className="flex justify-between items-center px-1 mb-1">
                                    <span className="text-[10px] font-black uppercase tracking-widest text-stone-500 dark:text-slate-500">Lựa chọn hành động</span>
                                    <button 
                                        onClick={() => {
                                            const lastModelIdx = [...history].reverse().findIndex(m => m.role === 'model');
                                            if (lastModelIdx !== -1) {
                                                const actualIdx = history.length - 1 - lastModelIdx;
                                                // Trigger the "Edit Raw" of the last model message
                                                // This is a bit hacky but it's the most direct way to edit choices
                                                // since they are derived from the raw text.
                                                // We can scroll to the last message and let the user click Edit Raw.
                                                const lastMsgEl = document.getElementById(`turn-${history[actualIdx].turnNumber}`);
                                                if (lastMsgEl) {
                                                    lastMsgEl.scrollIntoView({ behavior: 'smooth' });
                                                    // We can't easily trigger the state inside the child component from here
                                                    // but we can inform the user.
                                                }
                                            }
                                        }}
                                        className="text-[10px] font-bold text-mystic-accent hover:underline flex items-center gap-1"
                                    >
                                        <Edit2 size={10} /> Chỉnh sửa
                                    </button>
                                </div>
                                {activeChoices.map((choice, idx) => {
                                    // Parse choice format: "[time] action" or just "action"
                                    // More flexible regex to catch [time] even if not at start
                                    if (typeof choice !== 'string') return null;
                                    const timeMatch = choice.match(/\[(\d+)\s*(m|phút|min)?\]/i);
                                    const displayTime = timeMatch ? timeMatch[1] : null;
                                    // Remove the [time] part from the display action
                                    const displayAction = choice.replace(/\[\d+\s*(m|phút|min)?\]/i, '').trim();

                                    return (
                                        <div key={idx} className="flex gap-2 w-full items-stretch">
                                            <motion.button
                                                whileHover={{ scale: 1.002, backgroundColor: "rgba(56, 189, 248, 0.05)" }}
                                                whileTap={{ scale: 0.998 }}
                                                onClick={() => handleSend(displayAction)}
                                                className="flex-1 text-left text-[11px] md:text-sm p-3 rounded-lg bg-stone-300 dark:bg-slate-800 border border-stone-400 dark:border-slate-700 hover:border-mystic-accent/50 text-stone-700 dark:text-slate-300 hover:text-stone-900 dark:hover:text-white transition-all shadow-sm group h-auto whitespace-normal flex flex-row items-start justify-between gap-2"
                                            >
                                                <span className="flex-1 break-words font-medium leading-relaxed">{displayAction}</span>
                                                {displayTime && (
                                                    <span className="text-[9px] font-mono font-bold text-mystic-accent bg-mystic-accent/10 px-1.5 py-0.5 rounded border border-mystic-accent/20 whitespace-nowrap shrink-0 mt-0.5">
                                                        {displayTime}m
                                                    </span>
                                                )}
                                            </motion.button>
                                            <button 
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    navigator.clipboard.writeText(displayAction);
                                                }}
                                                className="px-3 rounded-lg bg-stone-300 dark:bg-slate-800 border border-stone-400 dark:border-slate-700 hover:border-mystic-accent/50 text-stone-400 dark:text-slate-500 hover:text-mystic-accent transition-all shadow-sm flex items-center justify-center shrink-0 active:scale-95"
                                                title="Sao chép nhanh"
                                            >
                                                <Copy size={14} />
                                            </button>
                                        </div>
                                    );
                                })}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Input Bar */}
                <div className="p-1.5 md:p-2 flex flex-col gap-2 border-t border-stone-400 dark:border-slate-800 bg-stone-300/50 dark:bg-slate-900/50 backdrop-blur-sm">
                    {/* Top Row: Input Area */}
                    <AnimatePresence initial={false}>
                        {!isInputCollapsed && (
                            <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                className="w-full overflow-hidden"
                            >
                                <div className="relative w-full group">
                                    <textarea 
                                        value={inputValue}
                                        onChange={(e) => setInputValue(e.target.value)}
                                        onKeyDown={handleKeyDown}
                                        placeholder="Nhập hành động..."
                                        className="w-full bg-stone-200 dark:bg-slate-800/80 border border-stone-400 dark:border-slate-700 rounded-lg p-2.5 pb-12 text-stone-800 dark:text-slate-200 outline-none focus:border-mystic-accent focus:bg-stone-100 dark:focus:bg-slate-800 transition-all resize-none h-20 md:h-28 custom-scrollbar font-sans text-sm md:text-base shadow-inner"
                                    />
                                    <div className="absolute bottom-2 right-2 flex items-center gap-1.5">
                                        {/* Phục hồi Button (Inside) */}
                                        <button 
                                            onClick={() => {
                                                if (lastAction) {
                                                    setInputValue(lastAction);
                                                }
                                            }} 
                                            disabled={isLoading || !lastAction} 
                                            className="h-8 px-3 text-[10px] font-bold uppercase tracking-tighter bg-stone-300/80 dark:bg-slate-700/80 hover:bg-stone-400 dark:hover:bg-slate-600 text-stone-600 dark:text-slate-300 rounded border border-stone-400 dark:border-slate-600 disabled:opacity-30 transition-all"
                                            title="Phục hồi hành động gần nhất"
                                        >
                                            Phục hồi
                                        </button>
                                        {/* Send Button (Inside) */}
                                        <button 
                                            onClick={() => handleSend()} 
                                            disabled={isLoading || !inputValue.trim()} 
                                            className="h-8 w-12 flex items-center justify-center bg-mystic-accent hover:bg-mystic-accent-light text-white rounded shadow-sm disabled:opacity-30 transition-all"
                                            title="Gửi hành động"
                                        >
                                            {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                                        </button>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Bottom Row: Controls */}
                    <div className="flex gap-1 items-center w-full overflow-x-auto no-scrollbar pb-0.5">
                        <div className="flex items-center gap-1 flex-1 justify-center">
                            {/* Send Button (Only when collapsed) */}
                            {isInputCollapsed && (
                                <Button 
                                    variant="primary" 
                                    onClick={() => handleSend()} 
                                    disabled={isLoading} 
                                    className="h-9 md:h-10 px-4 flex items-center justify-center shrink-0"
                                    title="Gửi hành động (Continue)"
                                >
                                    {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
                                </Button>
                            )}

                            {/* Thử Lại Button */}
                            <Button 
                                variant="ghost" 
                                onClick={() => {
                                    const lastModelIdx = [...history].reverse().findIndex(m => m.role === 'model');
                                    if (lastModelIdx !== -1) {
                                        const actualIdx = history.length - 1 - lastModelIdx;
                                        handleRegenerate(actualIdx);
                                    }
                                }} 
                                disabled={isLoading || !history.some(m => m.role === 'model')} 
                                className="h-9 md:h-10 px-3 text-[10px] font-bold uppercase tracking-tighter border border-stone-400 dark:border-slate-700 hover:border-mystic-accent/50 whitespace-nowrap flex items-center justify-center gap-1 shrink-0"
                                title="Tạo lại phản hồi cuối cùng của AI"
                            >
                                <RefreshCw size={14} className={isLoading ? "animate-spin" : ""} />
                                <span className="hidden sm:inline">Thử Lại</span>
                            </Button>

                            {/* Pagination Group */}
                            <div className="flex items-center h-9 md:h-10 bg-stone-200 dark:bg-slate-800 border border-stone-400 dark:border-slate-700 rounded overflow-hidden shrink-0">
                                <button 
                                    onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                                    disabled={currentPage === 1}
                                    className="h-full px-2 text-stone-500 hover:text-mystic-accent disabled:opacity-30 transition-colors border-r border-stone-400 dark:border-slate-700"
                                    title="Trang trước"
                                >
                                    <ChevronLeft size={14} />
                                </button>
                                <div className="px-2 flex flex-col items-center justify-center min-w-[40px]">
                                    <span className="text-[10px] font-bold text-stone-600 dark:text-slate-400 leading-none">
                                        {currentPage}/{totalPages}
                                    </span>
                                    <span className="text-[7px] uppercase opacity-50 font-bold">Trang</span>
                                </div>
                                <button 
                                    onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                                    disabled={currentPage === totalPages}
                                    className="h-full px-2 text-stone-500 hover:text-mystic-accent disabled:opacity-30 transition-colors border-l border-stone-400 dark:border-slate-700"
                                    title="Trang sau"
                                >
                                    <ChevronRight size={14} />
                                </button>
                            </div>

                            {/* Scroll Controls */}
                            <div className="flex items-center gap-1 shrink-0">
                                <button 
                                    onClick={scrollToTop}
                                    className="h-9 w-9 md:h-10 md:w-10 flex items-center justify-center rounded bg-stone-200 dark:bg-slate-800 border border-stone-400 dark:border-slate-700 text-stone-500 hover:text-mystic-accent hover:border-mystic-accent transition-all shadow-sm"
                                    title="Lên đầu lượt"
                                >
                                    <ChevronsUp size={16} />
                                </button>
                                <button 
                                    onClick={scrollToBottom}
                                    className="h-9 w-9 md:h-10 md:w-10 flex items-center justify-center rounded bg-stone-200 dark:bg-slate-800 border border-stone-400 dark:border-slate-700 text-stone-500 hover:text-mystic-accent hover:border-mystic-accent transition-all shadow-sm"
                                    title="Về cuối lượt"
                                >
                                    <ChevronsDown size={16} />
                                </button>
                            </div>
                        </div>

                        {/* Toggle Input Button - Fills remaining space */}
                        <button 
                            onClick={() => setIsInputCollapsed(!isInputCollapsed)}
                            className={`h-9 md:h-10 px-3 rounded border transition-all flex items-center justify-center gap-2 flex-1 shadow-sm min-w-[40px] ${
                                isInputCollapsed 
                                ? 'bg-mystic-accent/10 border-mystic-accent/30 text-mystic-accent' 
                                : 'bg-stone-200 dark:bg-slate-800 border-stone-400 dark:border-slate-700 text-stone-500'
                            }`}
                            title={isInputCollapsed ? "Mở rộng ô nhập" : "Thu gọn ô nhập"}
                        >
                            {isInputCollapsed ? <Maximize2 size={16} /> : <Minimize2 size={16} />}
                            <span className="text-[10px] font-bold uppercase whitespace-nowrap">
                                {isInputCollapsed ? 'Mở Rộng' : 'Thu Gọn'}
                            </span>
                        </button>
                    </div>
                </div>
            </div>
        </div>

        {/* SIDEBAR - DESKTOP */}
        <div className="hidden md:block w-80 shrink-0 h-full relative z-20 border-l border-stone-400 dark:border-slate-800">
            {renderSidebarContent()}
        </div>

        {/* SIDEBAR - MOBILE OVERLAY */}
        <AnimatePresence>
            {showMobileSidebar && (
                <>
                    <motion.div 
                        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/80 z-40 md:hidden backdrop-blur-sm"
                        onClick={() => setShowMobileSidebar(false)}
                    />
                    <motion.div
                        initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }}
                        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                        className="fixed top-0 right-0 h-full w-4/5 max-w-sm bg-stone-300 dark:bg-mystic-900 z-50 border-l border-stone-400 dark:border-slate-700 md:hidden"
                    >
                        {renderSidebarContent()}
                    </motion.div>
                </>
            )}
        </AnimatePresence>

        {/* MODALS */}
        {/* HISTORY & LOAD SAVE MODAL */}
        <AnimatePresence>
            {showHistoryModal && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-2">
                    <motion.div 
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="bg-stone-200 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 w-full max-w-4xl h-[90vh] rounded-xl shadow-2xl flex flex-col overflow-hidden"
                    >
                        <div className="p-4 border-b border-stone-400 dark:border-slate-800 flex justify-between items-center bg-stone-300 dark:bg-slate-900/50">
                            <h2 className="text-lg font-bold text-stone-800 dark:text-slate-200 flex items-center gap-2 uppercase tracking-widest">
                                <Database size={20} className="text-mystic-accent"/> Dữ Liệu & Lịch Sử
                            </h2>
                            <button onClick={() => setShowHistoryModal(false)} className="text-stone-500 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white p-1 rounded hover:bg-stone-400 dark:hover:bg-slate-800 transition-colors">
                                <X size={24} />
                            </button>
                        </div>
                        
                        {/* Tab Navigation */}
                        <div className="flex bg-stone-300 dark:bg-slate-900/80 p-1 gap-1 border-b border-stone-400 dark:border-slate-800">
                            {[
                                { id: 'history', label: 'Cốt truyện', icon: <History size={14} /> },
                                { id: 'manual', label: 'Lưu Thủ Công', icon: <Save size={14} /> },
                                { id: 'autosave', label: 'Lưu Tự Động', icon: <Clock size={14} /> },
                                { id: 'initial', label: 'Bản lưu lượt 0', icon: <Shield size={14} /> }
                            ].map(tab => (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveSaveTab(tab.id as 'manual' | 'autosave' | 'history' | 'initial')}
                                    className={`flex-1 py-2.5 flex items-center justify-center gap-2 text-[10px] font-black uppercase tracking-widest transition-all rounded-lg ${
                                        activeSaveTab === tab.id 
                                        ? 'bg-mystic-accent text-mystic-900 shadow-lg' 
                                        : 'text-stone-500 hover:bg-stone-400/20 dark:hover:bg-slate-800'
                                    }`}
                                >
                                    {tab.icon}
                                    <span className={isMobile ? 'hidden' : 'inline'}>{tab.label}</span>
                                    {isMobile && <span>{tab.label.split(' ')[0]}</span>}
                                </button>
                            ))}
                        </div>

                        <div className="flex-1 overflow-hidden bg-stone-200 dark:bg-mystic-950">
                            {activeSaveTab === 'history' ? (
                                <div className="h-full flex flex-col overflow-hidden">
                                    <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
                                        {history.filter(m => m.role === 'model').length === 0 ? (
                                            <div className="text-center text-stone-400 dark:text-slate-500 py-20">
                                                Chưa có tóm tắt cốt truyện.
                                            </div>
                                        ) : (
                                            [...history].filter(m => m.role === 'model').reverse().map((msg) => (
                                                <div key={msg.timestamp} className="bg-stone-300 dark:bg-slate-800/50 p-4 rounded-lg border border-stone-400 dark:border-slate-700 shadow-sm hover:border-blue-500/50 transition-all group">
                                                    <div className="flex justify-between items-start mb-2">
                                                        <div className="flex items-center gap-2">
                                                            <span className="text-[10px] font-bold bg-blue-600/20 text-blue-600 dark:text-blue-400 px-2 py-0.5 rounded uppercase tracking-wider">
                                                                Lượt {msg.turnNumber}
                                                            </span>
                                                            <span className="text-[10px] text-stone-500 dark:text-slate-500 font-mono">
                                                                {new Date(msg.timestamp).toLocaleTimeString()}
                                                            </span>
                                                        </div>
                                                    </div>
                                                    
                                                    {msg.userAction && (
                                                        <div className="mb-3 p-2 bg-stone-400/10 dark:bg-slate-900/30 rounded border-l-2 border-stone-400 dark:border-slate-700 italic text-xs text-stone-600 dark:text-slate-400">
                                                            "{msg.userAction}"
                                                        </div>
                                                    )}

                                                    <div className="text-sm text-stone-800 dark:text-slate-200 leading-relaxed whitespace-pre-line">
                                                        {msg.incrementalSummary || <span className="text-stone-400 italic">Không có tóm tắt.</span>}
                                                    </div>
                                                </div>
                                            ))
                                        )}
                                    </div>
                                </div>
                            ) : (
                                <div className="h-full flex flex-col overflow-hidden">
                                    <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
                                        {(activeSaveTab === 'manual' ? manualSaveList : activeSaveTab === 'autosave' ? autosaveList : initialSaveList).length === 0 ? (
                                            <div className="text-center text-stone-400 dark:text-slate-500 py-20">
                                                Chưa có tệp lưu {activeSaveTab === 'manual' ? 'thủ công' : activeSaveTab === 'autosave' ? 'tự động' : 'lượt 0'}.
                                            </div>
                                        ) : (
                                            (activeSaveTab === 'manual' ? manualSaveList : activeSaveTab === 'autosave' ? autosaveList : initialSaveList).map((save) => (
                                                <div 
                                                    key={save.id} 
                                                    className="bg-stone-300 dark:bg-slate-800/50 p-4 rounded-xl border border-stone-400 dark:border-slate-700 shadow-sm hover:border-mystic-accent transition-all group flex flex-col gap-4"
                                                >
                                                    <div className="flex-1">
                                                        <h4 className="text-sm font-bold text-stone-800 dark:text-slate-200 mb-1 group-hover:text-mystic-accent transition-colors">
                                                            {save.name}
                                                        </h4>
                                                        <div className="flex items-center gap-3 text-[10px] text-stone-500 dark:text-slate-500 font-medium">
                                                            <span className="flex items-center gap-1"><Clock size={10}/> {new Date(save.updatedAt).toLocaleString()}</span>
                                                            <span className="opacity-30">|</span>
                                                            <span className="flex items-center gap-1"><Zap size={10}/> Lượt: {save.data?.savedState?.turnCount || 0}</span>
                                                        </div>
                                                    </div>
                                                    
                                                    <div className="flex gap-2 pt-2 border-t border-stone-400/30 dark:border-slate-800/50">
                                                        <Button 
                                                            variant="primary" 
                                                            size="sm" 
                                                            onClick={() => handleLoadSave(save)}
                                                            className="flex-1 h-9 text-[11px] font-black uppercase tracking-widest shadow-lg shadow-mystic-accent/10"
                                                            icon={<BookOpen size={14} />}
                                                        >
                                                            Tải Dữ Liệu
                                                        </Button>
                                                        <Button 
                                                            variant="ghost" 
                                                            size="sm" 
                                                            onClick={() => handleDeleteSave(save.id)}
                                                            className="h-9 px-3 text-red-500 hover:bg-red-500/10 border border-stone-400/50 dark:border-slate-700 hover:border-red-500/50 transition-all"
                                                            title="Xóa tệp lưu"
                                                        >
                                                            <Trash2 size={16} />
                                                        </Button>
                                                    </div>
                                                </div>
                                            ))
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>

        {/* Character Modal (UPDATED WITH FULL INFO) */}
        <AnimatePresence>
            {showCharModal && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm">
                    <motion.div 
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="bg-stone-200 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 w-full h-full rounded-none shadow-2xl flex flex-col overflow-hidden"
                    >
                        <div className="p-2 border-b border-stone-400 dark:border-slate-800 flex justify-between items-center bg-stone-300 dark:bg-slate-900/50">
                            <h2 className="text-lg font-bold text-stone-800 dark:text-slate-200 flex items-center gap-2">
                                <User size={20} className="text-mystic-accent"/> Hồ Sơ Nhân Vật
                            </h2>
                            <button onClick={() => setShowCharModal(false)} className="text-stone-500 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white p-1 rounded hover:bg-stone-400 dark:hover:bg-slate-800 transition-colors">
                                <X size={24} />
                            </button>
                        </div>

                        <div className="p-2 overflow-y-auto custom-scrollbar space-y-2 bg-stone-200 dark:bg-mystic-900">
                            <div className="flex items-start gap-2 mb-2">
                                <button 
                                    onClick={() => {
                                        setSelectingAvatarFor({ type: 'player' });
                                        setShowImageLibrary(true);
                                    }}
                                    className="w-20 h-20 rounded-full bg-stone-300 dark:bg-slate-800 border-2 border-mystic-accent flex items-center justify-center shrink-0 shadow-lg overflow-hidden group relative"
                                >
                                    {activeWorld.player.avatar ? (
                                        <img src={activeWorld.player.avatar} alt={activeWorld.player.name} className="w-full h-full object-cover" referrerPolicy="no-referrer" />
                                    ) : (
                                        <User size={40} className="text-mystic-accent" />
                                    )}
                                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                        <Edit2 size={20} className="text-white" />
                                    </div>
                                </button>
                                <div>
                                    <h3 className="text-2xl font-bold text-stone-900 dark:text-white mb-1 font-mono text-[12px]">{activeWorld.player.name}</h3>
                                    <div className="flex gap-2 text-sm text-stone-500 dark:text-slate-400">
                                        <span className="bg-stone-300 dark:bg-slate-800 px-2 py-0.5 rounded border border-stone-400 dark:border-slate-700">{activeWorld.player.gender}</span>
                                        <span className="bg-stone-300 dark:bg-slate-800 px-2 py-0.5 rounded border border-stone-400 dark:border-slate-700">{activeWorld.player.age} tuổi</span>
                                    </div>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 gap-1">
                                <div className="space-y-1">
                                    <h4 className="text-xs font-bold text-mystic-accent uppercase tracking-wider">Ngoại hình</h4>
                                    <MarkdownRenderer 
                                        className="text-sm text-stone-700 dark:text-slate-300 bg-stone-300 dark:bg-slate-800/50 p-1 rounded border border-stone-400 dark:border-slate-700/50 leading-relaxed"
                                        content={activeWorld.player.appearance || "Chưa có mô tả ngoại hình."}
                                    />
                                </div>

                                <div className="space-y-1">
                                    <h4 className="text-xs font-bold text-mystic-accent uppercase tracking-wider">Tính cách</h4>
                                    <MarkdownRenderer 
                                        className="text-sm text-stone-700 dark:text-slate-300 bg-stone-300 dark:bg-slate-800/50 p-1 rounded border border-stone-400 dark:border-slate-700/50 leading-relaxed"
                                        content={activeWorld.player.personality || "Chưa có mô tả tính cách."}
                                    />
                                </div>

                                <div className="space-y-1">
                                    <h4 className="text-xs font-bold text-mystic-accent uppercase tracking-wider">Tiểu sử & Xuất thân</h4>
                                    <MarkdownRenderer 
                                        className="text-sm text-stone-700 dark:text-slate-300 bg-stone-300 dark:bg-slate-800/50 p-1 rounded border border-stone-400 dark:border-slate-700/50 leading-relaxed"
                                        content={activeWorld.player.background || "Chưa có tiểu sử."}
                                    />
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-1">
                                    <div className="space-y-1">
                                        <h4 className="text-xs font-bold text-mystic-accent uppercase tracking-wider">Kỹ năng</h4>
                                        <MarkdownRenderer 
                                            className="text-sm text-stone-700 dark:text-slate-300 bg-stone-300 dark:bg-slate-800/50 p-1 rounded border border-stone-400 dark:border-slate-700/50 leading-relaxed h-full"
                                            content={activeWorld.player.skills || "Không có kỹ năng đặc biệt."}
                                        />
                                    </div>
                                    <div className="space-y-1">
                                        <h4 className="text-xs font-bold text-mystic-accent uppercase tracking-wider">Mục tiêu</h4>
                                        <MarkdownRenderer 
                                            className="text-sm text-stone-700 dark:text-slate-300 bg-stone-300 dark:bg-slate-800/50 p-1 rounded border border-stone-400 dark:border-slate-700/50 leading-relaxed h-full"
                                            content={activeWorld.player.goal || "Chưa xác định mục tiêu."}
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>
        
        {/* GLOBAL INFO (LSR) MODAL (UPDATED) */}
        <AnimatePresence>
            {showGlobalModal && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm">
                    <motion.div 
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="bg-stone-200 dark:bg-mystic-900 border border-stone-400 dark:border-slate-700 w-full h-full rounded-none shadow-2xl flex flex-col overflow-hidden"
                    >
                        <div className="p-2 border-b border-stone-400 dark:border-slate-800 flex justify-between items-center bg-stone-300 dark:bg-slate-900/50">
                            <h2 className="text-[12px] leading-[22px] font-bold text-green-600 dark:text-green-400 flex items-center gap-2">
                                <Database size={20}/> LSR Database (Trạng thái thế giới)
                            </h2>
                            <button onClick={() => setShowGlobalModal(false)} className="text-stone-500 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white p-1 rounded hover:bg-stone-400 dark:hover:bg-slate-800 transition-colors">
                                <X size={24} />
                            </button>
                        </div>
                        <div className="p-2 overflow-y-auto custom-scrollbar space-y-2 bg-stone-200 dark:bg-mystic-900">
                            {lsrTables.length === 0 ? (
                                <div className="text-center text-stone-400 dark:text-slate-500 py-10">
                                    Không tìm thấy dữ liệu cấu trúc bảng LSR.
                                </div>
                            ) : (
                                <>
                                    {/* Tab Navigation / Dropdown */}
                                    <div className="mb-4">
                                        {isMobile ? (
                                            <div className="relative">
                                                <select 
                                                    value={activeLsrTableId || ''} 
                                                    onChange={(e) => setActiveLsrTableId(e.target.value)}
                                                    className="w-full p-2 bg-stone-300 dark:bg-slate-800 border border-stone-400 dark:border-slate-700 rounded-lg text-sm text-stone-800 dark:text-slate-200 outline-none appearance-none"
                                                >
                                                    {lsrTables.map(table => (
                                                        <option key={table.id} value={table.id}>{table.name}</option>
                                                    ))}
                                                </select>
                                                <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-stone-500">
                                                    <ChevronDown size={16} />
                                                </div>
                                            </div>
                                        ) : (
                                            <div className="flex flex-wrap gap-2 border-b border-stone-400 dark:border-slate-800 pb-2">
                                                {lsrTables.map(table => (
                                                    <button
                                                        key={table.id}
                                                        onClick={() => setActiveLsrTableId(table.id)}
                                                        className={`px-3 py-1.5 text-xs font-bold rounded-t-lg transition-all ${
                                                            activeLsrTableId === table.id
                                                            ? 'bg-mystic-accent text-mystic-950 shadow-[0_-2px_10px_rgba(56,189,248,0.3)]'
                                                            : 'bg-stone-300 dark:bg-slate-800 text-stone-500 dark:text-slate-400 hover:bg-stone-400 dark:hover:bg-slate-700'
                                                        }`}
                                                    >
                                                        {table.name}
                                                    </button>
                                                ))}
                                            </div>
                                        )}
                                    </div>

                                    {/* Active Table Content */}
                                    {lsrTables.filter(t => t.id === activeLsrTableId).map((table) => {
                                        const currentRows = lsrRuntimeData[table.id] || [];
                                        return (
                                            <div key={table.id} className="space-y-1 animate-in fade-in duration-300">
                                                <div className="flex items-center gap-1">
                                                    <span className="bg-stone-300 dark:bg-slate-800 text-stone-500 dark:text-slate-400 text-xs font-bold px-1 py-0.5 rounded">#{table.id}</span>
                                                    <h3 className="text-[12px] leading-[18px] font-bold text-stone-800 dark:text-slate-200">{table.name}</h3>
                                                </div>
                                                <div className="overflow-x-auto rounded-lg border border-stone-400 dark:border-slate-700 shadow-sm">
                                                    <table className="w-full text-sm text-left text-stone-500 dark:text-slate-400">
                                                        <thead className="text-xs text-stone-700 dark:text-slate-300 uppercase bg-stone-300 dark:bg-slate-800">
                                                            <tr>
                                                                {table.columns.map((col, idx) => (
                                                                    <th key={idx} scope="col" className="px-1 py-1 whitespace-nowrap border-r border-stone-400 dark:border-slate-700 last:border-r-0">
                                                                        {col}
                                                                    </th>
                                                                ))}
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            {currentRows.length === 0 ? (
                                                                <tr className="bg-stone-200 dark:bg-slate-900/50 border-b border-stone-400 dark:border-slate-800 hover:bg-stone-300 dark:hover:bg-slate-800/30 transition-colors">
                                                                    <td colSpan={table.columns.length} className="px-1 py-1 border-r border-stone-400 dark:border-slate-800 last:border-r-0 italic text-stone-400 dark:text-slate-600 text-center">
                                                                        (Chưa có dữ liệu)
                                                                    </td>
                                                                </tr>
                                                            ) : (
                                                                currentRows.map((row, rIdx) => (
                                                                    <tr key={rIdx} className="bg-stone-200 dark:bg-slate-900/50 border-b border-stone-400 dark:border-slate-800 hover:bg-stone-300 dark:hover:bg-slate-800/30 transition-colors">
                                                                        {table.columns.map((_, cIdx) => (
                                                                            <td key={cIdx} className="px-1 py-1 border-r border-stone-400 dark:border-slate-800 last:border-r-0 text-stone-700 dark:text-slate-300">
                                                                                {row[cIdx.toString()] || "-"}
                                                                            </td>
                                                                        ))}
                                                                    </tr>
                                                                ))
                                                            )}
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </>
                            )}
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>

        <ContextWindowModal />

        <AnimatePresence>
            {selectedEntity && (
                <EntityDetailModal 
                    entity={selectedEntity} 
                    onClose={() => setSelectedEntity(null)} 
                    onUpdateAvatar={(id) => {
                        setSelectingAvatarFor({ type: 'entity', id });
                        setShowImageLibrary(true);
                    }}
                />
            )}
        </AnimatePresence>

        {/* Image Library Modal */}
        <ImageLibraryModal 
            isOpen={showImageLibrary}
            onClose={() => {
                setShowImageLibrary(false);
                setSelectingAvatarFor(null);
            }}
            onSelect={handleAvatarSelect}
            selectedId={selectingAvatarFor?.type === 'player' ? activeWorld.player.avatar : activeWorld.entities.find(e => e.id === selectingAvatarFor?.id)?.avatar}
        />
    </div>
  );
};

export default GameplayScreen;
