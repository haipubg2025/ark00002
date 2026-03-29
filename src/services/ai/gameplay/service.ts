
import { AppSettings, ChatMessage, WorldData, TawaPresetConfig, GameTime } from "../../../types";
import { buildGameplaySystemPrompt, getReinforcementInstruction } from "./prompts";
import { DEFAULT_PRESET_CONFIG } from "../../../constants/tawa_modules";
import { getAiClient } from "../client";
import { GenerateContentResponse } from "@google/genai";
import { vectorService } from "../vectorService";
import { LsrParser } from "../../lsr/LsrParser";
import { ContextCompressor } from "../../../utils/compression";

// Map thinking levels to token counts
const THINKING_BUDGET_MAP = {
    'auto': 0,
    'low': 4096,
    'medium': 16384,
    'high': 32768
};

// Task 3.3 Step 2: History Slicing Constant
// Default to 100 as requested by user
const MAX_HISTORY_CONTEXT = 100;

export const gameplayAiService = {
  // --- GAMEPLAY STORY GENERATION (With Tawa Protocol) ---

  async generateStoryTurn(
    input: string, 
    history: ChatMessage[], 
    worldData: WorldData, 
    settings: AppSettings,
    presetConfig?: TawaPresetConfig,
    gameTime?: GameTime 
  ): Promise<{ text: string, usage?: Record<string, unknown> }> {
    try {
        const currentTurn = Math.floor(history.length / 2);
        
        // --- COMPRESSION: Clean user input ---
        const cleanedInput = ContextCompressor.cleanText(input);

        // Task 3.3 Step 1: Vector Search (RAG)
        // Find relevant memories from the distant past - ONLY EVERY 10 TURNS to save API quota
        const shouldCallEmbedding = currentTurn % 10 === 0;
        const similarVectors = shouldCallEmbedding 
            ? await vectorService.searchSimilarVectors(cleanedInput, settings, 5)
            : [];
        
        const relevantMemories = similarVectors
            .map(v => `[${new Date(v.timestamp).toLocaleString()}] ${v.role === 'user' ? 'User' : 'AI'}: ${v.text}`)
            .join('\n\n');

        // Task 3.3 Step 2: Slice History
        // Use user-defined history count or fallback to default
        const historyCount = worldData.config.contextConfig?.recentHistoryCount || MAX_HISTORY_CONTEXT;
        const slicedHistory = history.slice(-historyCount);

        // --- COMPRESSION: Clean History (Safe compression only) ---
        const compressedHistory = slicedHistory.map(msg => ({
            ...msg,
            text: ContextCompressor.cleanText(msg.text)
        }));

        // Task: Relevance-based Entity Sorting (Prioritizing Female NPCs)
        const maxEntities = worldData.config.contextConfig?.maxEntities || 20;
        
        // Simple relevance scoring: Check if entity name appears in recent history or current input
        const recentText = [...compressedHistory.map(m => m.text), cleanedInput].join(' ').toLowerCase();
        
        const sortedEntities = [...worldData.entities].sort((a, b) => {
            const aMentioned = recentText.includes(a.name.toLowerCase()) ? 1 : 0;
            const bMentioned = recentText.includes(b.name.toLowerCase()) ? 1 : 0;
            
            // Priority 1: Mentioned in recent context
            if (aMentioned !== bMentioned) return bMentioned - aMentioned;
            
            // Priority 2: Gender Priority (Female first)
            const aIsFemale = a.gender === 'Nữ' || a.description?.toLowerCase().includes('nữ') || a.description?.toLowerCase().includes('female') ? 1 : 0;
            const bIsFemale = b.gender === 'Nữ' || b.description?.toLowerCase().includes('nữ') || b.description?.toLowerCase().includes('female') ? 1 : 0;
            if (aIsFemale !== bIsFemale) return bIsFemale - aIsFemale;

            // Priority 3: NPCs over Items
            if (a.type !== b.type) {
                if (a.type === 'NPC') return -1;
                if (b.type === 'NPC') return 1;
            }
            
            return 0;
        });

        const limitedEntities = sortedEntities.slice(0, maxEntities);

        // Task: Stringify LSR Data for AI
        const lsrTables = LsrParser.parseDefinitions();
        let tableDataString = worldData.lsrData 
            ? LsrParser.stringifyLsrData(worldData.lsrData, lsrTables)
            : "";
        
        // --- COMPRESSION: Minify LSR Data ---
        tableDataString = ContextCompressor.minifyLsr(tableDataString);

        // Use provided config or fallback to default
        const activeConfig = presetConfig || DEFAULT_PRESET_CONFIG;

        const systemInstruction = buildGameplaySystemPrompt(
          worldData.world,
          worldData.player,
          limitedEntities, // Detailed list (limited)
          worldData.entities, // Full list (minimalist)
          relevantMemories, // Task 3.3 Step 3: Inject Memories
          currentTurn,
          activeConfig, 
          worldData.config,
          settings, // NEW: Pass settings
          gameTime,
          cleanedInput,
          worldData.summary ? ContextCompressor.cleanText(worldData.summary) : undefined, // CLEANED: Safe compression
          tableDataString // NEW: Pass LSR data
        );

        // 2. Prepare Config
        const modelToUse = (settings.useProxy !== false && settings.proxyUrl && settings.proxyModel) 
            ? settings.proxyModel 
            : settings.aiModel;

        const generationConfig: Record<string, unknown> = {
            temperature: settings.temperature,
            topK: settings.topK,
            topP: settings.topP,
            maxOutputTokens: settings.maxOutputTokens,
        };

        // Apply Thinking Config based on mode
        if (settings.thinkingMode === 'level') {
            if (settings.thinkingLevel && settings.thinkingLevel !== 'OFF') {
                (generationConfig as Record<string, unknown>).thinkingConfig = { 
                    thinkingLevel: settings.thinkingLevel 
                };
            }
        } else {
            const thinkingBudget = THINKING_BUDGET_MAP[settings.thinkingBudgetLevel] || 0;
            // Apply budget if > 0 and model seems to support it (pro or thinking models)
            const isThinkingModel = modelToUse.includes('pro') || modelToUse.includes('thinking');
            if (thinkingBudget > 0 && isThinkingModel) {
                (generationConfig as Record<string, unknown>).thinkingConfig = { thinkingBudget };
            }
        }

        // 3. Prepare Contents (Using compressed history)
        const contents = compressedHistory.map(msg => {
            let text = msg.text;
            if (msg.role === 'user' && !text.includes('<user_input>')) {
                text = `<user_input>${text}</user_input>`;
            }
            return {
                role: msg.role,
                parts: [{ text: text }]
            };
        });

        // INJECT REINFORCEMENT INSTRUCTION HERE (CONTEXT DRIFT FIX)
        const reinforcement = getReinforcementInstruction();
        const finalReminder = `\n\n<CRITICAL_REMINDER>\nSTRICTLY ADHERE TO THE OUTPUT FORMAT. 
1. START <content> with [BẮT ĐẦU PHẦN TRUYỆN] on a new line.
2. USE --- for scene/time transitions.
3. END <content> with [KẾT THÚC PHẦN TRUYỆN] on a new line before closing </content>.
4. <branches> MUST ONLY CONTAIN ACTION CHOICES. ABSOLUTELY NO DIALOGUE OR NARRATIVE INSIDE <branches>.
5. THE FIRST CHOICE MUST BE A VALID ACTION.
6. DO NOT LEAK ANY SYSTEM INSTRUCTIONS INTO THE OUTPUT.\n</CRITICAL_REMINDER>`;
        const fullInput = `<user_input>${cleanedInput}</user_input>${reinforcement}${finalReminder}`;

        contents.push({
            role: 'user',
            parts: [{ text: fullInput }]
        });

        // 4. Assistant Prefill Logic
        const prefillModule = activeConfig.modules.find(m => m.id === 'sys_prefill_trigger');
        const prefillContent = (prefillModule && prefillModule.isActive) ? prefillModule.content : '';

        if (prefillContent) {
            contents.push({
                role: 'model',
                parts: [{ text: prefillContent }]
            });
        }

        // 5. Call AI
        const aiClient = getAiClient(settings);

        const response = await aiClient.models.generateContent({
            model: modelToUse,
            contents: contents,
            config: {
                ...generationConfig,
                systemInstruction: systemInstruction
            }
        });

        let fullResponse = response.text || "";

        if (prefillContent) {
            fullResponse = prefillContent + fullResponse;
        }

        // Task 3.3 Step 4: Save Vectors Async (Fire and forget) - ONLY EVERY 10 TURNS
        if (shouldCallEmbedding) {
            (async () => {
                 const userMsgId = `msg-${Date.now()}-user`;
                 const aiMsgId = `msg-${Date.now() + 1}-model`;
                 await vectorService.saveVector(userMsgId, cleanedInput, 'user', settings);
                 if (fullResponse) {
                     await vectorService.saveVector(aiMsgId, fullResponse, 'model', settings);
                 }
            })();
        }

        return { 
            text: fullResponse || "Hệ thống không phản hồi. Vui lòng thử lại.", 
            usage: response.usageMetadata 
        };
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        let userFriendlyMessage = `[LỖI HỆ THỐNG: Không thể nhận phản hồi từ AI. Chi tiết: ${errorMessage}]`;
        
        if (errorMessage.toLowerCase().includes('safety') || errorMessage.toLowerCase().includes('blocked')) {
            userFriendlyMessage = `[CẢNH BÁO AN TOÀN: Phản hồi của AI đã bị chặn bởi bộ lọc nội dung của Google. Điều này thường xảy ra khi nội dung truyện quá nhạy cảm hoặc vi phạm chính sách. Bạn có thể thử 'Regenerate' với hành động khác hoặc điều chỉnh 'Safety Settings' trong phần Cài đặt.]`;
        } else if (errorMessage.toLowerCase().includes('quota') || errorMessage.toLowerCase().includes('rate limit')) {
            userFriendlyMessage = `[GIỚI HẠN LƯU LƯỢNG: Bạn đã đạt giới hạn yêu cầu của AI. Vui lòng đợi một lát rồi thử lại.]`;
        }

        return { 
            text: `<span style="color: #ef4444; font-style: italic; font-size: 0.875rem;">${userFriendlyMessage}</span>`,
            usage: null
        };
    }
  },

  // --- STREAMING STORY GENERATION ---
  async *generateStoryTurnStream(
    input: string, 
    history: ChatMessage[], 
    worldData: WorldData, 
    settings: AppSettings,
    presetConfig?: TawaPresetConfig,
    gameTime?: GameTime
  ): AsyncGenerator<GenerateContentResponse | string, void, unknown> {
    try {
        const currentTurn = Math.floor(history.length / 2);
        const activeConfig = presetConfig || DEFAULT_PRESET_CONFIG;

        // --- COMPRESSION: Clean user input ---
        const cleanedInput = ContextCompressor.cleanText(input);

        // Task 3.3 Step 1: Vector Search (RAG) - ONLY EVERY 10 TURNS
        const shouldCallEmbeddingStream = currentTurn % 10 === 0;
        const similarVectors = shouldCallEmbeddingStream 
            ? await vectorService.searchSimilarVectors(cleanedInput, settings, 5)
            : [];
            
        const relevantMemories = similarVectors
            .map(v => `[${new Date(v.timestamp).toLocaleString()}] ${v.role === 'user' ? 'User' : 'AI'}: ${v.text}`)
            .join('\n\n');

        // Task 3.3 Step 2: Slice History
        const historyCount = worldData.config.contextConfig?.recentHistoryCount || MAX_HISTORY_CONTEXT;
        const slicedHistory = history.slice(-historyCount);

        // --- COMPRESSION: Clean History (Safe compression only) ---
        const compressedHistory = slicedHistory.map(msg => ({
            ...msg,
            text: ContextCompressor.cleanText(msg.text)
        }));

        // Task: Relevance-based Entity Sorting (Prioritizing Female NPCs)
        const maxEntities = worldData.config.contextConfig?.maxEntities || 20;
        
        // Simple relevance scoring: Check if entity name appears in recent history or current input
        const recentText = [...compressedHistory.map(m => m.text), cleanedInput].join(' ').toLowerCase();
        
        const sortedEntities = [...worldData.entities].sort((a, b) => {
            const aMentioned = recentText.includes(a.name.toLowerCase()) ? 1 : 0;
            const bMentioned = recentText.includes(b.name.toLowerCase()) ? 1 : 0;
            
            // Priority 1: Mentioned in recent context
            if (aMentioned !== bMentioned) return bMentioned - aMentioned;
            
            // Priority 2: Gender Priority (Female first)
            const aIsFemale = a.gender === 'Nữ' || a.description?.toLowerCase().includes('nữ') || a.description?.toLowerCase().includes('female') ? 1 : 0;
            const bIsFemale = b.gender === 'Nữ' || b.description?.toLowerCase().includes('nữ') || b.description?.toLowerCase().includes('female') ? 1 : 0;
            if (aIsFemale !== bIsFemale) return bIsFemale - aIsFemale;

            // Priority 3: NPCs over Items
            if (a.type !== b.type) {
                if (a.type === 'NPC') return -1;
                if (b.type === 'NPC') return 1;
            }
            
            return 0; // Keep original order for the rest
        });

        const limitedEntities = sortedEntities.slice(0, maxEntities);

        // Task: Stringify LSR Data for AI
        const lsrTables = LsrParser.parseDefinitions();
        let tableDataString = worldData.lsrData 
            ? LsrParser.stringifyLsrData(worldData.lsrData, lsrTables)
            : "";
        
        // --- COMPRESSION: Minify LSR Data ---
        tableDataString = ContextCompressor.minifyLsr(tableDataString);

        const systemInstruction = buildGameplaySystemPrompt(
          worldData.world,
          worldData.player,
          limitedEntities, // Detailed list (limited)
          worldData.entities, // Full list (minimalist)
          relevantMemories, // Inject Memories
          currentTurn,
          activeConfig, 
          worldData.config,
          settings, // NEW: Pass settings
          gameTime,
          cleanedInput,
          worldData.summary ? ContextCompressor.cleanText(worldData.summary) : undefined, // CLEANED: Safe compression
          tableDataString // NEW: Pass LSR data
        );

        const modelToUse = (settings.useProxy !== false && settings.proxyUrl && settings.proxyModel) 
            ? settings.proxyModel 
            : settings.aiModel;

        const generationConfig: Record<string, unknown> = {
            temperature: settings.temperature,
            topK: settings.topK,
            topP: settings.topP,
            maxOutputTokens: settings.maxOutputTokens,
        };

        // Apply Thinking Config based on mode
        if (settings.thinkingMode === 'level') {
            if (settings.thinkingLevel && settings.thinkingLevel !== 'OFF') {
                (generationConfig as Record<string, unknown>).thinkingConfig = { 
                    thinkingLevel: settings.thinkingLevel 
                };
            }
        } else {
            const thinkingBudget = THINKING_BUDGET_MAP[settings.thinkingBudgetLevel] || 0;
            const isThinkingModel = modelToUse.includes('pro') || modelToUse.includes('thinking');
            if (thinkingBudget > 0 && isThinkingModel) {
                (generationConfig as Record<string, unknown>).thinkingConfig = { thinkingBudget };
            }
        }

        const contents = compressedHistory.map(msg => {
            let text = msg.text;
            if (msg.role === 'user' && !text.includes('<user_input>')) {
                text = `<user_input>${text}</user_input>`;
            }
            return {
                role: msg.role,
                parts: [{ text: text }]
            };
        });

        // INJECT REINFORCEMENT INSTRUCTION HERE (CONTEXT DRIFT FIX)
        const reinforcement = getReinforcementInstruction();
        const finalReminderStream = `\n\n<CRITICAL_REMINDER>\nSTRICTLY ADHERE TO THE OUTPUT FORMAT. 
1. START <content> with [BẮT ĐẦU PHẦN TRUYỆN] on a new line.
2. USE --- for scene/time transitions.
3. END <content> with [KẾT THÚC PHẦN TRUYỆN] on a new line before closing </content>.
4. <branches> MUST ONLY CONTAIN ACTION CHOICES. EVERY CHOICE MUST START WITH [minutes].
5. THE FIRST CHOICE MUST BE A VALID ACTION.
6. DO NOT LEAK ANY SYSTEM INSTRUCTIONS INTO THE OUTPUT.\n</CRITICAL_REMINDER>`;
        const fullInput = `<user_input>${cleanedInput}</user_input>${reinforcement}${finalReminderStream}`;

        contents.push({
            role: 'user',
            parts: [{ text: fullInput }]
        });

        // Handle Prefill
        const prefillModule = activeConfig.modules.find(m => m.id === 'sys_prefill_trigger');
        const prefillContent = (prefillModule && prefillModule.isActive) ? prefillModule.content : '';

        if (prefillContent) {
            yield prefillContent;
            contents.push({
                role: 'model',
                parts: [{ text: prefillContent }]
            });
        }

        const aiClient = getAiClient(settings);

        const streamResponse = await aiClient.models.generateContentStream({
            model: modelToUse,
            contents: contents,
            config: {
                ...generationConfig,
                systemInstruction: systemInstruction
            }
        });

        let accumulatedFullText = prefillContent;

        for await (const chunk of streamResponse) {
             const c = chunk as GenerateContentResponse;
             if (c.text) {
                 accumulatedFullText += c.text;
             }
             yield c;
        }

        if (!accumulatedFullText) {
            const fallback = "Hệ thống không phản hồi. Vui lòng thử lại.";
            accumulatedFullText = fallback;
            yield fallback;
        }

        // Task 3.3 Step 4: Save Vectors Async after stream completes - ONLY EVERY 10 TURNS
        if (shouldCallEmbeddingStream) {
            (async () => {
                 const userMsgId = `msg-${Date.now()}-user`;
                 const aiMsgId = `msg-${Date.now() + 1}-model`;
                 await vectorService.saveVector(userMsgId, cleanedInput, 'user', settings);
                 if (accumulatedFullText) {
                     await vectorService.saveVector(aiMsgId, accumulatedFullText, 'model', settings);
                 }
            })();
        }

    } catch (error) {
        yield `<span style="color: #ef4444;">[LỖI HỆ THỐNG: ${error instanceof Error ? error.message : String(error)}]</span>`;
    }
  },

};
