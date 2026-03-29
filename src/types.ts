
import { GameTime } from './utils/timeUtils';

export interface SaveFile {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  data: Record<string, unknown> | string; 
  _compressed?: boolean;
}

export interface SafetySetting {
  category: string;
  threshold: string;
}

export type ThinkingBudgetLevel = 'auto' | 'low' | 'medium' | 'high';
export type ThinkingLevel = 'OFF' | 'LOW' | 'MEDIUM' | 'HIGH';

export interface AppSettings {
  soundVolume: number;
  musicVolume: number;
  theme: 'dark' | 'light';
  fontSize: number;
  systemFont: string;
  realityDifficulty: string;
  contentBeautify: boolean;
  visualEffects: boolean;
  fullScreenMode: boolean;
  safetySettings?: SafetySetting[];
  aiModel: string;
  // Game Configuration (Moved from World Creation)
  perspective: NarrativePerspective;
  difficulty: DifficultyLevel;
  outputLength: OutputLength;
  customMinWords?: number;
  customMaxWords?: number;
  // Advanced AI Params
  contextSize: number;
  maxOutputTokens: number;
  temperature: number;
  topK: number;
  topP: number;
  thinkingBudgetLevel: ThinkingBudgetLevel;
  thinkingLevel: ThinkingLevel;
  thinkingMode: 'budget' | 'level'; // Để xác định cái nào đang hoạt động
  // New Settings
  streamResponse: boolean;
  geminiApiKey?: string[];
  proxyUrl?: string;
  proxyKey?: string;
  proxyModel?: string;
  proxyModels?: string[];
  proxyName?: string;
  proxyUrl2?: string;
  proxyKey2?: string;
  proxyModel2?: string;
  proxyModels2?: string[];
  proxyName2?: string;
  useGeminiApi: boolean;
  useProxy: boolean;
}

export interface SystemLog {
  id?: number;
  timestamp: number;
  message: string;
  type: 'info' | 'error' | 'warning';
}

export enum GameState {
  MENU = 'MENU',
  WORLD_CREATION = 'WORLD_CREATION',
  PLAYING = 'PLAYING',
  SETTINGS = 'SETTINGS',
  FANFIC = 'FANFIC'
}

export interface DifficultyLevel {
  id: string;
  label: string;
  prompt: string;
}

export interface OutputLength {
  id: string;
  label: string;
  minWords: number;
  maxWords?: number;
}

// --- NEW WORLD CREATION TYPES ---

export type EntityType = 'NPC' | 'LOCATION' | 'FACTION' | 'ITEM' | 'CUSTOM';

export interface Entity {
  id: string;
  type: EntityType;
  name: string;
  description: string;
  avatar?: string; // URL or Base64 from Image Library
  // Specific fields
  personality?: string; // Only for NPC
  age?: string;        // Added for NPC
  gender?: string;     // Added for NPC
  rarity?: string;     // Added for ITEM
  price?: string;      // Added for ITEM
  customType?: string; // Only for CUSTOM
}

export interface PlayerProfile {
  name: string;
  gender: string;
  age: string;
  personality: string;
  background: string;
  appearance: string;
  skills: string;
  goal: string;
  avatar?: string; // URL or Base64 from Image Library
}

export interface WorldSettingConfig {
  worldName: string;
  genre: string;
  context: string;
  startingScenario?: string; // New field for custom opening action
}

export type NarrativePerspective = 'first' | 'second' | 'third';

export interface GameConfig {
  difficulty: DifficultyLevel;
  outputLength: OutputLength;
  customMinWords?: number;
  customMaxWords?: number;
  rules: string[];
  perspective: NarrativePerspective; // New field for POV
  contextConfig?: ContextWindowConfig; // New field for Context Window settings
}

export interface ContextWindowConfig {
  items: {
    playerProfile: boolean;
    worldInfo: boolean;
    longTermMemory: boolean;
    relevantMemories: boolean;
    entities: boolean;
    npcRegistry: boolean;
    timeSystem: boolean;
    reinforcement: boolean;
  };
  maxEntities: number;
  recentHistoryCount: number;
}

export interface WorldData {
  player: PlayerProfile;
  world: WorldSettingConfig;
  config: GameConfig;
  entities: Entity[];
  gameTime?: GameTime; // Hệ thống thời gian
  summary?: string; // Trí nhớ tóm tắt tích lũy
  lsrData?: Record<string, unknown[]>; // Long-term State Representation (LSR)
  // Optional state for loading saved games
  savedState?: {
    history: ChatMessage[];
    turnCount: number;
    gameTime?: GameTime;
    aiMonitor?: {
      tokenHistory: {tokens: number, words: number, timestamp: number}[];
      totalTokens: number;
      lastTurnTotalTime: number;
    };
  };
}

export interface ImageMetadata {
  id: string;
  name: string;
  data: string; // Base64 data
  type: string; // mime type
  size: number;
  width?: number;
  height?: number;
  timestamp: number;
}

// Navigation Prop Type
export interface NavigationProps {
  onNavigate: (state: GameState) => void;
  onGameStart?: (data: WorldData) => void;
  onUpdateWorld?: (data: Partial<WorldData>) => void; // NEW: Callback to update world data
  onImportSetup?: (data: WorldData) => void; // New prop for importing setup only
  activeWorld?: WorldData | null;
}

export interface GameSnapshot {
  player: PlayerProfile;
  entities: Entity[];
  gameTime: GameTime;
  summary?: string;
  lsrData?: Record<string, unknown[]>;
  turnCount: number;
  dynamicRules?: string[];
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
  timestamp: number;
  gameTime?: GameTime; // Thời gian tại thời điểm tin nhắn
  choices?: string[]; // Added field to persist action choices
  userAction?: string; // Hành động người chơi đã chọn dẫn đến lượt này
  turnNumber?: number; // Số lượt của tin nhắn này
  // New fields for Swipe/Regenerate
  swipes?: string[]; // Array of message variations
  swipeIndex?: number; // Index of the currently shown message
  incrementalSummary?: string; // Tóm tắt tích lũy của cốt truyện tính đến lượt này
}

// --- NEW TAWA PRESET TYPES (REFACTOR V2) ---

// Define the depth layers for the prompt construction
export type PromptPosition = 'top' | 'system' | 'persona' | 'bottom' | 'final';

export interface PromptModule {
  id: string;
  label: string;
  isActive: boolean; // Trạng thái Bật/Tắt
  content: string;   // Nội dung Prompt
  isCore?: boolean;  // Đánh dấu nếu đây là COT (không thể tắt)
  
  // V2 Architecture Fields
  injectKey?: string; // Nếu có, nội dung sẽ được tiêm vào {{getvar::key}} thay vì nối chuỗi
  position?: PromptPosition; // Vị trí ưu tiên trong chuỗi prompt (nếu không có injectKey)
  order?: number; // Thứ tự sắp xếp chi tiết trong cùng một position (nhỏ xếp trước)
}

export interface TawaPresetConfig {
  cot: PromptModule; // Lõi tư duy
  modules: PromptModule[]; // Danh sách các module
}
