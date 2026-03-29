
import { Entity, GameConfig, PlayerProfile, WorldData, WorldSettingConfig } from "../../../types";
import { GameTime, INITIAL_GAME_TIME } from "../../../utils/timeUtils";

export interface WorldCreationState {
  currentTab: number;
  player: PlayerProfile;
  world: WorldSettingConfig;
  config: GameConfig;
  entities: Entity[];
  gameTime: GameTime; // Thêm gameTime vào state
  isGenerating: boolean; // General loading state
  generatingField: string | null; // Specific field loading
}

export const initialWorldState: WorldCreationState = {
  currentTab: 0,
  player: {
    name: '',
    gender: 'Nam',
    age: '',
    personality: '',
    background: '',
    appearance: '',
    skills: '',
    goal: ''
  },
  world: {
    worldName: '',
    genre: '',
    context: '',
    startingScenario: '' // Initial empty state for Starting Scenario
  },
  config: {
    rules: [], // Clean slate, no default rules
  },
  entities: [],
  gameTime: INITIAL_GAME_TIME,
  isGenerating: false,
  generatingField: null
};

// Generate simple ID if uuid not available
const simpleId = () => Math.random().toString(36).substr(2, 9);

export type WorldCreationAction = 
  | { type: 'SET_TAB', payload: number }
  | { type: 'UPDATE_PLAYER', field: keyof PlayerProfile, value: string }
  | { type: 'UPDATE_WORLD', field: keyof WorldSettingConfig, value: string }
  | { type: 'UPDATE_CONFIG', field: keyof GameConfig, value: string[] | number | boolean }
  | { type: 'UPDATE_CUSTOM_WORDS', min: number, max: number }
  | { type: 'ADD_RULE', rule: string }
  | { type: 'REMOVE_RULE', index: number }
  | { type: 'ADD_ENTITY', entity: Omit<Entity, 'id'> }
  | { type: 'UPDATE_ENTITY', id: string, entity: Partial<Entity> }
  | { type: 'REMOVE_ENTITY', id: string }
  | { type: 'UPDATE_GAME_TIME', field: keyof GameTime, value: number }
  | { type: 'SET_GENERATING', isGenerating: boolean, field?: string | null }
  | { type: 'AUTO_FILL_ALL', payload: Partial<WorldData> }
  | { type: 'IMPORT_DATA', payload: WorldData };

export const worldCreationReducer = (state: WorldCreationState, action: WorldCreationAction): WorldCreationState => {
  switch (action.type) {
    case 'SET_TAB':
      return { ...state, currentTab: action.payload };
    
    case 'UPDATE_PLAYER':
      return { ...state, player: { ...state.player, [action.field]: action.value } };
    
    case 'UPDATE_WORLD':
      return { ...state, world: { ...state.world, [action.field]: action.value } };
    
    case 'UPDATE_CONFIG':
      return { ...state, config: { ...state.config, [action.field]: action.value } };
    
    case 'UPDATE_CUSTOM_WORDS':
      return { 
        ...state, 
        config: { 
          ...state.config, 
          customMinWords: action.min, 
          customMaxWords: action.max 
        } 
      };

    case 'ADD_RULE':
      return { ...state, config: { ...state.config, rules: [...state.config.rules, action.rule] } };

    case 'REMOVE_RULE':
      return { 
        ...state, 
        config: { 
          ...state.config, 
          rules: state.config.rules.filter((_, i) => i !== action.index) 
        } 
      };

    case 'ADD_ENTITY':
      return { 
        ...state, 
        entities: [...state.entities, { ...action.entity, id: simpleId() }] 
      };

    case 'UPDATE_ENTITY':
      return {
        ...state,
        entities: state.entities.map(e => e.id === action.id ? { ...e, ...action.entity } : e)
      };

    case 'REMOVE_ENTITY':
      return {
        ...state,
        entities: state.entities.filter(e => e.id !== action.id)
      };

    case 'UPDATE_GAME_TIME':
      return {
        ...state,
        gameTime: { ...state.gameTime, [action.field]: action.value }
      };

    case 'SET_GENERATING':
      return { 
        ...state, 
        isGenerating: action.isGenerating,
        generatingField: action.field || null
      };

    case 'AUTO_FILL_ALL': {
      // Helper function to merge only if current value is empty
      const mergeIfEmpty = (current: Record<string, string>, incoming: Record<string, string>) => {
        const result = { ...current };
        Object.keys(incoming).forEach(key => {
          if (!current[key] || (current[key].trim() === '')) {
            result[key] = incoming[key];
          }
        });
        return result;
      };

      return {
        ...state,
        player: mergeIfEmpty(state.player, action.payload.player || {}),
        world: mergeIfEmpty(state.world, action.payload.world || {}),
        // For entities and rules, we trust the AI to have merged them as instructed, 
        // but we can also do a basic check or just take the new list if it's longer
        entities: (action.payload.entities && action.payload.entities.length >= state.entities.length) 
          ? action.payload.entities 
          : state.entities,
        gameTime: (action.payload.gameTime && state.gameTime.year === 2024) // Only update if still at default
          ? action.payload.gameTime 
          : state.gameTime,
        config: { 
          ...state.config, 
          rules: action.payload.config?.rules || state.config.rules 
        }
      };
    }

    case 'IMPORT_DATA':
      return {
        ...state,
        player: action.payload.player,
        world: action.payload.world,
        config: action.payload.config,
        entities: action.payload.entities,
        gameTime: action.payload.gameTime || state.gameTime,
        // Reset generating states just in case
        isGenerating: false,
        generatingField: null
      };

    default:
      return state;
  }
};
