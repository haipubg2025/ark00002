
import { openDB, DBSchema, IDBPDatabase } from 'idb';
import { AppSettings, SaveFile, SystemLog, ImageMetadata } from '../../types';
import { DEFAULT_SAFETY_SETTINGS, DIFFICULTY_LEVELS, OUTPUT_LENGTHS } from '../../constants/promptTemplates';
import { CompressionUtils } from '../../utils/compression';

export interface VectorData {
  id: string;
  text: string;
  embedding: number[];
  timestamp: number;
  role: 'user' | 'model'; // Added to distinguish who said what
}

interface RPGDatabase extends DBSchema {
  saves: {
    key: string;
    value: SaveFile;
  };
  settings: {
    key: string;
    value: AppSettings;
  };
  logs: {
    key: number;
    value: SystemLog;
    autoIncrement: true;
  };
  vectors: {
    key: string;
    value: VectorData;
  };
  assets: {
    key: string;
    value: { id: string; data: string; timestamp: number };
  };
  images: {
    key: string;
    value: ImageMetadata;
  };
}

const DB_NAME = 'aetheria-rpg-db';
const DB_VERSION = 4; // Upgraded version to add images store

export const DEFAULT_SETTINGS: AppSettings = {
    soundVolume: 50,
    musicVolume: 50,
    theme: 'dark',
    fontSize: 16,
    systemFont: 'Inter',
    realityDifficulty: 'Normal',
    contentBeautify: true,
    visualEffects: true,
    fullScreenMode: false,
    safetySettings: DEFAULT_SAFETY_SETTINGS,
    aiModel: 'gemini-3-flash-preview',
    // Game Configuration Defaults
    perspective: 'third',
    difficulty: DIFFICULTY_LEVELS[1], // Normal
    outputLength: OUTPUT_LENGTHS[4], // Supreme (5000 - 15000 từ)
    customMinWords: 1000,
    customMaxWords: 3000,
    // Advanced AI Params
    contextSize: 2000000, 
    maxOutputTokens: 65000, 
    temperature: 0.9,
    topK: 500,
    topP: 0.95,
    thinkingBudgetLevel: 'high',
    thinkingLevel: 'HIGH',
    thinkingMode: 'level',
    streamResponse: true,
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
};

class DatabaseService {
  private dbPromise: Promise<IDBPDatabase<RPGDatabase>>;

  constructor() {
    this.dbPromise = openDB<RPGDatabase>(DB_NAME, DB_VERSION, {
      upgrade(db) {
        if (!db.objectStoreNames.contains('saves')) {
          db.createObjectStore('saves', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('settings')) {
          db.createObjectStore('settings');
        }
        if (!db.objectStoreNames.contains('logs')) {
          db.createObjectStore('logs', { keyPath: 'id', autoIncrement: true });
        }
        // Task 3.1: Add vectors store
        if (!db.objectStoreNames.contains('vectors')) {
          const vectorStore = db.createObjectStore('vectors', { keyPath: 'id' });
          // Optional: Add index for timestamp if needed for cleanup later
          vectorStore.createIndex('timestamp', 'timestamp');
        }
        if (!db.objectStoreNames.contains('assets')) {
          db.createObjectStore('assets', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('images')) {
          const imageStore = db.createObjectStore('images', { keyPath: 'id' });
          imageStore.createIndex('timestamp', 'timestamp');
        }
      },
    });
  }

  async checkConnection(): Promise<boolean> {
    try {
      const db = await this.dbPromise;
      return !!db;
    } catch {
      return false;
    }
  }

  async logEvent(message: string, type: 'info' | 'error' | 'warning' = 'info') {
    const db = await this.dbPromise;
    await db.add('logs', {
      timestamp: Date.now(),
      message,
      type
    } as SystemLog);
  }

  async getSettings(): Promise<AppSettings> {
    const db = await this.dbPromise;
    const settings = await db.get('settings', 'user_settings');
    
    if (!settings) {
        return DEFAULT_SETTINGS;
    }

    // Merge defaults with saved settings to ensure new fields exist
    return { ...DEFAULT_SETTINGS, ...settings };
  }

  async saveSettings(settings: AppSettings): Promise<void> {
    const db = await this.dbPromise;
    await db.put('settings', settings, 'user_settings');
  }

  async hasSaves(): Promise<boolean> {
    const db = await this.dbPromise;
    const count = await db.count('saves');
    return count > 0;
  }

  async saveGameState(saveData: SaveFile): Promise<void> {
    const db = await this.dbPromise;
    // Compress the data part of the save file to save space
    const originalData = JSON.stringify(saveData.data);
    const compressedData = CompressionUtils.compress(originalData);
    
    // Create a copy with compressed data
    const compressedSave: SaveFile = {
      ...saveData,
      data: compressedData,
      _compressed: true // Flag to indicate compression
    };
    
    await db.put('saves', compressedSave);
  }

  async saveAutosave(saveData: SaveFile): Promise<void> {
    await this.saveGameState(saveData);
  }

  async getAllSaves(): Promise<SaveFile[]> {
    const db = await this.dbPromise;
    const rawSaves = await db.getAll('saves');
    
    return rawSaves.map((save: SaveFile) => {
      if (save._compressed && typeof save.data === 'string') {
        try {
          const decompressedData = CompressionUtils.decompress(save.data);
          return {
            ...save,
            data: JSON.parse(decompressedData),
            _compressed: undefined
          };
        } catch (e) {
          console.error('Failed to decompress save data:', e);
          return save;
        }
      }
      return save;
    });
  }

  async deleteSave(id: string): Promise<void> {
    const db = await this.dbPromise;
    await db.delete('saves', id);
  }

  async clearAllSaves(): Promise<void> {
    const db = await this.dbPromise;
    await db.clear('saves');
  }

  // --- Vector Operations ---

  async saveVector(vectorData: VectorData): Promise<void> {
    const db = await this.dbPromise;
    await db.put('vectors', vectorData);
  }

  async getVector(id: string): Promise<VectorData | undefined> {
    const db = await this.dbPromise;
    return db.get('vectors', id);
  }

  async getAllVectors(): Promise<VectorData[]> {
    const db = await this.dbPromise;
    return db.getAll('vectors');
  }

  async hasVector(id: string): Promise<boolean> {
     const db = await this.dbPromise;
     const key = await db.getKey('vectors', id);
     return !!key;
  }

  // --- Asset Operations (for large files like background images) ---

  async saveAsset(id: string, data: string): Promise<void> {
    const db = await this.dbPromise;
    await db.put('assets', { id, data, timestamp: Date.now() });
  }

  async getAsset(id: string): Promise<string | undefined> {
    const db = await this.dbPromise;
    const asset = await db.get('assets', id);
    return asset?.data;
  }

  async deleteAsset(id: string): Promise<void> {
    const db = await this.dbPromise;
    await db.delete('assets', id);
  }

  // --- Image Library Operations ---

  async saveImage(image: ImageMetadata): Promise<void> {
    const db = await this.dbPromise;
    await db.put('images', image);
  }

  async getImage(id: string): Promise<ImageMetadata | undefined> {
    const db = await this.dbPromise;
    return db.get('images', id);
  }

  async getAllImages(): Promise<ImageMetadata[]> {
    const db = await this.dbPromise;
    return db.getAll('images');
  }

  async deleteImage(id: string): Promise<void> {
    const db = await this.dbPromise;
    await db.delete('images', id);
  }

  async clearAllImages(): Promise<void> {
    const db = await this.dbPromise;
    await db.clear('images');
  }
}

export const dbService = new DatabaseService();
