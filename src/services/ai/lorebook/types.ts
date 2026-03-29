
export interface LorebookEntry {
  uid: number | string;
  key: string[]; // Keywords
  keysecondary?: string[];
  content: string;
  comment?: string;
  
  // Activation flags
  constant: boolean;
  disable: boolean;
  
  // Sorting & Positioning
  order: number;
  position?: number; // 0: before char, 1: after char, etc. (Simplified in this engine)
  
  // Recursion settings (Optional, included for completeness)
  selective?: boolean;
  probability?: number;
}

export interface Lorebook {
  entries: Record<string, LorebookEntry>;
  name?: string;
  description?: string;
}
