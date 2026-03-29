
export const buildWorldCreationPrompt = (fieldName: string, currentContext: Record<string, unknown>, userInput?: string) => {
  // MODE B: ENRICH / EXPAND (When user input is provided)
  if (userInput && userInput.trim().length > 0) {
    return `TASK: Rewrite and expand the following User Input for the field "${fieldName}".
CONTEXT: ${JSON.stringify(currentContext)}
USER INPUT: "${userInput}"

INSTRUCTIONS:
1. Enhance the vocabulary and descriptive quality.
2. Make it sound professional and fitting for a fantasy/sci-fi setting.
3. OUTPUT ONLY THE FINAL CONTENT. NO META-COMMENTARY.`;
  }

  // MODE A: CREATE NEW (When input is empty)
  return `
  Task: Create content for the data field: "${fieldName}".
  Current Context: ${JSON.stringify(currentContext)}
  
  Requirements:
  - Return ONLY the content of that field. No explanation, no introduction or conclusion.
  - Creative, unique, avoid clichés.
  - If field is "worldName": Create a poetic, symbolic, or evocative name that fits the genre and context. Avoid generic names like "Thế giới X".
  - Language: Vietnamese.
  `;
};

export const getWorldCreationSystemInstruction = (category: 'player' | 'world' | 'entity', field: string, userInput?: string) => {
  // SYSTEM INSTRUCTION FOR MODE B (ENRICH)
  if (userInput && userInput.trim().length > 0) {
    return `You are an expert editor and creative writer. Your task is to polish, expand, and enrich the user's rough idea into a high-quality description.

Strict Constraints:
1. Zero Conversational Filler: DO NOT say "Here is the improved version", "Based on your input", etc. Just return the final content.
2. Domain Isolation: Ensure the content fits the definition of field "${field}". Do not change the type of information (e.g. do not turn a Skill into an Appearance description).
3. Content Fidelity: Keep the core characteristics defined in the user input.
4. Language: Vietnamese.`;
  }

  // SYSTEM INSTRUCTION FOR MODE A (CREATE NEW) - Old logic
  if (category === 'player') {
    return `You are a professional RPG character creation assistant.
Task: Write content for the data field [${field}] of the main character.
Output Rules:
- Return ONLY the descriptive content. DO NOT write an introduction.
- Language: Vietnamese.
- Style: Creative, deep, fitting for the character setting.`;
  } 
  
  if (category === 'world') {
    return `You are a virtual world architect (World Builder).
Task: Write a detailed description for [${field}] of the world.
Output Rules:
- Return ONLY the main content. DO NOT write an introduction.
- If field is "worldName": Be extremely creative. Use metaphors, ancient languages, or symbolic terms.
- Language: Vietnamese.
- Style: Grand, logical, evocative of imagination.`;
  } 
  
  // Entity
  return `You are a creator of NPC content and events for RPG Games.
Task: Write [${field}] for an entity in the game.
Output Rules:
- Return ONLY the main content. DO NOT write an introduction.
- Language: Vietnamese.`;
};
