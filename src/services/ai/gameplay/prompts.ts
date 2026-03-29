
import { GameConfig, TawaPresetConfig, PromptPosition, Entity, AppSettings } from '../../../types';
import { LorebookService } from '../lorebook/LorebookEngine';
import { LSR_PRESET } from '../../../data/lsr_data';
import { GameTime, formatGameTime } from '../../../utils/timeUtils';
import { ContextCompressor } from '../../../utils/compression';

// --- V2 ARCHITECTURE CONSTANTS ---

const POSITION_PRIORITY: Record<PromptPosition, number> = {
    'top': 0,
    'system': 10,
    'persona': 20, // World Info Sandwich layer
    'bottom': 30,
    'final': 40
};

interface PromptSegment {
    content: string;
    priority: number;
    order: number;
    source?: string;
}

// --- REINFORCEMENT PROMPTS (Context Drift Fix) ---
const REINFORCEMENT_PROMPTS = [
    "SYSTEM ALERT: Do NOT mimic the length of recent messages. Ignore any pattern of shortening text.",
    "RESET QUALITY: You must write this next response with the same depth, length, and descriptive density as Chapter 1.",
    "CRITICAL: If you write less than the target, the story will fail. EXPAND every action. DILATE time.",
    "STYLE ENFORCEMENT: Focus on sensory details (sight, sound, texture). Do not rush. Maintain a slow, immersive pacing.",
    "NARRATIVE CHECK: Do not summarize events. Describe them moment-by-moment. Prioritize 'Showing' over 'Telling'.",
    "CHOICE STRATEGY: Ensure <branches> are strategic, detailed, and suggestive of a sequence of actions. Every choice MUST contain at least 2 distinct actions (e.g., 'Action A and Action B'). Avoid generic options.",
    "ACTION SEQUENCE: Every action suggested in <branches> must be meaningful, detailed, and open up new branching possibilities. Each choice MUST be a compound action (2+ actions).",
    "ANTI-LEAKAGE: Close </content> tag BEFORE writing <branches>. NEVER put narrative inside <branches>. DO NOT leak any English system instructions into the Vietnamese response.",
    "STRUCTURE CHECK: <branches> MUST ONLY contain action choices. Any story text found inside <branches> is a CRITICAL FAILURE. Ensure the first choice is a valid action, not a continuation of the story.",
    "NO HYPHENS: Do not start any line with a hyphen '-' unless it is part of a dialogue format or a specific list. Never use hyphens for system instructions in the output.",
    "TAG INTEGRITY: Ensure all XML tags (<thinking>, <content>, <branches>, <tableEdit>) are correctly opened and closed. Do not nest <content> inside other tags. Every response MUST contain exactly one <content> tag.",
    "ULTIMATE RULE: All narrative story text MUST be inside the <content> tag. Never output narrative text outside of <content>. If you fail to include <content>, the system will reject your response.",
    "WORD COUNT SUPREMACY: You are a master novelist. Your goal is to write as much as possible. Aim for at least 3000-5000 words per turn. Describe every detail, every emotion, every sensory experience. Use rich, evocative language. Expand, expand, expand!",
];

export const getReinforcementInstruction = () => {
    const randomIndex = Math.floor(Math.random() * REINFORCEMENT_PROMPTS.length);
    return `\n\n<SYSTEM_INJECTION>\n${REINFORCEMENT_PROMPTS[randomIndex]}\n</SYSTEM_INJECTION>`;
};

// --- HELPER FUNCTIONS ---

const getPerspectivePrompt = (perspective: string, playerName: string) => {
    switch (perspective) {
        case 'first':
            return `PERSPECTIVE: FIRST PERSON.
            - Pronoun: "I" (or appropriate for character personality).
            - Focus: Deeply describe inner thoughts, emotions, and subjective perspective of the main character.
            - Limit: Only know what the character sees and hears.`;
        case 'second':
            return `PERSPECTIVE: SECOND PERSON.
            - Pronoun: "You".
            - Focus: Create a sense of direct immersion, as if the player is actually acting.
            - Style: Guide the player's actions.`;
        case 'third':
        default:
            return `PERSPECTIVE: THIRD PERSON.
            - Pronoun: Call the character by name ("${playerName}"), or friendly pronouns ("He", "She", "Him", "Her").
            - NOTE: Absolutely avoid using derogatory terms when referring to the main character or friendly characters. Prioritize using names or polite, close pronouns.
            - Focus: Objective, cinematic, covering actions and the surrounding environment.`;
    }
};

/**
 * Hàm Prompt Gameplay Chính (REFACTORED V2 - TAWA ULTIMATE)
 * Uses Data-Driven Injection & Granular Modules
 * Task 3.3: Updated to accept relevantMemories string
 */
export const buildGameplaySystemPrompt = (
  worldSettings: Record<string, unknown>,
  playerProfile: Record<string, unknown>,
  entities: Entity[], // Detailed entities (limited)
  allEntities: Entity[], // Full list for minimalist NPC list
  relevantMemories: string, // RAG Context
  turnCount: number,
  presetConfig: TawaPresetConfig,
  gameConfig: GameConfig,
  appSettings?: AppSettings, // NEW: Pass global app settings
  gameTime?: GameTime, // Task: Time System
  lastUserMessage: string = "",
  summary?: string, // NEW: Summary Memory
  tableData: string = "" // NEW: LSR Table Data
) => {
  // --- BƯỚC 0: LOREBOOK PROCESSING (LSR) ---
  const lorebookEntries = LorebookService.loadLorebook(LSR_PRESET);
  
  const lsrDynamicVars = {
      'user': playerProfile.name,
      'tableData': tableData || "(Chưa có dữ liệu trạng thái thế giới)"
  };

  const lsrPromptContent = LorebookService.scanAndActivate(lastUserMessage, lorebookEntries, lsrDynamicVars);


  // --- BƯỚC 1: KHỞI TẠO BIẾN (VARIABLE MAP) ---
  // Ưu tiên lấy từ AppSettings (Cài đặt hệ thống) nếu có, nếu không thì dùng GameConfig (Cấu hình thế giới)
  const activePerspective = appSettings?.perspective || gameConfig.perspective || 'third';
  const activeDifficulty = appSettings?.difficulty || gameConfig.difficulty;
  const activeOutputLength = appSettings?.outputLength || gameConfig.outputLength;
  const activeRealityDifficulty = appSettings?.realityDifficulty || "Normal";

  const contextItems = gameConfig.contextConfig?.items || {
    playerProfile: true,
    worldInfo: true,
    longTermMemory: true,
    relevantMemories: true,
    entities: true,
    npcRegistry: true,
    timeSystem: true,
    reinforcement: true
  };
  
  let minWords: number;
  let maxWords: number;
  if (activeOutputLength.id === 'custom') {
      minWords = appSettings?.customMinWords || gameConfig.customMinWords || 1000;
      maxWords = appSettings?.customMaxWords || gameConfig.customMaxWords || (minWords + 2000);
  } else {
      minWords = activeOutputLength.minWords;
      maxWords = activeOutputLength.maxWords || (minWords + 2000); 
  }

  // Build Entity Content (Detailed - Lorebook/World Info)
  const entityContent = contextItems.entities ? entities.map((e: Entity) => {
      let desc = `[${e.type}] ${e.name}`;
      if (e.description) {
          // --- COMPRESSION: Clean whitespace only ---
          const cleanDesc = e.description.replace(/\s+/g, ' ').trim();
          desc += `: ${cleanDesc}`;
      }
      if (e.type === 'NPC' && e.personality) {
          const cleanPers = e.personality.replace(/\s+/g, ' ').trim();
          desc += `\n(Tính cách: ${cleanPers})`;
      }
      return desc;
  }).join('\n\n') : "";

  // Build Minimalist NPC List (Full list)
  // --- COMPRESSION: Ultra-minimalist format ---
  const minimalistNpcList = allEntities
    .filter((e: Entity) => e.type === 'NPC')
    .map((e: Entity) => {
        return `${e.name}(${e.gender || '?'},${e.age || '?'})`;
    }).join(', ');

  const minimalistSection = (contextItems.npcRegistry && minimalistNpcList) ? `
<MINIMALIST_NPC_REGISTRY>
(This is a list of all NPCs existing in this world. Use IDs for accurate reference if needed)
${minimalistNpcList}
</MINIMALIST_NPC_REGISTRY>`.trim() : "";

  // Build Player Content
  const playerContent = contextItems.playerProfile ? ContextCompressor.cleanText(`
[Main Character Profile <user>]
- Name: ${playerProfile.name}
- Gender: ${playerProfile.gender} | Age: ${playerProfile.age}
- Personality: ${playerProfile.personality}
- Appearance: ${playerProfile.appearance}
- Background: ${playerProfile.background}
- Skills: ${playerProfile.skills}
- Goal: ${playerProfile.goal}
  `) : "";

  // Build Scenario Content
  const scenarioContent = contextItems.worldInfo ? ContextCompressor.cleanText(`
[World Context & Plot]
- World Name: ${worldSettings.worldName}
- Genre: ${worldSettings.genre}
- Setting Details: ${worldSettings.context}
  `) : "";


  // Variable Map with Defaults
  const variables: Record<string, string> = {
      'word_min': String(minWords),
      'word_max': String(maxWords),
      'output_language': 'Vietnamese',
      '42': '',
      'Tiên Đề Thế Giới': '',
      '<Writing_Style>': '',
      'POV_rules': '',
      'thinking_chain': '',
      'anti_rules': '',
      'npc_logic': '',
      'Quan hệ nhân vật': '',
      'enigma': '',
      'seeds': '',
      'outside_cot': '',
      'meow_FM': '',
      'nsfw_thinking_chain': '',
      
      'world_info': entityContent || "(Chưa có thông tin thực thể)",
      'persona': playerContent,
      'scenario': scenarioContent,

      'user': playerProfile.name,
      'user_info': playerContent,

      'status_1': '',
      'status_2': '',
      'snow': '',
      'branches': '',
      'update_variable': '',
      
      'table_Edit': lsrPromptContent 
  };

  const activeModules = presetConfig.modules; 
  const segments: PromptSegment[] = [];

  // --- BƯỚC 2: QUÉT MODULE & INJECTION (FIXED LOGIC) ---
  
  activeModules.forEach(mod => {
      // Skip prefill logic module, only for Assistant
      if (mod.id === 'sys_prefill_trigger') return;

      // Ensure content is handled even if empty (for clearing variables)
      const content = mod.isActive ? mod.content : '';

      // CASE A: Module has injectKey -> Inject or Append to Variable
      if (mod.injectKey) {
          const key = mod.injectKey;
          // Only append if active and content exists
          if (content) {
              if (variables[key]) {
                  variables[key] += "\n" + content;
              } else {
                  variables[key] = content;
              }
          }
      } 
      // CASE B: Standalone Module -> Append to Segments
      else if (mod.isActive) {
          const position = mod.position || 'bottom'; 
          const priority = POSITION_PRIORITY[position] || POSITION_PRIORITY['bottom'];
          const order = mod.order || 999; 

          segments.push({
              priority: priority,
              order: order,
              content: content,
              source: `Module:${mod.id}`
          });
      }
  });

  // --- BƯỚC 3: XỬ LÝ SEGMENTS CỐ ĐỊNH (SYSTEM OVERRIDES) ---
  
  if (playerProfile) {
    segments.push({
      priority: POSITION_PRIORITY['system'],
      order: 1, 
      content: `
<ROLEPLAY_INSTRUCTION>
User is roleplaying the main character named "${playerProfile.name}".
- When referring to "${playerProfile.name}", it is the User.
- All thoughts and actions of "${playerProfile.name}" are controlled by the User or guided by the AI from the User's perspective.
- Absolutely DO NOT create a separate "User" character.
- Narrative perspective: 3rd person (following "${playerProfile.name}") or 2nd person (You - if config requires).
</ROLEPLAY_INSTRUCTION>`,
      source: 'RoleplayInstruction'
    });
  }

  // 0. Minimalist NPC Registry
  if (minimalistSection) {
      segments.push({
          priority: POSITION_PRIORITY['system'],
          order: 2, // Right after roleplay instruction
          content: minimalistSection,
          source: 'MinimalistRegistry'
      });
  }

    // Strategic Thinking & Character Vitality Segment
    segments.push({
        priority: POSITION_PRIORITY['system'],
        order: 5,
        content: `
<STRATEGIC_PLANNING>
Before writing responses and branching choices, analyze:
1. **Current Goal**: What is the main character trying to achieve?
2. **Obstacles/Issues**: What difficulties are hindering them? (Enemies, environment, psychology, etc.)
3. **Suggestiveness**: What directions will the next choices open for the plot?
=> Ensure choices in <branches> reflect the results of this analysis.
</STRATEGIC_PLANNING>

<CHARACTER_VITALITY_AND_EVOLUTION>
1. **MC & NPC are living entities**: Every character has their own thoughts, desires, fears, and goals. They are not puppets waiting for player actions.
2. **Evolution**: Characters must change (more or less) over time. Change can be in skills, attitude, psychology, or social status based on events.
3. **NPC Autonomy**: NPCs can take initiative, make offers, or change attitudes without MC's prior impact. They have their own lives running in parallel.
4. **Dynamic Relationships**: Affection and trust between characters are a flow, which can deepen or crack based on every word and action.
5. **Objective Existence**: The world continues even when MC is not observing. NPCs still perform daily activities and pursue their purposes.
</CHARACTER_VITALITY_AND_EVOLUTION>

<CRITICAL_CHOICE_FORMATTING>
⚠️ ACTION CHOICES (<branches>) MUST ADHERE TO THESE RULES:
1. **NO DIALOGUE**: Absolutely FORBIDDEN to put dialogue (e.g., [Name]: "...") inside <branches>.
2. **NO NARRATIVE**: Absolutely FORBIDDEN to put story descriptions, emotional states, or narrative text inside <branches>.
3. **FORMAT**: Each choice MUST be a concise but detailed compound action, e.g., "[15] Explore the dark corridor and search for hidden levers".
4. **COMPOUND ACTIONS (MANDATORY)**: Every choice MUST contain at least 2 distinct actions linked together. Avoid single, simple actions.
5. **PURITY**: If you have more story to tell, put it in <content> BEFORE closing it. <branches> is ONLY for the final menu of choices.
6. **FIRST CHOICE CHECK**: Ensure the very first line after <branches> is a valid action. Do not start with a sigh, a thought, or a description.
</CRITICAL_CHOICE_FORMATTING>`,
        source: 'CharacterVitality'
    });

  // 1. RAG Memories (Task 3.3)
  // Inject relevant memories from Vector Search
  if (contextItems.relevantMemories && relevantMemories && relevantMemories.trim().length > 0) {
      segments.push({ 
          priority: POSITION_PRIORITY['persona'], 
          order: 99, 
          content: ContextCompressor.cleanText(`
<RELEVANT_PAST_CONTEXT>
(The system has retrieved relevant memories from the past, use them to maintain consistency)
${relevantMemories}
</RELEVANT_PAST_CONTEXT>`), 
          source: 'Memories' 
      });
  }

  // 1.5. Summary Memory (Long Term)
  if (contextItems.longTermMemory && summary && summary.trim().length > 0) {
      segments.push({
          priority: POSITION_PRIORITY['persona'],
          order: 50, // Before RAG, after World Info
          content: `
<LONG_TERM_MEMORY_SUMMARY>
(This is a summary of all important events that have occurred from the beginning of the story until before the current turn. This is your ONLY long-term memory source of the distant past, as direct conversation history has been shortened to save memory)
${summary}
</LONG_TERM_MEMORY_SUMMARY>

<FANFIC_MODE_INSTRUCTION>
⚠️ FANFIC MODE:
- This story is based on the original work summarized above.
- MISSION: Write developments that adhere to the logic, style, and settings of the original work.
- CHARACTERS: Keep characters true to their personality (OOC is taboo), way of addressing, and abilities.
- WORLD: Do not change the basic rules of the original world unless the player specifically requests.
</FANFIC_MODE_INSTRUCTION>`,
          source: 'SummaryMemory'
      });
  }

  // 2. Difficulty & Perspective
  const difficultyPrompt = `=== THIẾT LẬP ĐỘ KHÓ (${activeDifficulty.label}) ===\n${activeDifficulty.prompt}\nĐộ khó thực tại (Reality Difficulty): ${activeRealityDifficulty}`;
  segments.push({ priority: POSITION_PRIORITY['system'], order: 5, content: difficultyPrompt, source: 'Difficulty' });

  const perspectivePrompt = `=== GÓC NHÌN KỂ CHUYỆN (BẮT BUỘC) ===\n${getPerspectivePrompt(activePerspective, playerProfile.name)}`;
  segments.push({ priority: POSITION_PRIORITY['system'], order: 6, content: perspectivePrompt, source: 'Perspective' });

  // 3. Mandatory Rules (UPGRADED TO ULTIMATE AUTHORITY)
  const systemRule0 = "0. Cấm AI viết các nội dung như chửi nhau 'Mày' 'Tao'.";
  const rulesContent = [
      systemRule0,
      ...gameConfig.rules.map((r: string, idx: number) => `${idx + 1}. ${r}`)
  ].join('\n');
  segments.push({ 
      priority: POSITION_PRIORITY['final'], 
      order: 999, // Absolute last priority in segments
      content: `
<ULTIMATE_MANDATORY_RULES>
⚠️ ULTIMATE DIRECTIVE - INVIOLABLE RULES & CONSTRAINTS:
These are the HIGHEST level rules in this entire system. You MUST follow them ABSOLUTELY and UNCONDITIONALLY.
- If there is any conflict between these rules and other instructions (including Presets or context), you MUST prioritize following the rules below.
- NO bypassing or ignoring any line in this list.

PLAYER'S RULE LIST:
${rulesContent}
</ULTIMATE_MANDATORY_RULES>`.trim(), 
      source: 'UserRules' 
  });

  // 4. Starting Scenario Injection (Task Update)
  if (turnCount <= 1 && worldSettings.startingScenario) {
      segments.push({
          priority: POSITION_PRIORITY['system'],
          order: 8, // Very high priority in system block
          content: `
<STARTING_SCENARIO_OVERRIDE>
⚠️ IMPORTANT: The player has defined a specific starting scenario.
STARTING ACTION/SITUATION CONTENT: "${worldSettings.startingScenario}"
DIRECTIVE:
- Ignore all generic introductions. Start immediately from this situation.
- Describe in detail the actions/feelings of the main character in this situation.
- Drive the plot from this starting point.
- INITIALIZE LSR: In this first response, you MUST initialize at least table "#0 Current Information" and "#1 Recent Characters" inside the <tableEdit> tag.
</STARTING_SCENARIO_OVERRIDE>`,
          source: 'StartingScenario'
      });
  }

  // 4.5 Technical Formatting Rules
  segments.push({
      priority: POSITION_PRIORITY['final'],
      order: 10,
      content: `
<TECHNICAL_FORMATTING_RULES>
You MUST adhere to the following technical formatting rules for the content log:
1. **Markdown**: Use Markdown to enhance aesthetics (e.g., **bold** for emphasized words, *italic* for inner thoughts).
2. **No Repetition**: Absolutely DO NOT repeat the player's action at the beginning of the response. Start with the reaction or result of that action.
3. **Paragraph Structure**: Divide content into short paragraphs (3-5 sentences each). Use 2 line breaks between paragraphs to create readable space. Absolutely DO NOT write a long continuous block of text without breaks.
4. **Dialogue Formatting (CRITICAL)**: 
   - When writing dialogue for ANY character (including the MC), you MUST always use the format: [Character Name]: "[Dialogue]" or [Character Name] nói: "[Dialogue]".
   - **MANDATORY**: Each dialogue block MUST be on its own line. Absolutely DO NOT mix narrative text and dialogue on the same line.
   - **MANDATORY**: Use exactly 2 line breaks before and after every dialogue block to ensure clear separation from narrative paragraphs.
   - Example: 
     Trần Thiên Vũ: "Chào em."
     
     Lý Mạc Sầu nói: "Ngươi là ai?"
   - For the Main Character (MC), use their name: ${playerProfile.name}: "...".
   - Absolutely DO NOT write dialogue in quotes without a preceding character name. This is to ensure the UI can correctly identify the speaker.
   - If the dialogue is from an unknown source or a generic narrator voice, use Người dẫn chuyện: "..." or Giọng nói lạ: "...".
5. **Dialogue Expansion (MC/PC)**: You ARE ALLOWED and ENCOURAGED to write dialogue for all characters, including the main character <user>. When the player provides an action containing dialogue, rewrite that dialogue in a detailed, polished, and content-rich way in the log. You can create additional dialogue for the MC if deemed necessary for the plot flow. However, absolutely FORBIDDEN to unilaterally decide actions or change the will/choices of the player.
6. **Response Structure (MANDATORY)**:
   - Wrap the entire story development in <content></content> tags.
   - The <content> tag MUST contain the entire narrative/story response for the current turn.
   - The <table_stored> or <tableEdit> tag MUST be AFTER the </content> tag and BEFORE the <branches> tag.
   - The <time_cost> and <set_time> tags MUST be after the </content> tag, before the <branches> tag.
   - The <branches> tag MUST be the final content of the entire response and MUST ONLY contain action choices.
   - **CRITICAL**: NEVER put story narrative, descriptions, dialogue, or LSR DATA (like #0 Current Information|0:...) inside the <branches> tag.
   - If you have more story to tell, put it in the <content> tag before the <branches> tag.
   - Format of <branches>:
     <branches>
     [Action 1]
     [Action 2]
     [Action 3]
     </branches>
7. **No Leakage (CRITICAL)**: 
   - Absolutely DO NOT explain system tags.
   - Absolutely DO NOT output text blocks like "[Loading Constitution]", "[DATA SYNC]", "[SYNCHRONIZATION]", "[INPUT DECODING]", "[Loading Variables]", etc.
   - **MANDATORY**: Do NOT leak any English instructions, reinforcement prompts, or system keywords into the final response to the player. These steps MUST be processed internally in "thinking" and NEVER appear in the final response.
   - All progress check notes, word count goals, segments (from word_count module) MUST be placed inside <thinking> tags and ABSOLUTELY MUST NOT appear outside these tags or in <content> tags.
   - All world status information MUST be updated via the <tableEdit> tag in LSR format (#ID Name|0:Val|1:Val).
   - Example of standard LSR format:
     <tableEdit>
     #0 Current Information|0:10:30 AM, 15/04/1204|1:Border Tea Party
     #1 Recent Characters|0:Tran Thien Vu|1:0|2:Diplomatic Negotiation|3:Banquet Room|4:Sitting|5:Normal|6:Noble Suit|7:Using "God-Level Acting"
     </tableEdit>
8. **End of Response**: After closing the </branches> tag, you MUST stop the response immediately. Absolutely DO NOT write any additional text, notes, or instructions after this tag.
9. **Cumulative Summary (MANDATORY)**:
   - After each response, you MUST update the cumulative summary (Incremental Summary) of the entire story up to the current moment.
   - This summary must include the most important events so far, plus what just happened in this turn.
   - **Length Requirement**: MUST list from 8 to 15 most important events/details.
   - **Style Requirement**: Use BULLET POINTS format to summarize events clearly, concisely but fully. This is extremely important because the main response can be very long (5,000 - 15,000 words).
   - This summary MUST be placed in <incrementalSummary></incrementalSummary> tags.
   - This tag must be after the </content> tag and before the <branches> tag.
10. **Structural Markers (BẮT BUỘC)**:
    - BẮT ĐẦU thẻ <content> bằng dấu mốc [BẮT ĐẦU PHẦN TRUYỆN] trên một dòng riêng biệt.
    - Sử dụng --- (ba dấu gạch ngang) trên một dòng riêng biệt để phân tách các thay đổi cảnh lớn hoặc bước nhảy thời gian trong <content>.
    - Ở CUỐI CÙNG của thẻ <content>, ngay trước thẻ đóng </content>, bạn PHẢI viết dấu mốc [KẾT THÚC PHẦN TRUYỆN] trên một dòng riêng. Đây là ranh giới cứng để ngăn rò rỉ nội dung vào các thẻ sau đó.
    - TUYỆT ĐỐI KHÔNG đặt các dấu mốc cấu trúc này bên trong thẻ <branches>. Thẻ <branches> CHỈ được chứa các lựa chọn hành động thực sự.
</TECHNICAL_FORMATTING_RULES>`,
      source: 'TechnicalRules'
  });

  // 5. Status & Time Instruction
  if (contextItems.timeSystem) {
    const timeString = gameTime ? formatGameTime(gameTime) : "Unknown";
    segments.push({
        priority: POSITION_PRIORITY['bottom'],
        order: -10,
        content: `
=== TIME SYSTEM (IMPORTANT) ===
- Current game time: ${timeString}
- Current turn: ${turnCount}

TIME & ACTION DIRECTIVE:
1. You are the ONLY one who decides the time elapsed for each player action.
2. **NARRATIVE CONSISTENCY**: You MUST ensure the story content (atmosphere, lighting, character activities, environment) strictly matches the "Current game time" provided above. 
   - Pay close attention to the day of the week (e.g., "Thứ Hai", "Chủ Nhật") provided in the "Current game time". If the day of the week is important for the story (e.g., a market day, a religious festival), you MUST respect it.
   - If it's night, describe the darkness, stars, or artificial lights. 
   - If it's morning, describe the sunrise or the world waking up.
   - Adjust the mood and sensory details to fit the specific hour and date.
3. SPECIAL NOTE: If the current time is "January 01, 2024", this is a TEMPORARY (placeholder) timestamp. 
   - In the FIRST response, you MUST choose a suitable starting timestamp based on the world context.
   - Use the <set_time>year|month|day|hour|minute</set_time> tag to set this starting point. Example: <set_time>1250|12|25|06|30</set_time>.
4. For subsequent turns, determine the minutes spent based on the action and return the <time_cost>X</time_cost> tag at the end of the response.
5. BRANCHING RULES (<branches>):
   - You MUST provide at least 3-4 next action choices inside the <branches></branches> tags. You can create more action choices if necessary (up to 6-8) to provide more variety and depth.
   - **CHOICE LOGIC (CRITICAL)**: Create actions based on the events that just occurred.
   - **PLAYER-CENTRIC ONLY**: Every choice MUST be an action that the PLAYER (MC) can take. 
   - **NO NARRATIVE**: Absolutely DO NOT include narrative descriptions of NPC actions, environment changes, or character feelings as choices. (e.g., DO NOT use "The villagers are angry..." as a choice).
   - **ACTION FORMAT (CRITICAL)**: Every choice MUST start with a time cost in brackets followed by a detailed compound action (at least 2 actions per choice).
   - **TIME COST REQUIRED**: Each choice MUST be accompanied by an estimated time: "[minutes] Action A and Action B". (e.g., "[10] Search the room for clues and try to unlock the safe", "[5] Talk to the guard and offer him a bribe"). Choices without [minutes] will be filtered out and NOT shown to the player.
   - **DETAIL & PROGRESSION**: Action content MUST be detailed, creative, and suggestive of a sequence of actions (opening up new possibilities and branching paths).
   - Each choice MUST be accompanied by an estimated time: "[minutes] Action A and Action B".
   - **STRICT SEPARATION**: The <branches> tag is ONLY for action choices. NEVER put story narrative, descriptions, dialogue, or LSR DATA (like #0 Current Information|0:...) inside the <branches> tag. All story content MUST be in the <content> tag.
   - **IMPORTANT WARNING**: Absolutely DO NOT nest any other system tags (such as <set_time>, <time_cost>, <finish>) inside the <branches> tags.
   - You MUST close the </branches> tag immediately after listing the choices.
   - Standard structure example:
     <content>
     (Story content here...)
     </content>
     <time_cost>15</time_cost>
     <branches>
     [15] Persuade the guard...
     [30] Find a way around the alley...
     </branches>
6. Always end the response by providing all necessary system tags in order: </content> -> <time_cost> -> <set_time> (if needed) -> <branches>.
`,
        source: 'GameStatus'
    });
  }

  // INJECT REINFORCEMENT INSTRUCTION HERE (CONTEXT DRIFT FIX)
  if (contextItems.reinforcement) {
    const reinforcement = getReinforcementInstruction();
    segments.push({
        priority: POSITION_PRIORITY['final'],
        order: 100,
        content: reinforcement,
        source: 'Reinforcement'
    });
  }
  
  const replaceVariables = (text: string, depth = 0): string => {
      if (depth > 5) return text; 

      let processed = text;
      const tawaRegex = /\{\{(?:getvar|getglobalvar)::(.*?)\}\}/g;
      const stRegex = /\{\{([^:]*?)\}\}/g;

      let hasMatch = false;

      processed = processed.replace(tawaRegex, (match, key) => {
          hasMatch = true;
          const cleanKey = key.trim();
          return variables[cleanKey] !== undefined ? variables[cleanKey] : '';
      });

      processed = processed.replace(stRegex, (match, key) => {
          if (key.includes('::')) return match;
          
          hasMatch = true;
          const cleanKey = key.trim();
          return variables[cleanKey] !== undefined ? variables[cleanKey] : match; 
      });

      if (hasMatch) {
          return replaceVariables(processed, depth + 1);
      }
      return processed;
  };

  segments.push({
      content: `<CRITICAL_OUTPUT_STRUCTURE_INTEGRITY>
1. **TAG ORDER**: You MUST follow this exact sequence: <thinking> -> <content> -> <incrementalSummary> -> <tableEdit> -> <time_cost> -> <set_time> -> <branches>.
2. **STRICT CONTENT SEPARATION**: 
   - <content> is for STORY ONLY.
   - <branches> is for ACTIONS ONLY.
   - Any narrative text, dialogue, description, or LSR DATA found OUTSIDE of <content> or INSIDE <branches> will cause a system crash.
3. **NO SYSTEM ARTIFACTS**: Do NOT output "Word count", "Progress", "Goal", or any other meta-commentary in the final response. These belong in <thinking> ONLY.
4. **MANDATORY CLOSURE**: Every tag you open MUST be closed. Never leave a tag unclosed.
</CRITICAL_OUTPUT_STRUCTURE_INTEGRITY>`,
      priority: POSITION_PRIORITY.final,
      order: 100,
      source: 'IntegrityRules'
  });

  segments.forEach(seg => {
      seg.content = ContextCompressor.cleanText(replaceVariables(seg.content));
  });

  // --- BƯỚC 5: SẮP XẾP & KẾT XUẤT ---

  segments.sort((a, b) => {
      if (a.priority !== b.priority) {
          return a.priority - b.priority;
      }
      return (a.order || 0) - (b.order || 0);
  });

  const finalPrompt = segments.map(s => s.content).join('\n\n');
  
  const supremeCommand = minWords >= 5000 ? "\n\n<SUPREME_COMMAND>\nThis is a SUPREME COMMAND from the user: You MUST write at least 5000 words in the <content> tag. This is non-negotiable. Expand every action, describe every sensation, and explore every thought in extreme detail. If you write less than 5000 words, the story will be considered a failure.\n</SUPREME_COMMAND>" : "";

  return `
<GLOBAL_LANGUAGE_SETTING>
CRITICAL: ALL responses to the player MUST be in Vietnamese. 
The instructions below are in English for your better understanding, but the output story, dialogue, and choices MUST be in Vietnamese.
</GLOBAL_LANGUAGE_SETTING>

` + finalPrompt + supremeCommand + `\n\n(REMINDER: STRICTLY ADHERE TO WORD COUNT FROM ${minWords} TO ${maxWords} WORDS IN <content>. ALL OUTPUT MUST BE IN VIETNAMESE.)`;
};
