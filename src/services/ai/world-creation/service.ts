
import { Type } from "@google/genai";
import { getAiClient } from "../client";
import { buildWorldCreationPrompt, getWorldCreationSystemInstruction } from "./prompts";
import { AppSettings, Entity } from "../../../types";

export const worldAiService = {
  // --- WORLD CREATION ASSISTANT (STRICT LOGIC) ---

  async generateFieldContent(
    category: 'player' | 'world' | 'entity', 
    field: string, 
    contextData: Record<string, unknown>, 
    modelName: string = 'gemini-3-pro-preview',
    currentInput?: string, // New Parameter for Enrich Mode
    settings?: AppSettings
  ): Promise<string> {
    try {
      // 1. Get System Instruction based on Mode (Create vs Enrich)
      const systemInstruction = getWorldCreationSystemInstruction(category, field, currentInput);

      // 2. Build User Prompt
      // Note: buildWorldCreationPrompt now handles the switching logic inside
      let userPrompt = "";

      if (currentInput && currentInput.trim().length > 0) {
          // Enrich Mode: Prompt is handled by buildWorldCreationPrompt entirely
          userPrompt = buildWorldCreationPrompt(field, contextData, currentInput);
      } else {
          // Create Mode: Keep existing context construction logic for better randomness
          if (category === 'player') {
             userPrompt = `CHARACTER INFORMATION:
- Name: ${contextData.name}
- Gender: ${contextData.gender}
- Age: ${contextData.age}
- World Genre: ${contextData.genre || "Optional"}

REQUIREMENT: Write content for field: "${field}".`;
          } else if (category === 'world') {
             userPrompt = `WORLD INFORMATION:
- Genre: ${contextData.genre}
- World Name: ${contextData.worldName || "Untitled"}

REQUIREMENT: Write content for field: "${field}".`;
          } else if (category === 'entity') {
             userPrompt = `ENTITY INFORMATION:
- Name: ${contextData.name}
- Type: ${contextData.type} (NPC/LOCATION/CUSTOM)
- World Genre: ${contextData.genre || "Optional"}

REQUIREMENT: Write content for field: "${field}".`;
          }
      }

      // 3. Call AI
      const aiClient = getAiClient(settings);
      const response = await aiClient.models.generateContent({
        model: modelName,
        contents: userPrompt,
        config: {
          systemInstruction: systemInstruction,
          temperature: currentInput ? 0.7 : 0.85, // Lower temp for enrichment to stay closer to source
          topK: 40,
          topP: 0.95,
        }
      });

      return response.text?.trim() || "";
    } catch {
      return "Không thể kết nối với AI. Vui lòng kiểm tra API Key hoặc thử lại sau.";
    }
  },

  async generateFullWorld(concept: string, modelName: string = 'gemini-3-pro-preview', settings?: AppSettings, existingData?: Record<string, unknown>): Promise<Record<string, unknown>> {
    let existingContext = "";
    if (existingData) {
        existingContext = `
[CURRENT DATA - MUST RESPECT AND NOT CHANGE]
${JSON.stringify(existingData, null, 2)}

IMPORTANT DIRECTIVES:
1. If a field in "CURRENT DATA" already has content, you MUST keep that content in the returned result.
2. You are only allowed to fill in empty fields (empty strings, empty arrays, or default values).
3. Use existing information to create new information that is logical and consistent.
4. If the 'entities' list already has data, keep them and add new entities until the required quantity is reached (total at least 4).
5. If the 'rules' list already has data, keep them and add new rules.
        `.trim();
    }

    const prompt = `
        You are a World Builder.
        Based on the core idea: "${concept}", build a complete RPG world setup.
        
        ${existingContext}

        Output requirements:
        1. Language: Vietnamese.
        2. Return in correct JSON format according to Schema.
        3. Content must be creative, logical, and have literary depth.
        4. World Name (worldName): MUST be unique, evocative, and deeply connected to the core idea and genre. Avoid generic names like "Thế giới huyền bí" or "Đại lục X". Use poetic, symbolic, or culturally relevant naming conventions.
        5. Include:
           - 1 Main Character (Player): Has a biography, personality, and clear goals related to the core idea.
           - World Setting (World): Name, genre, and detailed background/history description.
           - 4 Entities (Entities): Include at least 1 NPC, 1 Location.
           - 3-5 World Rules (Rules): Special rules, taboos, or operating mechanisms of this world.
           - Initial Game Time (initialGameTime): Choose a starting timestamp (Year, Month, Day, Hour, Minute) reasonable for the world context.
        
        NOTE ON ENTITY STRUCTURE:
        - For NPC: Must fill in Name, Gender, Age, Personality (keywords + explanation), Biography, Appearance, Introduction.
        - For Location/Custom: Gender/age fields can be empty, but description must be detailed.
      `;

      const aiClient = getAiClient(settings);
      const response = await aiClient.models.generateContent({
        model: modelName,
        contents: prompt,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              player: {
                type: Type.OBJECT,
                properties: {
                  name: { type: Type.STRING, description: "Tên nhân vật" },
                  gender: { type: Type.STRING, description: "Giới tính" },
                  age: { type: Type.STRING, description: "Tuổi" },
                  personality: { type: Type.STRING, description: "Tính cách nổi bật" },
                  background: { type: Type.STRING, description: "Tiểu sử và xuất thân" },
                  appearance: { type: Type.STRING, description: "Mô tả ngoại hình" },
                  skills: { type: Type.STRING, description: "Kỹ năng đặc biệt" },
                  goal: { type: Type.STRING, description: "Mục tiêu chính" },
                },
                required: ['name', 'gender', 'age', 'personality', 'background', 'appearance', 'skills', 'goal']
              },
              world: {
                type: Type.OBJECT,
                properties: {
                  worldName: { type: Type.STRING, description: "Tên thế giới" },
                  genre: { type: Type.STRING, description: "Thể loại" },
                  context: { type: Type.STRING, description: "Bối cảnh lịch sử, xã hội" },
                  initialGameTime: {
                    type: Type.OBJECT,
                    description: "Thời gian khởi đầu của thế giới",
                    properties: {
                      year: { type: Type.INTEGER },
                      month: { type: Type.INTEGER },
                      day: { type: Type.INTEGER },
                      hour: { type: Type.INTEGER },
                      minute: { type: Type.INTEGER }
                    },
                    required: ['year', 'month', 'day', 'hour', 'minute']
                  }
                },
                required: ['worldName', 'genre', 'context', 'initialGameTime']
              },
              rules: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
                description: "Danh sách các quy tắc thế giới"
              },
              entities: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    id: { type: Type.STRING },
                    type: { type: Type.STRING, enum: ['NPC', 'LOCATION', 'CUSTOM'] },
                    name: { type: Type.STRING },
                    // Detailed fields for generation
                    gender: { type: Type.STRING, nullable: true },
                    age: { type: Type.STRING, nullable: true },
                    personalityKeywords: { type: Type.STRING, description: "Từ khóa tính cách (Vui vẻ, Lạnh lùng...)" },
                    personalityDetail: { type: Type.STRING, description: "Diễn giải tính cách chi tiết" },
                    appearance: { type: Type.STRING, description: "Mô tả ngoại hình" },
                    background: { type: Type.STRING, description: "Tiểu sử/Lịch sử hình thành" },
                    intro: { type: Type.STRING, description: "Lời chào hoặc mô tả mở đầu" },
                    customType: { type: Type.STRING, nullable: true },
                  },
                  required: ['type', 'name', 'background', 'appearance']
                }
              }
            },
            required: ['player', 'world', 'entities']
          },
          temperature: 0.9
        }
      });

      if (response.text) {
        const data = JSON.parse(response.text);
        // Extract and map GameTime
        if (data.world && data.world.initialGameTime) {
            data.gameTime = data.world.initialGameTime;
            delete data.world.initialGameTime;
        }

        // Post-processing entities to match App Interface
        if (data.entities && Array.isArray(data.entities)) {
            data.entities = data.entities.map((ent: Record<string, unknown>, idx: number) => {
                // Merge details into the main 'description' field for the App
                let fullDesc: string;
                
                const entData = ent as Record<string, unknown>;
                const type = entData.type as string;
                const gender = entData.gender as string;
                const age = entData.age as string;
                const appearance = entData.appearance as string;
                const background = entData.background as string;
                const intro = entData.intro as string;
                const personalityKeywords = entData.personalityKeywords as string;
                const personalityDetail = entData.personalityDetail as string;
                const id = entData.id as string;
                const name = entData.name as string;
                const customType = entData.customType as string;

                if (type === 'NPC') {
                    fullDesc = `[Giới tính: ${gender || '?'}] [Tuổi: ${age || '?'}]\n`;
                    fullDesc += `\n>> NGOẠI HÌNH:\n${appearance}\n`;
                    fullDesc += `\n>> TIỂU SỬ:\n${background}\n`;
                    fullDesc += `\n>> GIỚI THIỆU:\n"${intro || '...'}"`;
                } else {
                    fullDesc = `${background}\n\n(Mô tả: ${appearance})`;
                }

                // Format Personality
                const fullPersonality = personalityKeywords 
                    ? `${personalityKeywords} - ${personalityDetail || ''}` 
                    : personalityDetail || "";

                return {
                    id: id || `ai-ent-${Date.now()}-${idx}`,
                    type: type,
                    name: name,
                    description: fullDesc, // App uses this
                    personality: fullPersonality, // App uses this for NPC
                    customType: customType
                } as Entity;
            });
        }
        return data;
      }
      throw new Error("AI trả về phản hồi rỗng.");
  },

  async generateInitialTime(genre: string, context: string, modelName: string = 'gemini-3-pro-preview', settings?: AppSettings): Promise<Record<string, unknown>> {
    const prompt = `Based on world genre: "${genre}" and context: "${context}", choose a reasonable starting timestamp (Year, Month, Day, Hour, Minute). 
    Return in correct JSON format: {"year": number, "month": number, "day": number, "hour": number, "minute": number}.
    Example: Modern is 2026, Xianxia could be 1 or 9999, etc.`;

    const aiClient = getAiClient(settings);
    const response = await aiClient.models.generateContent({
      model: modelName,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            year: { type: Type.INTEGER },
            month: { type: Type.INTEGER },
            day: { type: Type.INTEGER },
            hour: { type: Type.INTEGER },
            minute: { type: Type.INTEGER }
          },
          required: ['year', 'month', 'day', 'hour', 'minute']
        }
      }
    });

    if (response.text) {
      return JSON.parse(response.text);
    }
    throw new Error("AI không thể tạo thời gian.");
  }
};
