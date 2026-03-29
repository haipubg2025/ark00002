import { getAiClient } from "./client";
import { dbService, VectorData } from "../db/indexedDB";
import { ChatMessage, AppSettings } from "../../types";

// Task 3.2: Vector Service Implementation

export const vectorService = {
    /**
     * Calculates Cosine Similarity between two vectors
     */
    cosineSimilarity(vecA: number[], vecB: number[]): number {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    },

    /**
     * Generates embedding for a given text using free models with rotation/fallback
     */
    async getEmbedding(text: string, settings?: AppSettings): Promise<number[] | null> {
        if (!text || text.trim().length === 0) return null;

        const models = ['gemini-embedding-2-preview', 'embedding-001'];
        
        for (const modelName of models) {
            try {
                // FORCE DIRECT: Always use personal/system key for embeddings, ignore proxy
                const aiClient = getAiClient(settings, true);
                
                const result = await aiClient.models.embedContent({
                    model: modelName,
                    contents: [
                        {
                            parts: [{ text: text }]
                        }
                    ]
                });

                const embedding = result.embeddings?.[0];
                if (embedding?.values) {
                    return embedding.values;
                }
            } catch (error: unknown) {
                console.warn(`[VectorService] Failed with model ${modelName}, trying next...`, error);
                continue; // Try next model in list
            }
        }

        console.error("[VectorService] All embedding models failed.");
        return null;
    },

    /**
     * Saves a message (user or model) to Vector DB
     */
    async saveVector(id: string, text: string, role: 'user' | 'model', settings?: AppSettings): Promise<void> {
        // Avoid re-saving if exists
        const exists = await dbService.hasVector(id);
        if (exists) return;

        const embedding = await this.getEmbedding(text, settings);
        if (embedding) {
            const vectorData: VectorData = {
                id,
                text,
                embedding,
                timestamp: Date.now(),
                role
            };
            await dbService.saveVector(vectorData);
        }
    },

    /**
     * Searches for semantically similar text from the vector database
     */
    async searchSimilarVectors(queryText: string, settings?: AppSettings, limit: number = 10): Promise<VectorData[]> {
        const queryEmbedding = await this.getEmbedding(queryText, settings);
        if (!queryEmbedding) return [];

        const allVectors = await dbService.getAllVectors();
        
        // Calculate similarity for each vector
        const scoredVectors = allVectors.map(vec => ({
            ...vec,
            score: this.cosineSimilarity(queryEmbedding, vec.embedding)
        }));

        // Sort by score descending and take top 'limit'
        return scoredVectors
            .filter(v => v.score > 0.35) 
            .sort((a, b) => b.score - a.score)
            .slice(0, limit);
    },

    /**
     * Task 3.4: Process old history and vectorize missing messages
     */
    async vectorizeAllHistory(history: ChatMessage[], settings?: AppSettings): Promise<void> {
        for (let i = 0; i < history.length; i++) {
            const msg = history[i];
            const msgId = `msg-${msg.timestamp}-${msg.role}`;
            
            const exists = await dbService.hasVector(msgId);
            if (!exists && msg.text) {
                await new Promise(r => setTimeout(r, 200)); 
                await this.saveVector(msgId, msg.text, msg.role, settings);
            }
        }
    }
};
