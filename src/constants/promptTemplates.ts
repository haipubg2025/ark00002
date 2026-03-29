
import { DifficultyLevel, OutputLength, SafetySetting } from "../types";

export const DEFAULT_SAFETY_SETTINGS: SafetySetting[] = [
  { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_NONE' },
  { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_NONE' },
  { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_NONE' },
  { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_NONE' },
  { category: 'HARM_CATEGORY_CIVIC_INTEGRITY', threshold: 'BLOCK_NONE' },
];

export const DIFFICULTY_LEVELS: DifficultyLevel[] = [
  {
    id: 'easy',
    label: 'Dễ (Easy)',
    prompt: "Game Difficulty: Easy — The story progresses such that '<user>' is always in a warm world full of happiness and love."
  },
  {
    id: 'normal',
    label: 'Bình thường (Normal)',
    prompt: "Difficulty: Normal — The story progresses with a mix of fortune and misfortune, hope in despair, and challenges in hope."
  },
  {
    id: 'hard',
    label: 'Khó (Hard)',
    prompt: `Game Difficulty: Real World — Characters have "independent personalities" & events follow "realistic logic".
    Difficulty Verification:
    - Verify the rationality of '<input>':
    - Based on '<<user>>' and '<World & Character Settings>', judge if this is reasonable.
    - Based on game difficulty, see if this is reasonable. If not, correct the progression of '<input>' during the story drive.`
  },
  {
    id: 'realistic',
    label: 'Hiện thực (Realistic)',
    prompt: `Game Difficulty: Hard — The story progresses with '<user>' facing massive failures and challenges that strike like moving mountains and seas.
    Difficulty Verification:
    - Verify the rationality of '<input>':
    - Based on '<<user>>' and '<World & Character Settings>', judge if this is reasonable.
    - Based on game difficulty, see if this is reasonable. If not, correct the progression of '<input>' during the story drive.`
  },
  {
    id: 'torment',
    label: 'Hành hạ (Torment)',
    prompt: `Game Difficulty: Torment — '<user>' is exploited by petty people; ideals are ruthlessly trampled by reality; good intentions are betrayed by coldness and misunderstanding; hope instantly turns into a joke of fate; forever stuck between failure and suffering.
    Difficulty Verification:
    - Verify the rationality of '<input>':
    - Based on '<<user>>' and '<World & Character Settings>', judge if this is reasonable.
    - Based on game difficulty, see if this is reasonable. If not, correct the progression of '<input>' during the story drive.`
  }
];

export const OUTPUT_LENGTHS: OutputLength[] = [
  { id: 'short', label: 'Ngắn (1000 - 2000 từ)', minWords: 300, maxWords: 600 },
  { id: 'medium', label: 'Trung bình (2500 - 5000 từ)', minWords: 600, maxWords: 1200 },
  { id: 'default', label: 'Mặc định (5500 - 8000 từ)', minWords: 1200, maxWords: 2500 },
  { id: 'long', label: 'Dài (8500 - 15000 từ)', minWords: 2500, maxWords: 5000 },
  { id: 'supreme', label: 'Tối thượng (5000 - 15000 từ)', minWords: 5000, maxWords: 15000 },
  { id: 'custom', label: 'Tùy chỉnh', minWords: 0 }, 
];

export const generateWordCountPrompt = (min: number, max: number) => `
<word_count_protocol>
TARGET: ${min} - ${max} words. (STRICTLY ADHERE TO MAXIMUM LIMIT)

You are a professional AI. Before writing the main story, you MUST open the <word_count> tag to plan the length.

Mandatory structure in the <word_count> tag:
1. [Target] Set a specific word count target within the range ${min}-${max}. Absolutely do not exceed ${max} words.
2. [Segmentation] Break the story into 3-4 segments (Checkpoints), estimating the word count for each.
3. [Pacing] Determine the pacing (Fast/Slow/Intense) to achieve that word count without using filler text.

Rules:
- Only perform arithmetic calculations and layout here.
- DO NOT write story content or character psychological speculation here.
- If you write too long (> ${max} words), the response will be cut off and considered a failure.
- If there is no <word_count> tag at the beginning of the response, the system will consider it a serious error.
</word_count_protocol>
`;
