
import { Lorebook, LorebookEntry } from "./types";

// Default values for common LSR variables to avoid raw macro leaks
const LSR_DEFAULTS: Record<string, string> = {
    'tableConfigDateFormat': 'YYYY-MM-DD',
    'tableConfigTimeFormat': 'hh:mm',
    'tableConfigExtraTimeFormat': 'Day X',
    'tableConfigRecentLimit': '5',
    'tableConfigCharacterMember': 'Character',
    'tableConfigContentBegin': '<content>',
    'tableConfigContentEnd': '</content>',
    'tableConfigCoTBegin': '<thinking>',
    'tableConfigCoTEnd': '</thinking>',
    'tableConfigUserInput': '<user_input>',
    'tableConfigTagsBeforeTableEdit': '',
    'tableConfigTagsAfterTableEdit': '',
    // Dummy values for logic flags
    'tableConfigSexWrite': '1',
    'tableConfigScheduleWrite': '1',
    'tableConfigAbilityWrite': '1',
    'tableConfigOrganizationWrite': '1',
    'tableConfigLocationWrite': '1',
    'tableConfigHistoryRows': '3',
    'tableConfigHistoryLength': '100 tokens',
    'tableConfigSummaryRows': '5',
    'tableConfigSummaryLength': '200 tokens',
    'tableConfigPresumeMode': '1',
    'tableConfigCharacterReferenceName': 'none',
};

export class LorebookService {
    /**
     * Converts raw JSON structure to Array
     */
    static loadLorebook(jsonData: Lorebook): LorebookEntry[] {
        if (!jsonData || !jsonData.entries) return [];
        return Object.values(jsonData.entries);
    }

    /**
     * Replaces {{getvar::key}} macros with values or defaults.
     * Also strips EJS logic blocks <%_ ... _%> as we cannot execute them safely.
     */
    static processMacros(text: string, dynamicVars: Record<string, string> = {}): string {
        let processed = text;

        // 1. Remove EJS Script Blocks (<%_ ... _%>)
        // We assume the user wants the prompt instructions, not the JS logic code
        processed = processed.replace(/<%_[\s\S]*?_%>/g, '');

        // 2. Replace {{getvar::KEY}} and {{KEY}}
        processed = processed.replace(/\{\{(?:getvar::)?(.*?)\}\}/g, (match, key) => {
            const cleanKey = key.trim();
            // Check dynamic vars first (passed from game state), then defaults
            if (dynamicVars[cleanKey] !== undefined) return dynamicVars[cleanKey];
            if (LSR_DEFAULTS[cleanKey] !== undefined) return LSR_DEFAULTS[cleanKey];
            
            // If variable not found, return empty string to hide the macro syntax
            return '';
        });

        // 3. Simple cleanup of empty lines resulting from stripped scripts
        return processed.replace(/^\s*[\r\n]/gm, '');
    }

    /**
     * Scans inputs and returns the combined text of activated entries
     */
    static scanAndActivate(
        textToScan: string, 
        entries: LorebookEntry[], 
        dynamicVars: Record<string, string> = {}
    ): string {
        const uniqueEntries = new Set<LorebookEntry>();
        const lowerInput = textToScan.toLowerCase();

        entries.forEach(entry => {
            if (entry.disable) return;

            let activated = false;

            // 1. Constant Strategy
            if (entry.constant) {
                activated = true;
            }

            // 2. Keyword Strategy
            if (!activated && entry.key && entry.key.length > 0) {
                // Check if ANY key is present in input
                const hasMatch = entry.key.some(k => {
                    const cleanKey = k.trim().toLowerCase();
                    return cleanKey && lowerInput.includes(cleanKey);
                });
                if (hasMatch) activated = true;
            }

            if (activated) {
                uniqueEntries.add(entry);
            }
        });

        // Convert to array and sort
        const activeList = Array.from(uniqueEntries);

        // Sorting Logic: 
        // In ST, usually higher order means inserted later (closer to bottom).
        // If position is same, use order.
        activeList.sort((a, b) => a.order - b.order);

        // Process content (Macros cleaning) and join
        const combinedText = activeList
            .map(e => this.processMacros(e.content, dynamicVars))
            .filter(text => text.trim().length > 0) // Remove empty results
            .join('\n\n');

        return combinedText;
    }
}
