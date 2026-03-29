
import { LSR_DATA } from "../../data/lsr_config";

export interface LsrTableDefinition {
    id: string;
    name: string;
    columns: string[];
}

export const LsrParser = {
    /**
     * Parses the static structure of LSR tables from the configuration.
     * Scans all entries for patterns like "#0 TableName|0:Col1|1:Col2".
     */
    parseDefinitions(): LsrTableDefinition[] {
        const tables: LsrTableDefinition[] = [];
        const tableDefRegex = /^#(\d+)\s+([^|]+)\|(.*)$/;

        // Scan all entries in LSR_DATA
        Object.values(LSR_DATA.entries).forEach((entry: unknown) => {
            if (!entry || typeof entry !== 'object' || !('content' in entry)) return;
            const content = (entry as { content: string }).content;

            const lines = content.split('\n');
            lines.forEach((line: string) => {
                const trimmed = line.trim();
                const match = trimmed.match(tableDefRegex);
                if (match) {
                    const id = match[1];
                    // Check if table ID already exists to avoid duplicates
                    if (tables.some(t => t.id === id)) return;

                    const name = match[2].trim();
                    const rawColumns = match[3];

                    const columns = rawColumns.split('|').map(col => {
                        const colParts = col.split(':');
                        return colParts.length > 1 ? colParts.slice(1).join(':').trim() : col.trim();
                    });

                    tables.push({ id, name, columns });
                }
            });
        });

        if (tables.length === 0) {
            console.warn("LsrParser: Failed to parse any table definitions from LSR_DATA.");
        } else {
            // Sort tables by ID
            tables.sort((a, b) => parseInt(a.id) - parseInt(b.id));
            // console.log(`LsrParser: Successfully parsed ${tables.length} table definitions.`);
        }

        return tables;
    },

    /**
     * Parses the runtime output from AI (Text-based LSR format).
     * Format:
     * <table_stored>
     * #0 Thông tin Hiện tại|0:Năm Thương Lan 3025|1:Hang đá
     * #1 Nhân vật Gần đây|0:Lộ Na|1:0|2:Ăn uống
     * </table_stored>
     * 
     * Returns a map of TableID -> Array of Rows (objects)
     */
    parseLsrString(rawString: string): Record<string, Record<string, string>[]> {
        const result: Record<string, Record<string, string>[]> = {};
        
        if (!rawString) return result;

        // Clean up common AI artifacts that might be inside the tag
        const cleanString = rawString
            .replace(/```lsr/g, '')
            .replace(/```/g, '')
            .trim();

        // --- PHƯƠNG ÁN 1: ĐỊNH DẠNG LSR CHUẨN (#ID Name|0:Val...) ---
        // Sử dụng regex để tìm tất cả các khối bắt đầu bằng #ID
        // Pattern: #(\d+) theo sau là bất kỳ thứ gì cho đến khi gặp # tiếp theo hoặc hết chuỗi
        const tableBlocks = cleanString.split(/(?=#\d+)/);
        
        let hasAnyData = false;

        tableBlocks.forEach(block => {
            const trimmedBlock = block.trim();
            if (!trimmedBlock.startsWith('#')) return;

            // Tìm Table ID: #(\d+)
            const idMatch = trimmedBlock.match(/^#(\d+)/);
            if (!idMatch) return;

            const tableId = idMatch[1];
            
            // Tìm tất cả các cặp Index:Value trong block này
            // Pattern: (\d+)\s*:\s*([^|\n]+)
            // Chúng ta tìm các cặp số:giá trị, dừng lại khi gặp | hoặc xuống dòng
            const rowObj: Record<string, string> = {};
            
            // Regex tìm các cặp 0:Giá trị, 1:Giá trị...
            // Hỗ trợ cả trường hợp có khoảng trắng: "0 : Giá trị"
            const colRegex = /(\d+)\s*:\s*([^|\n]+)/g;
            let colMatch;
            
            while ((colMatch = colRegex.exec(trimmedBlock)) !== null) {
                const colIdx = colMatch[1];
                const colVal = colMatch[2].trim();
                rowObj[colIdx] = colVal;
            }

            if (Object.keys(rowObj).length > 0) {
                if (!result[tableId]) result[tableId] = [];
                
                // Kiểm tra xem dòng này đã tồn tại chưa (dựa trên cột 0 - ID/Tên)
                // Trong parseLsrString, ta thường giả định mỗi block #ID là một dòng
                // Nếu AI output nhiều block cùng ID, ta thêm vào mảng
                result[tableId].push(rowObj);
                hasAnyData = true;
            }
        });

        // Giới hạn 10 dòng cho bảng "Thông tin Hiện tại" (ID #0)
        if (result["0"] && result["0"].length > 10) {
            result["0"] = result["0"].slice(-10);
        }

        if (hasAnyData) return result;

        // --- PHƯƠNG ÁN 2: DỰ PHÒNG BẢNG MARKDOWN (NẾU AI QUÊN ĐỊNH DẠNG) ---
        const lines = cleanString.split('\n');
        const markdownRows = lines.filter(l => l.trim().startsWith('|') && l.trim().endsWith('|'));
        if (markdownRows.length >= 3) {
            console.warn("LsrParser: Phát hiện bảng Markdown thay vì định dạng LSR chuẩn.");
        }

        return result;
    },

    /**
     * Converts the runtime data map back to the text-based LSR format for AI consumption.
     */
    stringifyLsrData(data: Record<string, Record<string, string>[]>, tables: LsrTableDefinition[]): string {
        let result = "";
        
        tables.forEach(table => {
            const rows = data[table.id] || [];
            if (rows.length === 0) return;

            rows.forEach(row => {
                // Format: #ID Name|0:Val|1:Val
                let rowStr = `#${table.id} ${table.name}|`;
                
                const colEntries = Object.entries(row)
                    .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
                    .map(([idx, val]) => `${idx}:${val}`);
                
                rowStr += colEntries.join('|');
                result += rowStr + "\n";
            });
        });

        return result.trim();
    },

    /**
     * Merges new data into existing data.
     * If a row with the same key (column 0) exists, it updates it.
     * Otherwise, it adds a new row.
     */
    mergeLsrData(existing: Record<string, Record<string, string>[]>, incoming: Record<string, Record<string, string>[]>): Record<string, Record<string, string>[]> {
        const next = { ...existing };

        Object.keys(incoming).forEach(tableId => {
            const existingRows = next[tableId] ? [...next[tableId]] : [];
            const incomingRows = incoming[tableId];

            incomingRows.forEach(newRow => {
                const keyVal = newRow["0"]; // Column 0 is usually the ID/Name
                if (keyVal === undefined) {
                    existingRows.push(newRow);
                } else {
                    const existingIndex = existingRows.findIndex(r => r["0"] === keyVal);
                    if (existingIndex !== -1) {
                        // Update existing row
                        existingRows[existingIndex] = { ...existingRows[existingIndex], ...newRow };
                    } else {
                        // Add new row
                        existingRows.push(newRow);
                    }
                }
            });

            // Giới hạn 10 dòng cho bảng "Thông tin Hiện tại" (ID #0)
            if (tableId === "0" && existingRows.length > 10) {
                next[tableId] = existingRows.slice(-10);
            } else {
                next[tableId] = existingRows;
            }
        });

        return next;
    }
};
