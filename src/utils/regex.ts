/**
 * Các Regex hệ thống cho Tawa Protocol
 */
export const TAWA_REGEX = {
  // Các mẫu cần loại bỏ khỏi văn bản hiển thị
  ARTIFACTS_REMOVAL: [
    /\[Hệ thống:.*?\]/gi,
    /\[Thông báo:.*?\]/gi,
    /\[Status:.*?\]/gi,
    /\[Cảnh báo:.*?\]/gi,
    /\[Lưu ý:.*?\]/gi,
    /\[Gợi ý:.*?\]/gi,
    /\[Time:.*?\]/gi,
    /\[Turn:.*?\]/gi,
    /\[Mục tiêu:.*?\]/gi,
    /\[Tiến độ:.*?\]/gi,
    /\[Giai đoạn:.*?\]/gi,
    /\[Đoạn:.*?\]/gi,
    /\[Số chữ:.*?\]/gi,
    /\[Tình trạng:.*?\]/gi,
    /\[Kiểm tra:.*?\]/gi,
    /\[Phân tích:.*?\]/gi,
    /\[Thinking:.*?\]/gi,
    /\[Content:.*?\]/gi,
    /\[Branches:.*?\]/gi,
    /\[Choices:.*?\]/gi,
    /\[Actions:.*?\]/gi,
    /\[LSR_SYNC:.*?\]/gi,
    /\[DATA_SYNC:.*?\]/gi,
    /\[INPUT_DECODING:.*?\]/gi,
    /\[SYNCHRONIZATION:.*?\]/gi,
    /\[Loading.*?:.*?\]/gi,
    /\[Tải.*?:.*?\]/gi,
    /\[KẾT THÚC PHẦN TRUYỆN\]/gi,
    /\[BẮT ĐẦU PHẦN TRUYỆN\]/gi,
    /^\s*->\s*$/gm,
    /^\s*=>\s*$/gm
  ]
};

/**
 * Trích xuất nội dung nằm giữa tag mở và tag đóng
 * Hỗ trợ cả trường hợp tag chưa đóng (cho streaming) hoặc có khoảng trắng trong tag
 * @param text Văn bản gốc
 * @param tagName Tên thẻ (không bao gồm < >)
 * @returns Nội dung bên trong hoặc null
 */
export const extractTagContent = (text: string, tagName: string): string | null => {
  if (typeof text !== 'string' || !text) return null;
  
  // Regex hỗ trợ khoảng trắng trong tag mở: <tag >
  const openTagRegex = new RegExp(`<${tagName}\\s*>`, 'i');
  const closeTagRegex = new RegExp(`</${tagName}\\s*>`, 'i');
  
  const openMatch = text.match(openTagRegex);
  if (!openMatch) return null;
  
  const startIndex = openMatch.index! + openMatch[0].length;
  const closeMatch = text.match(closeTagRegex);
  
  if (closeMatch) {
    return text.substring(startIndex, closeMatch.index).trim();
  }
  
  // Nếu không tìm thấy tag đóng, lấy hết phần còn lại (hữu ích cho streaming)
  return text.substring(startIndex).trim();
};

/**
 * Phân tách nội dung tag branches thành danh sách các lựa chọn
 * @param content Nội dung thô bên trong tag branches
 * @returns Danh sách các chuỗi lựa chọn
 */
export const parseChoices = (content: string | null): string[] => {
  if (!content) return [];
  
  // Tách theo dòng. Không tách theo | hoặc ; vì chúng có thể xuất hiện trong dữ liệu LSR rò rỉ
  const rawLines = content.split('\n');
  const validChoices: string[] = [];
  const fallbackChoices: string[] = [];
    
  for (const line of rawLines) {
    // 1. Làm sạch dòng: loại bỏ số thứ tự (1., 2., - , * , v.v.) ở đầu câu
    // Điều này giúp các regex sau đó (như narrativePatterns) hoạt động chính xác hơn.
    const cleaned = line.trim().replace(/^(\d+[.)-\s]+|[-*\u2022]\s*)/, '').trim();
    if (!cleaned) continue;

    // 0. BẮT BUỘC: Phải có định dạng chi phí thời gian [phút] ở đầu hoặc trong câu
    // Theo yêu cầu người dùng: "chỉ những hành động có chi phí thời gian mới được hiển thị"
    // Cải tiến: Hỗ trợ [10], [10 phút], [10 min], v.v.
    // Tránh khớp với các phân số như [580/1800] bằng cách giới hạn ký tự bên trong ngoặc
    const hasTime = cleaned.match(/\[\d+\s*(m|phút|min)?\]/i);

    // CẢI TIẾN QUAN TRỌNG: Nếu một dòng quá dài, nó có khả năng là văn bản truyện bị rò rỉ
    // Một lựa chọn hành động thường không quá 300 ký tự.
    if (cleaned.length > 350) continue;

    // KIỂM TRA CẤU TRÚC VĂN KỂ CHUYỆN & ĐỐI THOẠI:
    // Nới lỏng: Không lọc các câu kết thúc bằng dấu chấm trừ khi quá dài (văn kể chuyện thường rất dài)
    if (cleaned.endsWith('.') && cleaned.length > 200) continue;

    // 2. Nếu dòng chứa dấu ngoặc kép và có dấu hai chấm (định dạng đối thoại: Tên: "..."), loại bỏ.
    // Nhưng cho phép đối thoại ngắn trong lựa chọn (ví dụ: [5] Nói "Chào")
    if (cleaned.includes('"') && cleaned.includes(':') && cleaned.indexOf(':') < cleaned.indexOf('"')) continue;

    // 3. Nếu dòng bắt đầu bằng một cái tên và dấu hai chấm (ví dụ: Lâm Phong: ), loại bỏ.
    if (cleaned.match(/^[A-ZÀ-Ỹ][a-zà-ỹ]*(\s+[A-ZÀ-Ỹ][a-zà-ỹ]*)*\s*:/)) continue;

    // Narrative often starts with a name followed by a verb (Vietnamese specific patterns)
    // e.g., "Elara khẽ thở dài", "Hắn nhìn quanh", "Cô ấy mỉm cười", "Dân làng phẫn nộ"
    const narrativePatterns = [
      /^[A-ZÀ-Ỹ][a-zà-ỹ]*\s+(khẽ|đang|vừa|đã|sẽ|mỉm|nhìn|bước|thở|nói|hỏi|đáp|nghĩ|cảm|thấy|đứng|ngồi|đi|chạy|cười|khóc|mất|ra|lấy|cầm|đưa|nhảy|la|hét|gầm|nhếch|quay|tiến|lùi|đấm|đá|tấn|phòng|vung|chém|bắn|tung|rơi|bay|nằm|gục|chết|sống|biến|hiện|tỏa|thu|phát|truyền|hút|đẩy|kéo|nâng|hạ|mở|đóng|xé|nát|vỡ|tan|hợp|chia|đổi|giữ|buông|thả|bắt|thả|giết|cứu|giúp|hại|yêu|ghét|giận|vui|buồn|lo|sợ|kinh|ngạc|ngỡ|ngàng|bàng|hoàng|sửng|sốt|phẫn|nộ|hậm|hực|cay|cú|uất|ức|đau|khổ|sung|sướng|hạnh|phúc|tự|hào|thất|vọng|chán|nản|mệt|mỏi|yếu|ớt|mạnh|mẽ|hung|hăng|hung|dữ|hiền|lành|ngoan|ngoãn|hư|hỏng|tốt|bụng|xấu|xa|tham|lam|nhát|gan|dũng|cảm|thông|minh|ngu|ngốc|khờ|khạo|tinh|ranh|xảo|quyệt|gian|xảo|thật|thà|hiền|hậu|nhân|từ|độc|ác|tàn|nhẫn|vô|tình|hữu|ý|vô|ý|cố|tình|vô|tình)/,
      /^(Hắn|Cô|Anh|Chị|Nó|Họ|Tôi|Ta|Chúng|Bọn|Đám|Dân|Lính|Người|Kẻ|Tên|Gã|Mụ|Lão|Bà|Ông|Con|Cái|Sự|Cái|Điều)\s+(khẽ|đang|vừa|đã|sẽ|mỉm|nhìn|bước|thở|nói|hỏi|đáp|nghĩ|cảm|thấy|đứng|ngồi|đi|chạy|cười|khóc|mất|ra|lấy|cầm|đưa|nhảy|la|hét|gầm|nhếch|quay|tiến|lùi|đấm|đá|tấn|phòng|vung|chém|bắn|tung|rơi|bay|nằm|gục|chết|sống|biến|hiện|tỏa|thu|phát|truyền|hút|đẩy|kéo|nâng|hạ|mở|đóng|xé|nát|vỡ|tan|hợp|chia|đổi|giữ|buông|thả|bắt|thả|giết|cứu|giúp|hại|yêu|ghét|giận|vui|buồn|lo|sợ|kinh|ngạc|ngỡ|ngàng|bàng|hoàng|sửng|sốt|phẫn|nộ|hậm|hực|cay|cú|uất|ức|đau|khổ|sung|sướng|hạnh|phúc|tự|hào|thất|vọng|chán|nản|mệt|mỏi|yếu|ớt|mạnh|mẽ|hung|hăng|hung|dữ|hiền|lành|ngoan|ngoãn|hư|hỏng|tốt|bụng|xấu|xa|tham|lam|nhát|gan|dũng|cảm|thông|minh|ngu|ngốc|khờ|khạo|tinh|ranh|xảo|quyệt|gian|xảo|thật|thà|hiền|hậu|nhân|từ|độc|ác|tàn|nhẫn|vô|tình|hữu|ý|vô|ý|cố|tình|vô|tình)/,
      /^[A-ZÀ-Ỹ][a-zà-ỹ]*\s+[a-zà-ỹ]*\s+(khẽ|đang|vừa|đã|sẽ|mỉm|nhìn|bước|thở|nói|hỏi|đáp|nghĩ|cảm|thấy|đứng|ngồi|đi|chạy|cười|khóc)/,
      /^[A-ZÀ-Ỹ][a-zà-ỹ]*\s+có\s+chút\s+[a-zà-ỹ]*/,
      /^(Dân làng|Bọn lính|Đám đông|Mọi người|Kẻ địch|Đối phương|Nhân vật|Hệ thống)\s+/
    ];
    if (narrativePatterns.some(p => p.test(cleaned))) continue;

    // 4. Nếu dòng bắt đầu bằng # hoặc có định dạng index:Value (thông tin hệ thống rò rỉ), loại bỏ.
    if (cleaned.startsWith('#')) continue;
    // Regex mạnh hơn để bắt format d: hoặc d:d (LSR data)
    if (cleaned.match(/^\d+\s*:/)) continue;
    if (cleaned.match(/^[a-zA-Z0-9_]+\s*:/) && !cleaned.includes(' ')) continue; // Tên biến hệ thống: giá trị

    // 5. Nếu dòng quá ngắn (dưới 3 ký tự) và không phải là số trong ngoặc, loại bỏ.
    if (cleaned.length < 3 && !hasTime) continue;
    if (/^[^\w\s]+$/.test(cleaned)) continue;

    // 6. Nới lỏng: Cho phép dấu ngoặc kép trong lựa chọn hành động
    // if (cleaned.includes('"') || cleaned.includes('“') || cleaned.includes('”')) continue;

    // LOẠI BỎ CÁC DÒNG TƯ DUY / LOG HỆ THỐNG (Dựa trên ảnh lỗi của người dùng)
    const lower = cleaned.toLowerCase();
    if (lower === 'is clean.' || 
        lower === 'is clean' ||
        lower.includes('đoạn:') || 
        lower.includes('số chữ:') || 
        lower.includes('giai đoạn:') || 
        lower.includes('mục tiêu:') || 
        lower.includes('tiến độ:') || 
        lower.includes('đạt chuẩn') || 
        lower.includes('đang viết') ||
        lower.includes('kiểm tra') ||
        lower.includes('phân tích') ||
        lower.includes('thinking') ||
        lower.includes('content') ||
        lower.includes('branches') ||
        lower.includes('choices') ||
        lower.includes('actions') ||
        lower.includes('bắt đầu phần truyện') ||
        lower.includes('kết thúc phần truyện') ||
        cleaned.match(/\[(BẮT ĐẦU|KẾT THÚC)\s+PHẦN\s+TRUYỆN\]/i) ||
        cleaned.match(/^\[.*?\]$/) && (lower.includes('truyện') || lower.includes('phần')) ||
        lower.includes('lsr_sync') ||
        lower.includes('data_sync') ||
        lower.includes('input_decoding') ||
        lower.includes('synchronization') ||
        lower.includes('loading') ||
        lower.includes('tải biến') ||
        lower.includes('tải hiến pháp') ||
        lower.includes('luật cấm') ||
        lower.includes('đồng bộ') ||
        lower.includes('văn bản chính') ||
        lower.includes('chính văn') ||
        lower.includes('kết quả:') ||
        lower.includes('trạng thái:') ||
        lower.includes('thời gian:') ||
        lower.includes('địa điểm:') ||
        lower.includes('nhân vật:') ||
        lower.includes('hành động:') ||
        lower.includes('bối cảnh:') ||
        lower.includes('director\'s notes') ||
        lower.includes('director notes') ||
        lower.includes('system boot') ||
        lower.includes('resource loading') ||
        lower.includes('- ensure') ||
        lower.includes('- style') ||
        lower.includes('- anti-leakage') ||
        lower.includes('- structure') ||
        lower.includes('- no dialogue') ||
        lower.includes('- no narrative') ||
        lower.includes('- format') ||
        lower.includes('- purity') ||
        lower.includes('style enforcement') ||
        lower.includes('anti-leakage') ||
        lower.includes('structure check') ||
        lower.includes('player-centric') ||
        lower.includes('choice logic') ||
        lower.includes('action format') ||
        lower.includes('detail & progression') ||
        lower.includes('strict separation') ||
        lower.includes('important warning') ||
        lower.includes('branching rules') ||
        lower.includes('output structure') ||
        lower.includes('integrity check') ||
        lower.includes('word count') ||
        lower.includes('paragraphs:') ||
        lower.includes('plan for next stage') ||
        lower.includes('words per paragraph') ||
        lower.includes('internal dialogue') ||
        lower.includes('decision-making process') ||
        lower.includes('inprogress') ||
        cleaned.match(/^\w+\s+\d+\s+word\s+count/i) ||
        cleaned.match(/^\w+\s+\d+\s+paragraphs/i) ||
        cleaned.match(/^\w+\s+\d+\s+plan\s+for/i) ||
        cleaned.match(/\[\d+\/\d+\]/) || // Loại bỏ các fraction như [580/1800]
        cleaned.match(/^\d{4}-\d{2}-\d{2}/) || // Loại bỏ dòng ngày tháng rò rỉ
        cleaned.match(/^\d{2}:\d{2}/) // Loại bỏ dòng giờ giấc rò rỉ
    ) continue;

    // Loại bỏ các dòng rác hoặc tag rò rỉ
    if (cleaned.includes('|') && (cleaned.includes(':') || cleaned.startsWith('#'))) continue; // Lọc bỏ dữ liệu LSR dạng bảng
    if (cleaned.includes('<') || cleaned.includes('>')) continue;

    // Loại bỏ các dòng nằm trong ngoặc đơn (thường là lời nhắn của AI)
    if (cleaned.startsWith('(') && cleaned.endsWith(')')) continue;

    // Loại bỏ các dòng chỉ có dấu chấm hoặc ký tự đặc biệt rác
    if (cleaned === '.' || cleaned === '...' || cleaned === '-' || cleaned === '->' || cleaned === '=>' || cleaned === '---') continue;

    // Loại bỏ các dòng bắt đầu bằng dấu gạch ngang hoặc mũi tên rác (thường là list chỉ dẫn rò rỉ)
    if (cleaned.startsWith('- ') || cleaned.startsWith('->') || cleaned.startsWith('=>') || cleaned.startsWith('---')) continue;
    
    // Loại bỏ các dòng có định dạng tiêu đề [THIẾT LẬP...] hoặc dấu mốc cấu trúc
    if (cleaned.match(/^\[(THIẾT LẬP|HỆ THỐNG|THÔNG BÁO|STATUS|START WRITING|WRITING START|STORY START|CONTINUE|NEXT|BẮT ĐẦU PHẦN TRUYỆN|KẾT THÚC PHẦN TRUYỆN).*?\]$/i)) continue;

    // 7. Thêm vào danh sách lựa chọn nếu vượt qua tất cả các bộ lọc
    if (cleaned) {
      if (hasTime) {
        validChoices.push(cleaned);
      } else {
        fallbackChoices.push(cleaned);
      }
    }
  }
  
  // Nếu có lựa chọn có thời gian, ưu tiên chúng
  if (validChoices.length > 0) return validChoices;
  
  // Nếu không có lựa chọn có thời gian, nhưng có lựa chọn tiềm năng, gắn thời gian mặc định [5]
  if (fallbackChoices.length > 0) {
    return fallbackChoices.map(c => `[5] ${c}`).slice(0, 10);
  }

  return [];
};

/**
 * Loại bỏ các thẻ hệ thống để lấy nội dung truyện thuần túy hiển thị ra UI
 * @param text Văn bản gốc
 * @returns Văn bản đã làm sạch
 */
export const cleanRawText = (text: string): string => {
  if (typeof text !== 'string' || !text) return "";
  
  let cleaned = text;

  // Danh sách các thẻ hệ thống cần loại bỏ hoàn toàn (bao gồm cả nội dung bên trong)
  const tagsToRemove = [
    'thinking',
    'table_stored',
    'tableEdit',
    'branches',
    'choices',
    'actions',
    'memory_table_guide',
    'user_input',
    'word_count',
    'details',
    'set_time',
    'time_cost',
    'incrementalSummary'
  ];

  // 1. Xóa các khối thẻ hệ thống (Thẻ + Nội dung bên trong)
  // Chúng ta xóa các khối này TRƯỚC để tránh nội dung bên trong chúng bị hiển thị
  // CẢI TIẾN: Không xóa nội dung nếu nội dung bên trong quá dài (> 1200 ký tự) 
  // hoặc nếu thẻ chưa đóng VÀ nội dung bên trong dài (> 600 ký tự).
  tagsToRemove.forEach(tag => {
    const openTagStr = `<${tag}\\s*>`;
    const closeTagStr = `</${tag}\\s*>`;
    
    // Sử dụng vòng lặp để xử lý tất cả các cặp thẻ của cùng một loại
    let searchIndex = 0;
    while (true) {
        const openTagRegex = new RegExp(openTagStr, 'i');
        const remainingText = cleaned.substring(searchIndex);
        const openMatch = remainingText.match(openTagRegex);
        
        if (!openMatch) break;
        
        const startIndex = searchIndex + openMatch.index!;
        const afterOpen = cleaned.substring(startIndex + openMatch[0].length);
        const closeTagRegex = new RegExp(closeTagStr, 'i');
        const closeMatch = afterOpen.match(closeTagRegex);
        
        if (closeMatch) {
            // Thẻ đã đóng
            const contentInside = afterOpen.substring(0, closeMatch.index!);
            const endIndex = startIndex + openMatch[0].length + closeMatch.index! + closeMatch[0].length;
            
            // Nếu nội dung bên trong quá dài, có khả năng AI đã viết truyện nhầm vào trong tag
            // Chúng ta giữ lại nội dung đó nhưng xóa thẻ
            if (contentInside.length > 1200) {
                cleaned = cleaned.substring(0, startIndex) + "\n" + contentInside + "\n" + cleaned.substring(endIndex);
                searchIndex = startIndex + contentInside.length + 2;
            } else {
                // Nội dung ngắn, xóa toàn bộ khối
                cleaned = cleaned.substring(0, startIndex) + cleaned.substring(endIndex);
                searchIndex = startIndex;
            }
        } else {
            // Thẻ chưa đóng (đang streaming)
            const contentAfter = afterOpen;
            if (contentAfter.length < 600) {
                // Nội dung ngắn, xóa từ thẻ mở đến hết để tránh rò rỉ tag đang viết dở
                cleaned = cleaned.substring(0, startIndex);
                break; // Đã xóa đến hết chuỗi
            } else {
                // Nội dung dài, có khả năng AI đang viết truyện nhầm vào trong tag
                // Xóa thẻ mở nhưng giữ lại nội dung
                cleaned = cleaned.substring(0, startIndex) + "\n" + contentAfter;
                searchIndex = startIndex + 1; 
            }
        }
        
        // Tránh vòng lặp vô tận nếu không có sự thay đổi (phòng hờ)
        if (searchIndex >= cleaned.length) break;
    }
  });

  // 2. Xử lý thẻ <content> hoặc <story>
  // Nếu có thẻ này, chúng ta ưu tiên lấy nội dung bên trong nó.
  // NHƯNG nếu không có, chúng ta vẫn giữ lại văn bản gốc (đã xóa các thẻ hệ thống ở bước 1)
  const contentBody = extractTagContent(cleaned, 'content');
  const storyBody = extractTagContent(cleaned, 'story');
  
  if (contentBody) {
      cleaned = contentBody;
  } else if (storyBody) {
      cleaned = storyBody;
  }

  // 3. XÓA BỎ CÁC THẺ CÒN SÓT LẠI (Chỉ xóa thẻ, giữ lại văn bản bên ngoài/giữa các thẻ)
  const allTagsToStrip = [...tagsToRemove, 'content', 'story', 'incrementalSummary'];
  allTagsToStrip.forEach(tag => {
      const openRegex = new RegExp(`<${tag}\\s*>`, 'gi');
      const closeRegex = new RegExp(`</${tag}\\s*>`, 'gi');
      cleaned = cleaned.replace(openRegex, '').replace(closeRegex, '');
  });

  const originalBeforeFiltering = cleaned;

  // 4. LỌC CÁC DÒNG CHỈ DẪN HỆ THỐNG DẠNG VĂN BẢN THUẦN
  const lines = cleaned.split('\n');
  const filteredLines = lines.filter(line => {
    const trimmed = line.trim();
    if (!trimmed) return true; // Giữ lại dòng trống để ngắt đoạn

    // Loại bỏ các artifact kỹ thuật
    if (trimmed.toLowerCase().includes('<finish>') || 
        trimmed.toLowerCase() === 'is clean.' ||
        trimmed.toLowerCase() === 'is clean' ||
        trimmed.match(/^['\s,.]*$/) || // Xóa dòng chỉ có dấu phẩy, chấm, nháy đơn rác
        trimmed === '->' || trimmed === '=>' || // Xóa dòng chỉ có mũi tên
        trimmed.toLowerCase().includes('[kết thúc phần truyện]') ||
        trimmed.toLowerCase().includes('[bắt đầu phần truyện]') ||
        trimmed.toLowerCase().includes('[start writing]') ||
        trimmed.toLowerCase().includes('[writing_start]') ||
        trimmed.toLowerCase().includes('[story_start]') ||
        trimmed.toLowerCase().includes('[continue]') ||
        trimmed.toLowerCase().includes('[next]') ||
        trimmed.toLowerCase().includes('perspective:') ||
        trimmed.toLowerCase().includes('director\'s notes') ||
        trimmed.toLowerCase().includes('director notes') ||
        trimmed.toLowerCase().includes('system boot') ||
        trimmed.toLowerCase().includes('resource loading')
    ) return false;
    
    // Chỉ xóa dòng nếu nó CHỈ chứa một thẻ hệ thống (không có văn bản truyện đi kèm)
    if (trimmed.match(/^<\/?(set_time|time_cost|branches|thinking|content|story|tableEdit|table_stored|incrementalSummary).*?>$/i)) return false;
    
    // Loại bỏ các dòng log hệ thống đặc thù của Tawa
    if (trimmed.match(/^\d+:/)) return false; // Lọc bỏ rò rỉ LSR (0:Value)
    if (trimmed.includes('|')) return false; // Lọc bỏ rò rỉ LSR dạng bảng
    if (trimmed.match(/^\[(Tải Hiến Pháp|DATA SYNC|INPUT DECODING|ĐỒNG BỘ HÓA|Tải Biến Số|Luật Cấm|Đồng bộ Canon|Đồng bộ Trạng thái|Thiết lập thời gian|Địa điểm|Mục tiêu|Giai đoạn|Đoạn|Số chữ|Tiến độ|Tình trạng|Kiểm tra|Phân tích|Thinking|Content|Branches|Choices|Actions|LSR_SYNC|DATA_SYNC|INPUT_DECODING|SYNCHRONIZATION|Loading|Tải).*?\]/i)) return false;
    if (trimmed.match(/^\d+\.\s*\[(Tải Hiến Pháp|DATA SYNC|INPUT DECODING|ĐỒNG BỘ HÓA|Tải Biến Số|Luật Cấm|Mục tiêu|Giai đoạn|Đoạn|Số chữ|Tiến độ|Tình trạng|Kiểm tra|Phân tích).*?\]/i)) return false;
    if (trimmed.match(/^(\*|-) (User|NPC|Đồng bộ|Thiết lập|Địa điểm|Perspective|Cảnh báo|Lưu ý|Mục tiêu|Giai đoạn|Đoạn|Số chữ|Tiến độ|Tình trạng|Kiểm tra|Phân tích):/i)) return false;
    if (trimmed.match(/^(Mục tiêu|Giai đoạn|Đoạn|Số chữ|Tình trạng đạt chuẩn|Sắp xếp số chữ|Perspective|Cảnh báo|Lưu ý|Tiến độ|Tình trạng|Kiểm tra|Phân tích|Thinking|Content|Branches|Choices|Actions|LSR_SYNC|DATA_SYNC|INPUT_DECODING|SYNCHRONIZATION|Loading|Tải).*?:/i)) return false;
    if (trimmed.match(/^<!--.*?-->$/)) return false;
    if (trimmed.toLowerCase().includes('tiến hành viết chính văn')) return false;
    if (trimmed.toLowerCase().includes('bắt đầu viết')) return false;
    if (trimmed.toLowerCase().includes('đang viết')) return false;
    if (trimmed.toLowerCase().includes('kết thúc viết')) return false;
    if (trimmed.toLowerCase().includes('hoàn thành viết')) return false;
    if (trimmed.includes('[START WRITING]')) return false;
    if (trimmed.includes('[WRITING_START]')) return false;
    if (trimmed.includes('[STORY_START]')) return false;
    if (trimmed.includes('[CONTINUE]')) return false;
    if (trimmed.includes('[NEXT]')) return false;

    // Loại bỏ các dòng chỉ dẫn hệ thống tiếng Anh bị rò rỉ
    if (trimmed.startsWith('- Ensure') || 
        trimmed.startsWith('- STYLE') || 
        trimmed.startsWith('- ANTI-LEAKAGE') || 
        trimmed.startsWith('- STRUCTURE') ||
        trimmed.startsWith('- NO DIALOGUE') ||
        trimmed.startsWith('- NO NARRATIVE') ||
        trimmed.startsWith('- FORMAT') ||
        trimmed.startsWith('- PURITY') ||
        trimmed.startsWith('STYLE ENFORCEMENT') ||
        trimmed.startsWith('ANTI-LEAKAGE') ||
        trimmed.startsWith('STRUCTURE CHECK') ||
        trimmed.startsWith('CRITICAL_CHOICE_FORMATTING') ||
        trimmed.startsWith('TECHNICAL_FORMATTING_RULES')) {
      return false;
    }

    return true;
  });

  cleaned = filteredLines.join('\n').trim();

  // FALLBACK: Nếu sau khi lọc mà không còn gì, nhưng trước đó có văn bản, hãy trả về văn bản trước khi lọc
  // Điều này giúp tránh việc hiển thị "Nội dung truyện trống" khi AI viết truyện nhưng bị nhầm là hệ thống
  if (!cleaned && originalBeforeFiltering.trim().length > 10) {
      return originalBeforeFiltering.trim();
  }

  return cleaned;
};
