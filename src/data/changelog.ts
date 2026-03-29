
export interface ChangelogEntry {
    version: string;
    date: string;
    changes: string[];
}

export const CHANGELOG_DATA: ChangelogEntry[] = [
    {
        version: "v0.4.4",
        date: "2024-05-25",
        changes: [
            "💉 [AI Core] Thêm cơ chế 'Anti-Lazy Injection' để ngăn chặn hiện tượng AI viết ngắn dần sau nhiều lượt chat.",
            "🔄 [System] Tích hợp Style Reinforcement ngẫu nhiên vào mỗi lượt để duy trì chất lượng văn phong.",
            "🛠️ [Fix] Sửa lỗi Context Drift (trôi ngữ cảnh) khi hội thoại kéo dài."
        ]
    },
    {
        version: "v0.4.3",
        date: "2024-05-24",
        changes: [
            "✨ [World Creation] Thêm trường 'Kịch bản khởi đầu' (Starting Scenario) cho phép người chơi tùy chỉnh tình huống mở đầu game.",
            "📜 [Main Menu] Thêm mục 'Lịch sử cập nhật' (Version History) để theo dõi thay đổi.",
            "🧠 [AI Core] Tích hợp Vector RAG (Retrieval-Augmented Generation) để AI ghi nhớ các sự kiện quá khứ tốt hơn.",
            "🤖 [System] Tối ưu hóa logic xử lý LSR (Lorebook System) và Tawa Modules.",
            "🐛 [Fix] Sửa lỗi hiển thị Markdown trong chế độ Stream."
        ]
    },
    {
        version: "v0.4.2",
        date: "2024-05-20",
        changes: [
            "✨ [Gameplay] Thêm tính năng LSR (Lorebook Status Record) hiển thị bảng trạng thái động.",
            "🎨 [UI] Cải thiện giao diện hiển thị tin nhắn và các tùy chọn hành động.",
            "🚀 [System] Nâng cấp IndexedDB lên version 2 để hỗ trợ lưu trữ Vector.",
            "🧠 [AI] Cập nhật Preset Tawa Ultimate cho chất lượng văn phong tốt hơn."
        ]
    },
    {
        version: "v0.4.1",
        date: "2024-05-15",
        changes: [
            "✨ [Gameplay] Thêm tính năng Streaming Response (phản hồi từng chữ).",
            "🔧 [Settings] Thêm tùy chọn Thinking Budget cho Model Gemini 3.0 Pro.",
            "🐛 [Fix] Sửa lỗi tràn bộ nhớ khi lịch sử chat quá dài."
        ]
    },
    {
        version: "v0.4.0",
        date: "2024-05-01",
        changes: [
            "🎉 [Release] Phát hành phiên bản Ark v2 SillyTavern Alpha.",
            "✨ [Core] Hệ thống kiến tạo thế giới và nhân vật cơ bản.",
            "💾 [System] Hỗ trợ lưu trữ cục bộ (Local Storage/IndexedDB)."
        ]
    }
];
