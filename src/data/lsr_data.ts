
import { Lorebook } from "../services/ai/lorebook/types";

// LSR Table Preset v2.0.14 Data
export const LSR_PRESET: Lorebook = {
  "entries": {
    "0": {
      "key": [],
      "keysecondary": [],
      "comment": "===Hướng dẫn (Giữ ở trạng thái tắt)===",
      "content": "**Một preset bảng tăng cường trí nhớ tương đối nhẹ...**",
      "constant": true,
      "disable": true, // Disabled by default per instructions
      "order": 2000,
      "uid": 0
    },
    "1": {
      "key": [],
      "keysecondary": [],
      "comment": "Xử lý tham số",
      "content": "<%_ //Logic xử lý tham số phức tạp... _%>",
      "constant": true,
      "disable": false,
      "order": 2000,
      "uid": 1
    },
    "2": {
      "key": [],
      "keysecondary": [],
      "comment": "Tham số cấu hình",
      "content": "Set defaults...",
      "constant": true,
      "disable": false,
      "order": 2000,
      "uid": 2
    },
    "4": {
      "key": [],
      "keysecondary": [],
      "comment": "Prompt của bảng",
      "content": "<memory_table_guide>\nBảng trí nhớ định dạng CSV được lưu trong <table_stored> chứa dữ liệu...\n<table_format>\n...\n</table_format>\n</memory_table_guide>\n<table_stored>\nBảng trí nhớ hiện đang được lưu trữ mà phản hồi của bạn cần dựa vào:\n{{tableData}}\n</table_stored>",
      "constant": true,
      "disable": false,
      "order": 1000,
      "uid": 4
    },
    "6": {
      "key": [],
      "keysecondary": [],
      "comment": "Nhấn mạnh sau cùng",
      "content": "<table_rule>\n# Lưu ý về thao tác bảng\n## Cấu trúc bảng\n#0 Thông tin Hiện tại|0:Thời gian ({{getvar::tableConfigDateFormat}})|1:Địa điểm\n#1 Nhân vật Gần đây|0:Tên nhân vật|1:Số lượt vắng mặt|2:Mục tiêu tức thì|3:Vị trí|4:Tư thế|5:Trạng thái cơ thể|6:Trang phục|7:Trạng thái đặc biệt\n#2 Thông tin Nhân vật|0:Tên nhân vật|1:Giới tính|2:Tuổi|3:Thân phận|4:Đặc điểm cơ thể|5:Phong cách trang phục ưa thích|6:Tính cách|7:Sở thích|8:Mục tiêu dài hạn|9:Mối quan hệ|10:Thái độ với <user>|11:Mối quan hệ quan trọng giữa các nhân vật|12:Thiết lập bối cảnh quan trọng|13:Ghi chú quan trọng\n#3 Thông tin Tình dục|0:Tên nhân vật|1:Bộ phận cơ thể tương đối nhạy cảm|2:Lần đầu|3:Kỹ năng tình dục thành thạo|4:Chi tiết bộ phận riêng tư|5:Đối tượng tình dục gần đây|6:Ghi chú quan trọng\n#4 Lịch trình|0:Tóm tắt|1:Nội dung tổng thể|2:Tiến độ hiện tại|3:Người thực hiện|4:Người ủy thác|5:Phần thưởng hoàn thành|6:Địa điểm|7:Thời gian bắt đầu|8:Thời gian kết thúc hoặc giới hạn|9:Ghi chú quan trọng\n#5 Năng lực Đặc biệt|0:Năng lực|1:Người sở hữu|2:Công dụng|3:Hạn chế|4:Ghi chú quan trọng\n#6 Vật phẩm Quan trọng|0:Vật phẩm|1:Người sở hữu|2:Vị trí hiện tại|3:Số lượng|4:Hình thái|5:Công dụng|6:Hạn chế|7:Ghi chú quan trọng\n#7 Tổ chức Quan trọng|0:Tổ chức|1:Cấu trúc thành viên đã biết|2:Đặc điểm thành viên|3:Mục đích|4:Ghi chú quan trọng\n#8 Địa điểm Quan trọng|0:Địa điểm|1:Vị trí|2:Cấu trúc không gian|3:Ghi chú quan trọng\n#9 Tổng kết Lớn|0:Phạm vi thời gian|1:Nội dung\n#10 Lịch sử Sự kiện|0:Thời gian|1:Địa điểm|2:Sự kiện\n#11 Chỉ dẫn Quan trọng|0:Nội dung\n\n<format_request>\nBạn phải dựa vào các yêu cầu liên quan và nội dung của <table_stored>, hoàn thành <tableThink>, <tableEdit>...\n</format_request>",
      "constant": true,
      "disable": false,
      "order": 1000,
      "uid": 6
    }
  }
};
