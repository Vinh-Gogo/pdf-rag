from PIL import Image
import io
import pymupdf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class PDFPageProcessor:
    def __init__(self, pdf_path, page_num=0):
        self.pdf_path = pdf_path
        self.page_num = page_num
        self.doc = None
        self.page = None
        self.words = None
        self.pix = None
        self.img = None
        self.blocks = None
        self.sorted_words = None
        self.sorted_blocks = None
        self.merged_blocks = None
        self.final_blocks = None
        self.title_blocks = None
        self.body_blocks = None
        self.block_sizes = None
        self.title_levels = None  # list of tuples: (block, level)
        self.level_palette = [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
            "#ff7f00", "#a65628", "#f781bf", "#999999"
        ]
        # Grouping containers (fixed ranges)
        self.range_groups = None  # dict: {"title": [blocks], "body": [blocks], "header_footer": [blocks]}

    def load_page(self):
        self.doc = pymupdf.open(self.pdf_path)
        self.page = self.doc[self.page_num]

    def get_words_and_image(self):
        self.words = self.page.get_text("words")
        self.pix = self.page.get_pixmap(matrix=pymupdf.Matrix(1, 1))
        self.img = Image.open(io.BytesIO(self.pix.tobytes("png")))

    def extract_blocks(self):
        page_dict = self.page.get_text("dict")
        self.blocks = []
        for block in page_dict["blocks"]:
            if block["type"] == 0:  # chỉ text block
                x0, y0, x1, y1 = block["bbox"]
                spans = [span for line in block["lines"] for span in line["spans"]]
                self.blocks.append((x0, y0, x1, y1, spans, block["number"], 0))

    def sort_elements(self):
        self.sorted_words = sorted(self.words, key=lambda w: (w[1], w[0]))
        self.sorted_blocks = sorted(self.blocks, key=lambda b: (b[1], b[0]))

    def merge_blocks(self):
        self.merged_blocks = []
        for block in self.sorted_blocks:
            x0, y0, x1, y1, spans, bno, _ = block
            if self.merged_blocks:
                prev_x0, prev_y0, prev_x1, prev_y1, prev_spans, prev_bno, _ = self.merged_blocks[-1]
                if y0 - prev_y1 < 0.2:  # khoảng cách theo y nhỏ hơn 5 pixel
                    # Gộp block cũ và block mới
                    new_x0 = min(prev_x0, x0)
                    new_y0 = min(prev_y0, y0)
                    new_x1 = max(prev_x1, x1)
                    new_y1 = max(prev_y1, y1)
                    new_spans = prev_spans + spans
                    self.merged_blocks[-1] = (new_x0, new_y0, new_x1, new_y1, new_spans, prev_bno, 0)
                    continue
            self.merged_blocks.append(block)

    def split_blocks(self, x_gap: float = 4.0, y_gap: float = 10.0):
        """
        Cắt block theo cả chiều ngang (x) và chiều dọc (y).
        - x_gap: nếu khoảng cách giữa span hiện tại và biên phải nhóm hiện tại > x_gap => tách cột (sub-block theo x)
        - y_gap: trong từng cột, nếu khoảng cách theo y giữa span hiện tại và biên dưới nhóm hiện tại > y_gap => tách đoạn (sub-block theo y)
        Kết quả self.final_blocks luôn có dạng: (x0, y0, x1, y1, spans(list), bno, 0)
        """
        self.final_blocks = []
        for merged_block in self.merged_blocks:
            x0, y0, x1, y1, spans, bno, _ = merged_block
            if not spans:
                self.final_blocks.append(merged_block)
                continue
            # 1) Chia theo chiều ngang thành các cột dựa trên x_gap
            # spans_by_x = sorted(spans, key=lambda s: (s["bbox"][0])
            spans_by_x = sorted(spans, key=lambda s: (s["bbox"][0]))
            x_columns = []
            current_col = [spans_by_x[0]]
            current_x1 = spans_by_x[0]["bbox"][2]
            for span in spans_by_x[1:]:
                if span["bbox"][0] - current_x1 > x_gap:
                    x_columns.append(current_col)
                    current_col = [span]
                    current_x1 = span["bbox"][2]
                else:
                    current_col.append(span)
                    current_x1 = max(current_x1, span["bbox"][2])
            x_columns.append(current_col)

            # 2) Trong mỗi cột, chia theo chiều dọc dựa trên y_gap
            for col_spans in x_columns:
                col_sorted = sorted(col_spans, key=lambda s: (s["bbox"][1]))
                y_groups = []
                current_group = [col_sorted[0]]
                current_y1 = col_sorted[0]["bbox"][3]
                for span in col_sorted[1:]:
                    if span["bbox"][1] - current_y1 > y_gap:
                        y_groups.append(current_group)
                        current_group = [span]
                        current_y1 = span["bbox"][3]
                    else:
                        current_group.append(span)
                        current_y1 = max(current_y1, span["bbox"][3])
                y_groups.append(current_group)

                # 3) Tạo final blocks từ các nhóm theo y
                for group in y_groups:
                    if not group:
                        continue
                    group_x0 = min(s["bbox"][0] for s in group)
                    group_y0 = min(s["bbox"][1] for s in group)
                    group_x1 = max(s["bbox"][2] for s in group)
                    group_y1 = max(s["bbox"][3] for s in group)
                    self.final_blocks.append((group_x0, group_y0, group_x1, group_y1, group, bno, 0))
                    self.final_blocks.sort(key=lambda b: ( b[1], b[0], b[5]))

    def classify_title_levels(self, method: str = "max", title_tolerance: float = 0.1, level_tolerance: float = 0.5):
        """
        Phân cấp title theo size font:
        - method: "avg" dùng trung bình size của spans trong block, hoặc "max" dùng size lớn nhất
        - title_tolerance: dung sai khi xác định block nào là title (gần bằng kích thước lớn nhất)
                - level_tolerance: dung sai (absolute) khi gom các size gần nhau vào 1 cluster; nếu muốn tách nhiều cấp hơn, giảm thông số này.
        Kết quả:
          - self.title_blocks, self.body_blocks
          - self.block_sizes: list kích thước đại diện cho mỗi block (cùng thứ tự final_blocks)
          - self.title_levels: list (block, level_index) với level 1 là lớn nhất
        """
        if not self.final_blocks:
            self.title_blocks = []
            self.body_blocks = []
            self.block_sizes = []
            self.title_levels = []
            return

        # 1) Tính size đại diện cho từng block
        sizes = []
        for blk in self.final_blocks:
            spans = blk[4]
            if not spans:
                sizes.append(0.0)
                continue
            if method == "max":
                val = max(float(s.get("size", 0.0)) for s in spans)
            else:
                val = sum(float(s.get("size", 0.0)) for s in spans) / max(1, len(spans))
            sizes.append(val)
        self.block_sizes = sizes

        # 2) Xác định title: kích thước gần bằng kích thước lớn nhất
        max_size = max(sizes) if sizes else 0.0
        self.title_blocks = []
        self.body_blocks = []
        for blk, sz in zip(self.final_blocks, sizes):
            if abs(sz - max_size) <= title_tolerance:
                self.title_blocks.append(blk)
            else:
                self.body_blocks.append(blk)

        # 3) Gom các kích thước của TITLE thành các cấp (level)
        title_sizes = []
        for blk in self.title_blocks:
            spans = blk[4]
            if not spans:
                title_sizes.append(0.0)
            else:
                if method == "max":
                    title_sizes.append(max(float(s.get("size", 0.0)) for s in spans))
                else:
                    title_sizes.append(sum(float(s.get("size", 0.0)) for s in spans) / max(1, len(spans)))

        # Tạo các mức (levels) không làm "hợp nhất" mất nhiều phân cấp.
        # Chiến lược: sort giảm dần, mỗi size so với đại diện level trước đó; nếu chênh > level_tolerance => tạo level mới.
        # Điều này giúp giữ được nhiều cấp khi có nhiều font size khác nhau.
        distinct = []
        for sz in sorted(title_sizes, reverse=True):
            if not distinct:
                distinct.append(sz)
                continue
            rep = distinct[-1]  # đại diện level cuối cùng (nhỏ hơn hoặc bằng các trước đó vì sort reverse)
            if abs(rep - sz) > level_tolerance:
                distinct.append(sz)
            else:
                # không merge để tránh giảm phân cấp; bỏ qua (giữ rep đầu tiên lớn hơn)
                pass

        # Map kích thước -> level index (1-based)
        def size_to_level(val: float) -> int:
            # Gán vào level có rep gần nhất (không yêu cầu trong tolerance để vẫn phân bổ)
            best_level = None
            best_diff = float("inf")
            for idx, rep in enumerate(distinct, start=1):
                d = abs(val - rep)
                if d < best_diff:
                    best_diff = d
                    best_level = idx
            return best_level or 1

        self.title_levels = []
        for blk in self.title_blocks:
            spans = blk[4]
            if not spans:
                self.title_levels.append((blk, len(distinct)))
            else:
                if method == "max":
                    val = max(float(s.get("size", 0.0)) for s in spans)
                else:
                    val = sum(float(s.get("size", 0.0)) for s in spans) / max(1, len(spans))
                lvl = size_to_level(val)
                self.title_levels.append((blk, lvl))

    def classify_by_size_ranges(self, method: str = "avg"):
        """
        Phân nhóm cố định theo size:
        - Title:     16 .. 100
        - Body:      11 .. 15
        - Header/Footnote: 1 .. 10
        """
        if not self.final_blocks:
            self.range_groups = {"title": [], "body": [], "header_footer": []}
            return
        def rep_size(spans):
            if not spans:
                return 0.0
            if method == "max":
                return max(float(s.get("size", 0.0)) for s in spans)
            return sum(float(s.get("size", 0.0)) for s in spans) / max(1, len(spans))

        title, body, hf = [], [], []
        for blk in self.final_blocks:
            spans = blk[4]
            sz = rep_size(spans)
            if 16 <= sz <= 100:
                title.append(blk)
            elif 11 <= sz < 16:
                body.append(blk)
            else:  # 0..10 (and negatives default to hf)
                hf.append(blk)
        self.range_groups = {"title": title, "body": body, "header_footer": hf}

    def plot(self):
        fig, ax = plt.subplots(1, figsize=(12, 16))
        ax.imshow(self.img)

        # # 🟩 Bounding box cho từng từ
        # for idx, w in enumerate(self.sorted_words):
        #     x0, y0, x1, y1, text, block_no, line_no, word_no = w
        #     ax.add_patch(patches.Rectangle((float(x0), float(y0)), float(x1)-float(x0), float(y1)-float(y0),
        #                                    linewidth=0.8, edgecolor='lime', facecolor='none', alpha=0.7))
        #     ax.text(float(x0), float(y0)-3, text, color='red', fontsize=6,
        #             bbox=dict(boxstyle='round', facecolor='yellow', alpha=.5))

        # 🟥 Bounding box cho block: ưu tiên vẽ theo cấp độ title nếu có
        def _repr_size(spans):
            if not spans:
                return 0
            return round(sum(float(s.get("size", 0.0)) for s in spans) / max(1, len(spans)))

        if self.range_groups:
            # Ưu tiên hiển thị theo nhóm cố định nếu đã phân nhóm
            # Vẽ header/footer (xám), rồi body (cam), rồi title (đỏ)
            def _repr_size(spans):
                if not spans:
                    return 0
                return round(sum(float(s.get("size", 0.0)) for s in spans) / max(1, len(spans)))

            for idx, b in enumerate(sorted(self.range_groups.get("header_footer", []), key=lambda bl: (bl[1], bl[0]))):
                x0, y0, x1, y1, spans, bno, _ = b
                size_val = _repr_size(spans)
                ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                               linewidth=1.5, edgecolor='#666666', facecolor='none', alpha=0.6))
                ax.text(x0, y0-3, f'Header/Footer {idx+1} | {size_val}', color='#666666', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=.6))

            for idx, b in enumerate(sorted(self.range_groups.get("body", []), key=lambda bl: (bl[1], bl[0]))):
                x0, y0, x1, y1, spans, bno, _ = b
                size_val = _repr_size(spans)
                ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                               linewidth=2, edgecolor='orange', facecolor='none', alpha=0.7))
                ax.text(x0, y0-3, f'Body {idx+1} | {size_val}', color='blue', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=.7))

            for idx, b in enumerate(sorted(self.range_groups.get("title", []), key=lambda bl: (bl[1], bl[0]))):
                x0, y0, x1, y1, spans, bno, _ = b
                size_val = _repr_size(spans)
                ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                               linewidth=2.5, edgecolor='red', facecolor='none', alpha=0.95))
                ax.text(x0, y0-3, f'Title {idx+1} | {size_val}', color='red', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=.95))

        elif self.title_levels:
            # vẽ body trước
            for idx, b in enumerate(self.body_blocks or []):
                x0, y0, x1, y1, spans, bno, _ = b
                size_val = _repr_size(spans)
                ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                               linewidth=2, edgecolor='orange', facecolor='none', alpha=0.6))
                ax.text(x0, y0-3, f'Body {idx+1} | {size_val}', color='blue', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=.6))
            # sắp xếp title theo vị trí từ trên xuống (y0 tăng dần), rồi đánh số thứ tự
            ordered_titles = sorted(self.title_levels, key=lambda t: (t[0][1], t[0][0]))
            for ordinal_idx, (blk, lvl) in enumerate(ordered_titles, start=1):
                x0, y0, x1, y1, spans, bno, _ = blk
                color = self.level_palette[(lvl-1) % len(self.level_palette)]
                size_val = _repr_size(spans)
                ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                               linewidth=2.5, edgecolor=color, facecolor='none', alpha=0.95))
                # Hiển thị: Title n | size
                ax.text(x0, y0-3, f'Title {ordinal_idx} | {size_val}', color=color, fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=.95))
        elif self.title_blocks is not None and self.body_blocks is not None:
            # fallback: chỉ có title/body
            for idx, b in enumerate(self.body_blocks or []):
                x0, y0, x1, y1, spans, bno, _ = b
                size_val = _repr_size(spans)
                ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                               linewidth=2, edgecolor='orange', facecolor='none', alpha=0.7))
                ax.text(x0, y0-3, f'Body {idx+1} | {size_val}', color='blue', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=.7))
            for idx, b in enumerate(self.title_blocks or []):
                x0, y0, x1, y1, spans, bno, _ = b
                size_val = _repr_size(spans)
                ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                               linewidth=2.5, edgecolor='red', facecolor='none', alpha=0.9))
                ax.text(x0, y0-3, f'Title {idx+1} | {size_val}', color='red', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=.9))
        else:
            for idx, b in enumerate(self.final_blocks):
                x0, y0, x1, y1, spans, bno, _ = b
                size_val = _repr_size(spans)
                ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                               linewidth=2, edgecolor='orange', facecolor='none', alpha=0.7))
                ax.text(x0, y0-3, f'Block {idx+1} | {size_val}', color='blue', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=.7))

        ax.axis('off')
        plt.tight_layout()
        plt.show()

# Usage example
if __name__ == "__main__":
    for i in range(0,168):
        pdf_path = fr"src/data/pdfs/pages/page_{i+1}.pdf"
        processor = PDFPageProcessor(pdf_path, page_num=0)
        processor.load_page()
        processor.get_words_and_image()
        processor.extract_blocks()
        processor.sort_elements()
        processor.merge_blocks()
        processor.split_blocks(x_gap=2.0, y_gap=2.0)
        processor.classify_title_levels(method="max", title_tolerance=0.01, level_tolerance=0.5)
        # Đếm theo level để in
        level_counts = {}
        for _, lvl in (processor.title_levels or []):
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
        counts_str = ", ".join([f"L{lvl}:{cnt}" for lvl, cnt in sorted(level_counts.items())]) or "no titles"
        print(f"✅ Processed page {i+1}: titles by level -> {counts_str}; body blocks: {len(processor.body_blocks or [])}")
        processor.plot()
