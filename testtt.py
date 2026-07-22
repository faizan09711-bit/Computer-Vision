import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
#
# Every failure mode described in the brief traces back to one root cause:
# the original code hardcoded assumptions about *absolute* pixel values
# (">15 is paper", ">240 is a bright seam", "250 is roughly white") instead
# of measuring what "background" and "foreground" actually look like in the
# image in front of it. The fixes below all follow the same recipe:
#
#   1. Estimate the real background color/brightness from the image itself
#      (border sampling, since fragments are never flush against the frame
#      edge and gaps in a stitched canvas are always the initial zero-fill).
#   2. Turn that into a distance-from-background map (works for color and
#      grayscale alike, and for backgrounds/paper of any tone).
#   3. Let Otsu find the split point on that distance map, where it's
#      guaranteed to be roughly bimodal - instead of on raw brightness, where
#      it's only bimodal if the background happens to be white/black.
#   4. Fall back to a gradient/edge-based method (Canny) when the distance
#      map still doesn't produce a sane result, since edges don't care about
#      absolute brightness at all.


def _estimate_background_color(img_bgr):
    """Sample a thin strip around the image border to estimate the true
    background color, instead of assuming it's white (or black).

    Fragments/paper are placed away from the frame edge in essentially every
    real photo of torn paper on a surface, and a stitched canvas is
    zero-padded on all sides by merge_patch_to_canvas, so the border is a
    reliable place to measure "background" regardless of what color it is.
    """
    h, w = img_bgr.shape[:2]
    border = max(2, min(h, w) // 40)
    strips = [
        img_bgr[:border, :].reshape(-1, 3),
        img_bgr[-border:, :].reshape(-1, 3),
        img_bgr[:, :border].reshape(-1, 3),
        img_bgr[:, -border:].reshape(-1, 3),
    ]
    samples = np.concatenate(strips, axis=0)
    return np.median(samples, axis=0).astype(np.float32)


def _distance_from_background(img_bgr, bg_color):
    """Per-pixel Euclidean distance (in color space) from the estimated
    background color, normalized to 0-255. This is the quantity Otsu should
    actually be thresholding - it is bimodal (near 0 for background, large
    for anything else) no matter what color the background or the paper is.
    """
    diff = np.linalg.norm(
        img_bgr.astype(np.float32) - bg_color.astype(np.float32), axis=2
    )
    if diff.max() < 1e-3:
        # Degenerate/flat patch (e.g. a tiny solid-color sliver) - there is
        # no meaningful background/foreground split to find, so treat the
        # whole thing as foreground rather than dividing by zero.
        return None
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def get_mask(img):
    """Foreground (paper) vs. background mask for a single image/patch.

    Original approach: `(gray > 15)`. This assumes the background is always
    much darker than the paper by at least 15 gray levels. It breaks as soon
    as:
      - the paper itself has dark regions (black ink, dark/aged paper, deep
        shadow folds) that fall below the 15 cutoff and get carved out of the
        fragment, or
      - the background isn't near-black (a tinted scanning surface, a photo
        with warm ambient lighting casting a mid-gray background).

    Fix: estimate the background color from the border, threshold the
    distance-from-background map with Otsu (adaptive, not a fixed constant),
    and clean up with a morphological open + "keep the largest connected
    component" pass, since a genuine paper fragment is a single blob and
    anything else surviving the threshold is noise.
    """
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img

    bg_color = _estimate_background_color(img_bgr)
    diff_u8 = _distance_from_background(img_bgr, bg_color)

    if diff_u8 is None:
        return np.full(img_bgr.shape[:2], 255, dtype=np.uint8)

    _, mask = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    return mask


def _edge_based_mask(img):
    """Edge/gradient-based fragment mask. Unlike any color- or
    brightness-threshold approach, this only cares about local contrast at a
    fragment's boundary, so it still finds a fragment whose fill color is
    close to the background's (a case a single global color-distance
    threshold can genuinely lose, since Otsu is choosing one cut point for
    the *whole* frame and a low-contrast fragment's distance values can get
    swamped by higher-contrast fragments elsewhere in the same histogram).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    filled = edges.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, flood_mask, (0, 0), 255)
    interior = cv2.bitwise_not(filled)
    return cv2.bitwise_or(interior, edges)


def _segment_foreground(img):
    """Full-frame segmentation used to find fragment contours in the source
    photo.

    We combine two independent signals rather than treating the Canny path
    as a rare fallback:
      - a color-distance-from-background mask (great for fragments that
        clearly differ from the background, robust to non-white surfaces),
      - an edge-based mask (great for fragments whose color is close to the
        background - a single global Otsu cut can quietly swallow a
        low-contrast fragment into "background" when other, more distinct
        fragments dominate the same histogram).
    Taking the union catches both cases; a fragment only needs one of the
    two signals to be picked up correctly.
    """
    h, w = img.shape[:2]
    bg_color = _estimate_background_color(img)
    diff_u8 = _distance_from_background(img, bg_color)

    color_mask = None
    if diff_u8 is not None:
        _, candidate = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_ratio = np.count_nonzero(candidate) / float(h * w)
        # A sane segmentation shouldn't call almost the whole frame (or
        # almost none of it) "paper" - that's the signature of Otsu having
        # no real valley to split on, so we discard it rather than union in
        # something that's mostly noise.
        if 0.01 <= fg_ratio <= 0.95:
            color_mask = candidate

    edge_mask = _edge_based_mask(img)

    if color_mask is None:
        return edge_mask
    return cv2.bitwise_or(color_mask, edge_mask)


def display_detected_fragments(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image could not be loaded. Check your file path.")
        return None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape

    fg_mask = _segment_foreground(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = (h * w) * 0.005
    fragments = [c for c in contours if cv2.contourArea(c) > min_area]

    temporary_storage = []
    for idx, contour in enumerate(fragments):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        piece = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

        x, y, pw, ph = cv2.boundingRect(contour)
        cropped_piece = piece[y : y + ph, x : x + pw]

        cropped_bgr = cv2.cvtColor(cropped_piece, cv2.COLOR_RGB2BGR)

        temporary_storage.append(
            {"img": cropped_bgr, "orig_x": x, "orig_y": y}
        )

    return temporary_storage, w


def find_best_alignment(canvas, patch):
    """Generically aligns a new patch onto the canvas by finding the translation
    (dx, dy) that maximizes edge boundary contact while minimizing overlap.

    Original approach used fixed pixel constants everywhere: a search step of
    8px, a "touching" cutoff of 5.0px, a "deep overlap" cutoff of 6.0px. Those
    numbers were implicitly tuned for one particular fragment resolution.
    Fed a 4K scan, the search becomes far too coarse to land a good fit;
    fed a small thumbnail, the same absolute distances swallow the entire
    fragment and every candidate looks like "deep overlap". Scaling every
    threshold and step size to the patch's own diagonal keeps the heuristic's
    behavior consistent across fragment/canvas resolutions.
    """
    m_canvas = get_mask(canvas)
    m_patch = get_mask(patch)

    h_c, w_c = canvas.shape[:2]
    h_p, w_p = patch.shape[:2]

    patch_scale = max(1.0, float(np.hypot(h_p, w_p)))
    touching_thresh = max(2.0, patch_scale * 0.01)
    deep_overlap_thresh = max(3.0, patch_scale * 0.012)
    coarse_step = max(2, int(patch_scale * 0.02))
    fine_step = max(1, coarse_step // 4)

    canny_p = cv2.Canny(m_patch, 50, 150)
    y_p, x_p = np.where(canny_p == 255)

    if len(y_p) == 0:
        return 0, 0, float("inf")

    max_edge_points = 350
    if len(y_p) > max_edge_points:
        idx = np.linspace(0, len(y_p) - 1, max_edge_points, dtype=int)
        y_p, x_p = y_p[idx], x_p[idx]

    dist_out = cv2.distanceTransform(~m_canvas, cv2.DIST_L2, 5)
    dist_in = cv2.distanceTransform(m_canvas, cv2.DIST_L2, 5)

    def score_at(dx, dy):
        sy, sx = y_p + dy, x_p + dx
        valid = (sy >= 0) & (sy < h_c) & (sx >= 0) & (sx < w_c)
        if np.sum(valid) < 10:
            return None

        vy, vx = sy[valid], sx[valid]
        overlap = m_canvas[vy, vx] == 255
        non_overlap_y, non_overlap_x = vy[~overlap], vx[~overlap]

        touching = (
            np.sum(dist_out[non_overlap_y, non_overlap_x] <= touching_thresh)
            if len(non_overlap_y) > 0
            else 0
        )
        deep_overlap = (
            np.sum(dist_in[vy[overlap], vx[overlap]] > deep_overlap_thresh)
            if np.any(overlap)
            else 0
        )
        return -touching + (5.0 * deep_overlap)

    best_dx, best_dy = 0, 0
    min_score = float("inf")

    # Coarse search across potential bounding overlaps.
    for dy in range(-int(h_p * 0.7), int(h_c * 0.9), coarse_step):
        for dx in range(-int(w_p * 0.7), int(w_c * 0.9), coarse_step):
            score = score_at(dx, dy)
            if score is not None and score < min_score:
                min_score = score
                best_dx, best_dy = dx, dy

    # Fine search refinement around the coarse winner.
    for dy in range(best_dy - coarse_step, best_dy + coarse_step + 1, fine_step):
        for dx in range(best_dx - coarse_step, best_dx + coarse_step + 1, fine_step):
            score = score_at(dx, dy)
            if score is not None and score < min_score:
                min_score = score
                best_dx, best_dy = dx, dy

    return best_dx, best_dy, min_score


def merge_patch_to_canvas(canvas, patch, dx, dy):
    """Generically expands canvas if needed and overlays patch onto canvas at
    (dx, dy).

    Original approach placed the patch with a hard boolean mask
    (`patch_area[m_patch == 255] = patch[...]`). Any anti-aliased or slightly
    blurred fragment edge (common after JPEG compression or a soft camera
    focus) has partially-transparent border pixels; a hard mask either keeps
    or drops each one completely, stamping a visible "cookie-cutter" ring
    around every fragment. Feathering the last ~1% of the patch's width via
    its own inward distance transform blends those edge pixels smoothly into
    whatever is already at that location (existing canvas content or the
    zero background), which is what apply_inpainting_transformation's seam
    detector was really fighting against before.
    """
    h_c, w_c = canvas.shape[:2]
    h_p, w_p = patch.shape[:2]

    m_patch = get_mask(patch)

    oy, ox = max(0, -dy), max(0, -dx)
    new_h = max(h_c + oy, h_p + dy + oy)
    new_w = max(w_c + ox, w_p + dx + ox)

    new_canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    new_canvas[oy : oy + h_c, ox : ox + w_c] = canvas

    patch_target_y = dy + oy
    patch_target_x = dx + ox

    region = new_canvas[
        patch_target_y : patch_target_y + h_p,
        patch_target_x : patch_target_x + w_p,
    ].astype(np.float32)

    feather_px = max(1.0, min(h_p, w_p) * 0.01)
    dist_in = cv2.distanceTransform(m_patch, cv2.DIST_L2, 5)
    alpha = np.clip(dist_in / feather_px, 0.0, 1.0)
    alpha[m_patch == 0] = 0.0
    alpha = alpha[..., None]

    blended = region * (1 - alpha) + patch.astype(np.float32) * alpha
    new_canvas[
        patch_target_y : patch_target_y + h_p,
        patch_target_x : patch_target_x + w_p,
    ] = np.clip(blended, 0, 255).astype(np.uint8)

    return new_canvas


def display_final_canvas(raw_fragment_metadata):
    """Fully generalized dynamic stitching loop using greedy best-match edge search."""
    if not raw_fragment_metadata:
        return None

    unstitched = [item["img"] for item in raw_fragment_metadata]

    # Sort fragments by size (area) to start with the largest piece as base canvas
    unstitched.sort(
        key=lambda img: np.count_nonzero(get_mask(img)), reverse=True
    )

    canvas = unstitched.pop(0)

    # Greedily pick and stitch the next best-fitting fragment
    while unstitched:
        best_candidate_idx = 0
        best_dx, best_dy = 0, 0
        best_score = float("inf")

        for idx, patch in enumerate(unstitched):
            dx, dy, score = find_best_alignment(canvas, patch)
            if score < best_score:
                best_score = score
                best_dx, best_dy = dx, dy
                best_candidate_idx = idx

        # Merge the winning piece
        winning_patch = unstitched.pop(best_candidate_idx)
        canvas = merge_patch_to_canvas(canvas, winning_patch, best_dx, best_dy)

    return canvas


def apply_inpainting_transformation(canvas):
    """Original approach flagged "seams" purely by absolute brightness
    (`gray > 240`), assuming any gap left behind is close to pure white. That
    only holds if the surface the pieces were placed on happened to be
    near-white. Our own canvas background is guaranteed pure black (see
    merge_patch_to_canvas's `np.zeros` initialization), and a real seam along
    a torn or feathered edge is usually a *blend* of the neighboring paper
    tones - it can be any brightness, including quite dark on dark paper.

    Fix: keep the mask-hole detection (gaps enclosed by paper that didn't
    get picked up as paper - this part was already brightness-agnostic), and
    replace the fixed 240 cutoff with a local-contrast test: compare each
    pixel to a locally median-blurred version of the paper and threshold the
    *deviation* with Otsu. A seam stands out as a local anomaly relative to
    its neighborhood regardless of whether that neighborhood is bright,
    dark, or colored.
    """
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    paper_mask = get_mask(canvas)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_paper = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel)
    gaps_mask = cv2.bitwise_and(closed_paper, cv2.bitwise_not(paper_mask))

    local_bg = cv2.medianBlur(gray, 9)
    local_contrast = cv2.absdiff(gray, local_bg)
    _, contrast_seams = cv2.threshold(
        local_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    seams_mask = cv2.bitwise_or(gaps_mask, contrast_seams)
    seams_mask = cv2.bitwise_and(seams_mask, closed_paper)

    inpainted = cv2.inpaint(
        canvas, seams_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )

    lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    enhanced_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

    # Block size must scale with resolution (and stay odd): a fixed size of
    # 3 is far too small for high-resolution scans (produces salt-and-pepper
    # noise in the binarized output) and wastes the adaptivity on tiny
    # thumbnails. maxValue is also corrected to 255 (the original passed 200,
    # which silently produced a non-standard binary image of 0/200 instead
    # of 0/255).
    block = max(3, (min(enhanced_gray.shape[:2]) // 20) | 1)
    binarized = cv2.adaptiveThreshold(
        enhanced_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        2,
    )

    return binarized


class ApplicationWindow:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Document Fragment Stitcher")
        self.root.geometry("560x540")
        self.root.configure(bg="#0F172A")
        self.root.resizable(False, False)

        self.selected_path = None
        self.input_preview_photo = None
        self.setup_ui()

    def setup_ui(self):
        header = tk.Frame(self.root, bg="#1E293B", height=60)
        header.pack(fill=tk.X, side=tk.TOP)

        lbl_title = tk.Label(
            header,
            text="✨ Document Reconstruction Engine",
            font=("Segoe UI", 15, "bold"),
            bg="#1E293B",
            fg="#38BDF8",
        )
        lbl_title.pack(pady=12)

        body = tk.Frame(self.root, bg="#0F172A", padx=25, pady=20)
        body.pack(fill=tk.BOTH, expand=True)

        file_frame = tk.Frame(
            body, bg="#1E293B", bd=1, relief=tk.SOLID, padx=12, pady=10
        )
        file_frame.pack(fill=tk.X, pady=(0, 15))

        self.btn_browse = tk.Button(
            file_frame,
            text="📁 Browse Original Image",
            font=("Segoe UI", 10, "bold"),
            bg="#2563EB",
            fg="white",
            activebackground="#1D4ED8",
            activeforeground="white",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            command=self.browse_file,
        )
        self.btn_browse.pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_path = tk.Label(
            file_frame,
            text="No image selected...",
            font=("Segoe UI", 9, "italic"),
            bg="#1E293B",
            fg="#94A3B8",
            anchor="w",
        )
        self.lbl_path.pack(side=tk.LEFT, fill=tk.X, expand=True)

        preview_card = tk.LabelFrame(
            body,
            text=" Original Image Preview ",
            font=("Segoe UI", 10, "bold"),
            bg="#1E293B",
            fg="#F1F5F9",
            bd=1,
            relief=tk.SOLID,
            padx=10,
            pady=10,
        )
        preview_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        self.lbl_preview = tk.Label(
            preview_card,
            text=(
                "🖼️ No image loaded yet\nClick 'Browse Original Image' above"
                " to load"
            ),
            font=("Segoe UI", 10),
            bg="#0F172A",
            fg="#64748B",
        )
        self.lbl_preview.pack(fill=tk.BOTH, expand=True)

        self.btn_convert = tk.Button(
            body,
            text="🧩 Stitch Fragments",
            font=("Segoe UI", 11, "bold"),
            bg="#10B981",
            fg="white",
            activebackground="#059669",
            activeforeground="white",
            disabledforeground="#475569",
            bd=0,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.process_image,
        )
        self.btn_convert.pack(fill=tk.X, pady=(0, 10))

        self.lbl_status = tk.Label(
            body,
            text="Status: Ready",
            font=("Segoe UI", 9, "bold"),
            bg="#0F172A",
            fg="#94A3B8",
        )
        self.lbl_status.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File from PC",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            self.selected_path = file_path
            self.lbl_path.config(
                text=os.path.basename(file_path), fg="#F8FAFC"
            )
            self.btn_convert.config(state=tk.NORMAL, bg="#10B981")
            self.lbl_status.config(
                text="Status: Original Image Loaded. Click 'Stitch Fragments'!",
                fg="#38BDF8",
            )

            pil_img = Image.open(file_path)
            pil_img.thumbnail((260, 190))
            self.input_preview_photo = ImageTk.PhotoImage(pil_img)
            self.lbl_preview.config(
                image=self.input_preview_photo, text="", bg="#1E293B"
            )

    def process_image(self):
        if not self.selected_path:
            return

        self.btn_convert.config(state=tk.DISABLED, bg="#334155")
        self.btn_browse.config(state=tk.DISABLED)
        self.lbl_status.config(
            text="Status: ⏳ Dynamically matching & stitching fragments...",
            fg="#FBBF24",
        )
        self.root.update()

        raw_fragment_metadata, _ = display_detected_fragments(
            self.selected_path
        )
        if not raw_fragment_metadata:
            messagebox.showerror(
                "Error",
                "No distinct paper fragments found or image path error.",
            )
            self.reset_ui()
            return

        stitched_canvas = display_final_canvas(raw_fragment_metadata)

        self.root.withdraw()
        ResultViewer(self.root, stitched_canvas)

    def reset_ui(self):
        self.btn_convert.config(state=tk.NORMAL, bg="#10B981")
        self.btn_browse.config(state=tk.NORMAL)
        self.lbl_status.config(text="Status: Ready", fg="#94A3B8")

    def run(self):
        self.root.mainloop()


class ResultViewer:

    def __init__(self, main_root, stitched_img):
        self.main_root = main_root
        self.stitched_img = stitched_img
        self.current_display_img = stitched_img
        self.inpainted_img = None

        self.window = tk.Toplevel()
        self.window.title("Stitched Canvas Viewer")
        self.window.geometry("820x700")
        self.window.configure(bg="#0F172A")
        self.window.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.setup_ui()

    def setup_ui(self):
        header = tk.Frame(self.window, bg="#1E293B", padx=15, pady=10)
        header.pack(fill=tk.X, side=tk.TOP)

        self.btn_inpaint = tk.Button(
            header,
            text="✨ Apply Inpainting",
            font=("Segoe UI", 10, "bold"),
            bg="#8B5CF6",
            fg="white",
            activebackground="#7C3AED",
            activeforeground="white",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            command=self.trigger_inpainting,
        )
        self.btn_inpaint.pack(side=tk.LEFT, padx=(0, 10))

        btn_save = tk.Button(
            header,
            text="💾 Save Image",
            font=("Segoe UI", 10, "bold"),
            bg="#059669",
            fg="white",
            activebackground="#047857",
            activeforeground="white",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            command=self.save_image,
        )
        btn_save.pack(side=tk.LEFT)

        self.title_lbl = tk.Label(
            header,
            text="Stitched Image (Generalized Dynamic Stitcher)",
            font=("Segoe UI", 12, "bold"),
            bg="#1E293B",
            fg="#38BDF8",
        )
        self.title_lbl.pack(side=tk.LEFT, expand=True)

        btn_exit = tk.Button(
            header,
            text="❌ Exit",
            font=("Segoe UI", 10, "bold"),
            bg="#E11D48",
            fg="white",
            activebackground="#BE123C",
            activeforeground="white",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            command=self.exit_app,
        )
        btn_exit.pack(side=tk.RIGHT)

        self.img_card = tk.Frame(
            self.window, bg="#1E293B", bd=1, relief=tk.SOLID
        )
        self.img_card.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.lbl_image = tk.Label(self.img_card, bg="#0F172A", bd=0)
        self.lbl_image.pack(expand=True, anchor="center", padx=10, pady=10)

        self.render_image(self.stitched_img)

    def render_image(self, img_array):
        if len(img_array.shape) == 2:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((760, 550))
        self.photo = ImageTk.PhotoImage(pil_img)

        self.lbl_image.config(image=self.photo)

    def trigger_inpainting(self):
        if self.inpainted_img is None:
            self.title_lbl.config(
                text="Processing Inpainting...", fg="#FBBF24"
            )
            self.window.update()

            self.inpainted_img = apply_inpainting_transformation(
                self.stitched_img
            )

        self.current_display_img = self.inpainted_img
        self.render_image(self.inpainted_img)
        self.title_lbl.config(
            text="Stitched Image with Inpainting Applied", fg="#10B981"
        )

        self.btn_inpaint.config(
            state=tk.DISABLED, bg="#475569", text="✓ Inpainting Applied"
        )

    def save_image(self):
        if self.current_display_img is None:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("All Files", "*.*"),
            ],
            title="Save Output Image",
        )
        if file_path:
            cv2.imwrite(file_path, self.current_display_img)
            messagebox.showinfo(
                "Saved", f"File successfully saved to:\n{file_path}"
            )

    def exit_app(self):
        self.window.destroy()
        self.main_root.destroy()
        sys.exit()


if __name__ == "__main__":
    app = ApplicationWindow()
    app.run()