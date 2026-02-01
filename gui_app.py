import sys
import os
import cv2
import numpy as np
import threading
import gc
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QSlider, QFrame, QSizePolicy,
                             QGraphicsDropShadowEffect, QShortcut)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QKeySequence, QIcon, QFont, QCursor

# --- CONFIGURATION ---
INPUT_SIZE = 512

# --- STYLESHEET ---
STYLESHEET = """
QMainWindow { background-color: #1e1e2e; }
QWidget { font-family: "Segoe UI", sans-serif; color: #cdd6f4; }
QFrame#Sidebar { background-color: #181825; border-right: 1px solid #313244; }
QLabel#Title { font-size: 22px; font-weight: bold; color: #f5c2e7; padding: 10px; }
QLabel#Status { color: #a6adc8; font-size: 12px; }
QPushButton { background-color: #313244; border: none; border-radius: 8px; padding: 12px; font-size: 14px; font-weight: 600; color: #cdd6f4; }
QPushButton:hover { background-color: #45475a; }
QPushButton:pressed { background-color: #585b70; }
QPushButton#Primary { background-color: #89b4fa; color: #1e1e2e; }
QPushButton#Primary:hover { background-color: #b4befe; }
QPushButton#Primary:disabled { background-color: #45475a; color: #7f849c; }
QSlider::groove:horizontal { border: 1px solid #313244; height: 8px; background: #181825; border-radius: 4px; }
QSlider::handle:horizontal { background: #89b4fa; width: 18px; height: 18px; margin: -7px 0; border-radius: 9px; }
"""

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- AI WORKER (PURE OPENCV ONLY) ---
class AIWorker(QThread):
    status_signal = pyqtSignal(str, str)
    result_signal = pyqtSignal(np.ndarray, str)
    
    def __init__(self, lama_path):
        super().__init__()
        self.lama_path = lama_path
        self.net = None
        self.image = None
        self.manual_mask = None
        self.mode = "init"

    def run(self):
        if self.mode == "init":
            try:
                # Load LaMa using OpenCV directly
                self.net = cv2.dnn.readNet(self.lama_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.status_signal.emit("success", "AI Ready • Manual Mode")
            except Exception as e:
                self.status_signal.emit("error", f"Load Error: {str(e)}")
        
        elif self.mode == "process":
            try:
                if self.image is None or self.manual_mask is None: return
                
                h, w = self.image.shape[:2]
                
                # Check if mask is empty
                if cv2.countNonZero(self.manual_mask) == 0:
                    self.status_signal.emit("error", "Please paint over the watermark first!")
                    return

                # LAMA INPAINTING
                img_rez = cv2.resize(self.image, (INPUT_SIZE, INPUT_SIZE))
                mask_rez = cv2.resize(self.manual_mask, (INPUT_SIZE, INPUT_SIZE))
                
                # Preprocess
                img_blob = cv2.dnn.blobFromImage(img_rez, scalefactor=1/255.0, size=(512, 512), mean=(0,0,0), swapRB=False, crop=False)
                mask_blob = (mask_rez.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
                mask_blob = mask_blob[np.newaxis, np.newaxis, :, :] 

                self.net.setInput(img_blob, "image")
                self.net.setInput(mask_blob, "mask")
                
                output = self.net.forward()
                
                # Post-process
                output = output.squeeze().transpose(1, 2, 0)
                output = np.clip(output * 255, 0, 255).astype(np.uint8)
                result_final = cv2.resize(output, (w, h))
                
                # Blend
                mask_3ch = cv2.cvtColor(self.manual_mask, cv2.COLOR_GRAY2BGR)
                final = np.where(mask_3ch > 0, result_final, self.image)
                
                self.result_signal.emit(final, "Erased Successfully")
                gc.collect()
                
            except Exception as e:
                self.status_signal.emit("error", f"Error: {e}")

# --- PAINT CANVAS ---
class PaintCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.mask = None
        self.brush_size = 20
        self.last_point = QPoint()
        self.drawing = False
        self.history = []
        self.redo_stack = []
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)

    def set_image(self, cv_img):
        h, w, ch = cv_img.shape
        self.image = QImage(cv_img.data, w, h, ch * w, QImage.Format_BGR888)
        self.mask = QImage(w, h, QImage.Format_Grayscale8)
        self.mask.fill(Qt.black)
        self.history = [self.mask.copy()]
        self.redo_stack = []
        self.update()

    def get_manual_mask(self):
        if self.mask is None: return None
        ptr = self.mask.bits()
        ptr.setsize(self.mask.byteCount())
        return np.array(ptr).reshape(self.mask.height(), self.mask.width())

    def undo(self):
        if len(self.history) > 1:
            self.redo_stack.append(self.history.pop())
            self.mask = self.history[-1].copy()
            self.update()

    def redo(self):
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.history.append(state)
            self.mask = state.copy()
            self.update()

    def paintEvent(self, event):
        if self.image is None: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        ww, wh = self.width(), self.height()
        iw, ih = self.image.width(), self.image.height()
        self.scale_factor = min(ww/iw, wh/ih)
        nw, nh = int(iw * self.scale_factor), int(ih * self.scale_factor)
        self.offset = QPoint((ww - nw) // 2, (wh - nh) // 2)
        
        painter.drawImage(self.offset.x(), self.offset.y(), self.image.scaled(nw, nh, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        if self.mask:
            overlay = self.mask.scaled(nw, nh, Qt.KeepAspectRatio)
            bitmap = QPixmap.fromImage(overlay).createMaskFromColor(QColor(0,0,0), Qt.MaskOutColor)
            painter.setClipRegion(bitmap)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 0, 0, 100))
            painter.drawRect(self.offset.x(), self.offset.y(), nw, nh)

    def mousePressEvent(self, event):
        if self.image and event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = self.map_to_image(event.pos())
            self.draw_on_mask(self.last_point)

    def mouseMoveEvent(self, event):
        if self.drawing:
            pt = self.map_to_image(event.pos())
            self.draw_line(self.last_point, pt)
            self.last_point = pt

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            self.history.append(self.mask.copy())
            self.redo_stack.clear()

    def map_to_image(self, pos):
        return QPoint(int((pos.x() - self.offset.x()) / self.scale_factor), int((pos.y() - self.offset.y()) / self.scale_factor))

    def draw_on_mask(self, pt):
        p = QPainter(self.mask)
        p.setPen(Qt.NoPen); p.setBrush(Qt.white)
        p.drawEllipse(pt, self.brush_size, self.brush_size)
        self.update()

    def draw_line(self, p1, p2):
        p = QPainter(self.mask)
        p.setPen(QPen(Qt.white, self.brush_size*2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(p1, p2)
        self.update()

# --- MAIN WINDOW ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Eraser v9 (Lightweight)")
        self.resize(1200, 800)
        self.setStyleSheet(STYLESHEET)
        self.current_cv_img = None
        self.ai_thread = AIWorker(resource_path("lama.onnx"))
        self.ai_thread.status_signal.connect(self.update_status)
        self.ai_thread.result_signal.connect(self.on_result)
        self.ai_thread.start()
        self.setup_ui()

    def setup_ui(self):
        main = QWidget(); self.setCentralWidget(main)
        layout = QHBoxLayout(main); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)
        
        sidebar = QFrame(); sidebar.setObjectName("Sidebar"); sidebar.setFixedWidth(280)
        sb = QVBoxLayout(sidebar); sb.setContentsMargins(20,30,20,30); sb.setSpacing(15)
        
        sb.addWidget(QLabel("Magic Eraser", objectName="Title"))
        self.btn_open = QPushButton("📂 Open Image"); self.btn_open.clicked.connect(self.open_image)
        sb.addWidget(self.btn_open)
        
        sb.addWidget(QLabel("Brush Size"))
        slider = QSlider(Qt.Horizontal); slider.setRange(5,100); slider.setValue(20)
        slider.valueChanged.connect(lambda v: setattr(self.canvas, 'brush_size', v))
        sb.addWidget(slider)
        
        btns = QHBoxLayout()
        u = QPushButton("↩ Undo"); u.clicked.connect(lambda: self.canvas.undo())
        r = QPushButton("↪ Redo"); r.clicked.connect(lambda: self.canvas.redo())
        btns.addWidget(u); btns.addWidget(r); sb.addLayout(btns)
        
        sb.addStretch()
        self.btn_run = QPushButton("✨ Erase Paint", objectName="Primary"); self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_ai)
        sb.addWidget(self.btn_run)
        
        self.btn_save = QPushButton("💾 Save"); self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_image)
        sb.addWidget(self.btn_save)
        
        self.lbl_status = QLabel("Init Engine...", objectName="Status"); self.lbl_status.setAlignment(Qt.AlignCenter)
        sb.addWidget(self.lbl_status)
        layout.addWidget(sidebar)
        
        self.canvas = PaintCanvas(); layout.addWidget(self.canvas)
        
        QShortcut(QKeySequence("Ctrl+Z"), self, self.canvas.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.canvas.redo)

    def update_status(self, type, msg):
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet(f"color: {'#a6e3a1' if type=='success' else '#f38ba8'};")
        if "Ready" in msg and self.current_cv_img is not None: self.btn_run.setEnabled(True)

    def open_image(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open", "", "Images (*.png *.jpg)")
        if f:
            self.current_cv_img = cv2.imread(f)
            self.canvas.set_image(self.current_cv_img)
            self.btn_run.setEnabled(True); self.lbl_status.setText("Ready")

    def run_ai(self):
        self.btn_run.setEnabled(False); self.lbl_status.setText("Processing...")
        self.ai_thread.image = self.current_cv_img
        self.ai_thread.manual_mask = self.canvas.get_manual_mask()
        self.ai_thread.mode = "process"
        self.ai_thread.start()

    def on_result(self, img, msg):
        self.current_cv_img = img
        self.canvas.set_image(img)
        self.btn_run.setEnabled(True); self.btn_save.setEnabled(True)
        self.update_status("success", msg)

    def save_image(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save", "", "JPG (*.jpg)")
        if f: cv2.imwrite(f, self.current_cv_img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())