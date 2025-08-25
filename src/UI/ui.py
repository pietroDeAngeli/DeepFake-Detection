import sys
import os
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QUrl, QTimer
from PyQt6.QtGui import QFont, QDragEnterEvent, QDropEvent, QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QStackedWidget, QMessageBox,
    QFrame
)

import tools.tools as tools


# This will run the pipeline in a separate thread
class PipelineWorker(QObject):
    finished = pyqtSignal(object)
    failed   = pyqtSignal(str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run_real_pipeline(self) -> tools.Result:
        models_dir = "../../models"
        temp_dir = "../../temp"
        return tools.pipeline(models_dir=models_dir, temp_dir=temp_dir, video_path=self.file_path)

    def run(self):
        try:
            if not os.path.isfile(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
            result = self.run_real_pipeline()
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))

# UI pages
# -----------------------------

class UploadPage(QWidget):
    fileSelected = pyqtSignal(str)
    detectClicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.file_path = None

        title = QLabel("DeepFake - Detection tool")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))

        drop_label = QLabel("Select a File\n\nor Drag and drop")
        drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_label.setStyleSheet(
            "QLabel {border: 2px dashed #888; border-radius: 10px; padding: 30px; color: #666;}"
        )

        self.path_label = QLabel("No file selected")
        self.path_label.setStyleSheet("color: #555;")
        self.path_label.setWordWrap(True)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.on_browse)

        detect_btn = QPushButton("Detect")
        detect_btn.setEnabled(False)
        detect_btn.clicked.connect(lambda: self.detectClicked.emit())
        self.detect_btn = detect_btn

        top = QVBoxLayout()
        top.addWidget(title)
        top.addSpacing(8)
        top.addWidget(drop_label)

        row = QHBoxLayout()
        row.addWidget(self.path_label, stretch=1)
        row.addWidget(browse_btn)
        top.addLayout(row)
        top.addSpacing(8)
        top.addWidget(detect_btn, alignment=Qt.AlignmentFlag.AlignRight)

        self.setLayout(top)

    def set_file(self, path: str):
        self.file_path = path
        self.path_label.setText(path)
        self.detect_btn.setEnabled(True)
        self.fileSelected.emit(path)

    def on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a video",
            "", "Videos (*.mp4 *.avi *.mov *.mkv);;All files (*.*)"
        )
        if path:
            self.set_file(path)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e: QDropEvent):
        urls = e.mimeData().urls()
        if not urls:
            return
        local = urls[0].toLocalFile()
        if local:
            self.set_file(local)


class ProgressPage(QWidget):
    def __init__(self):
        super().__init__()

        title = QLabel("DeepFake - Detection tool")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))

        # Label animata "Loading..."
        self.loading_label = QLabel("Loading   ")  # 3 spazi riservati
        self.loading_label.setFont(QFont("Arial", 16))
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Timer per animazione puntini
        self._dot_count = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_loading_text)
        self.timer.start(500)  # aggiorna ogni 500ms

        col = QVBoxLayout()
        col.addWidget(title)
        col.addSpacing(20)
        col.addWidget(self.loading_label, alignment=Qt.AlignmentFlag.AlignCenter)
        col.addSpacing(20)
        self.setLayout(col)

    def update_loading_text(self):
        self._dot_count = (self._dot_count + 1) % 4
        dots = "." * self._dot_count
        spaces = " " * (3 - self._dot_count)
        self.loading_label.setText(f"Loading{dots}{spaces}")

    def reset(self):
        self._dot_count = 0
        self.timer.start()

    def stop(self):
        self.timer.stop()
        self.loading_label.setText("Done   ")


class ResultPage(QWidget):
    goBack = pyqtSignal()

    def __init__(self, temp_dir="../../temp"):
        super().__init__()
        self.temp_dir = os.path.abspath(temp_dir)

        title = QLabel("DeepFake - Detection tool")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))

        self.face_time = QLabel("Face extraction: - sec")
        self.mv_time = QLabel("Motion vector extr.: - sec")
        self.cls_time = QLabel("Classification: - sec")

        self.pred = QLabel("PREDICTION: -")
        self.pred.setFont(QFont("Arial", 14, QFont.Weight.Bold))

        back = QPushButton("Go Back")
        back.clicked.connect(lambda: self.goBack.emit())

        open_temp = QPushButton("Open Temp Folder")
        open_temp.clicked.connect(self.open_temp_folder)

        times_layout = QVBoxLayout()
        times_layout.addWidget(self.face_time)
        times_layout.addWidget(self.mv_time)
        times_layout.addWidget(self.cls_time)

        box = QFrame()
        box.setFrameShape(QFrame.Shape.StyledPanel)
        box.setLayout(times_layout)

        v = QVBoxLayout()
        v.addWidget(title)
        v.addSpacing(8)
        v.addWidget(box)
        v.addSpacing(12)
        v.addWidget(self.pred)
        v.addStretch(1)

        btns = QHBoxLayout()
        btns.addWidget(open_temp)
        btns.addWidget(back)
        v.addLayout(btns)

        self.setLayout(v)

    def set_result(self, res: tools.Result):
        self.face_time.setText(f"Face extraction: {res.face_time_s:.2f} sec")
        self.mv_time.setText(f"Motion vector extr.: {res.mv_time_s:.2f} sec")
        self.cls_time.setText(f"Classification: {res.cls_time_s:.2f} sec")
        self.pred.setText(
            f"PREDICTION: {res.label_str} {res.prob_real*100:.1f}%" if res.label_str == "REAL"
            else f"PREDICTION: {res.label_str} {(1-res.prob_real)*100:.1f}%"
        )

    def open_temp_folder(self):
        if os.path.isdir(self.temp_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.temp_dir))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepFake Detection tool")
        self.resize(700, 520)

        self.setStyleSheet("""
            QWidget { font-size: 14px; }
            QPushButton { padding: 8px 16px; border-radius: 8px; }
        """)

        self.stack = QStackedWidget()
        self.upload_page = UploadPage()
        self.progress_page = ProgressPage()
        self.result_page = ResultPage()

        self.stack.addWidget(self.upload_page)
        self.stack.addWidget(self.progress_page)
        self.stack.addWidget(self.result_page)
        self.setCentralWidget(self.stack)

        self.selected_file = None
        self.thread = None
        self.worker = None

        self.upload_page.fileSelected.connect(self.on_file_selected)
        self.upload_page.detectClicked.connect(self.on_detect)
        self.result_page.goBack.connect(self.on_back)

    def on_file_selected(self, path: str):
        self.selected_file = path

    def on_detect(self):
        if not self.selected_file:
            QMessageBox.warning(self, "No file", "Please select a file first.")
            return

        self.progress_page.reset()
        self.stack.setCurrentIndex(1)

        self.thread = QThread()
        self.worker = PipelineWorker(self.selected_file)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.failed.connect(self.thread.quit)
        self.worker.failed.connect(self.worker.deleteLater)

        self.thread.start()

    def on_finished(self, res: tools.Result):
        self.progress_page.stop()
        self.result_page.set_result(res)
        self.stack.setCurrentIndex(2)

    def on_failed(self, msg: str):
        self.progress_page.stop()
        QMessageBox.critical(self, "Error", msg)
        self.stack.setCurrentIndex(0)

    def on_back(self):
        self.stack.setCurrentIndex(0)


# Entry point
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
