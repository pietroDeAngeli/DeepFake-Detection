from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QPushButton, QLabel, QFileDialog
)
from PySide6.QtCore import Qt
import sys

class FileLoaderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepFake Detection - File Loader")
        self.resize(500, 350)
        self.setAcceptDrops(True)

        self.current_file = None

        # Widget centrale e layout
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        self.setCentralWidget(central)

        # Label drag&drop
        self.label = QLabel("Drag the file here\nOr click on Browse", alignment=Qt.AlignCenter)
        self.label.setFixedHeight(100)
        self.label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                border: 2px dashed #888;
                border-radius: 8px;
            }
        """)
        layout.addWidget(self.label)

        # Pulsante Browse
        btn = QPushButton("Browse")
        btn.setFixedSize(120, 40)
        btn.clicked.connect(self.open_file_dialog)
        layout.addWidget(btn, alignment=Qt.AlignCenter)

        # Add the Predict button (hidden until file is loaded)
        self.predict_button = QPushButton("Predict")
        self.predict_button.setFixedSize(120, 40)
        # Dull green background with slight transparency
        self.predict_button.setStyleSheet(
            "background-color: rgba(40, 167, 69, 0.8); color: white; border-radius: 5px;"
        )
        self.predict_button.hide()
        self.predict_button.clicked.connect(lambda: self.predict(self.current_file))
        layout.addWidget(self.predict_button, alignment=Qt.AlignCenter)

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select a file")
        if path:
            self.show_selected(path, method="Selected")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.show_selected(path, method="Uploaded")

    def show_selected(self, path, method):
        # Salva il percorso, mostra il Predict button
        self.current_file = path
        self.predict_button.show()
        # Mostra solo il nome del file
        filename = path.split('/')[-1]
        self.label.setText(f"{method}:\n{filename}")

    def predict(self, path):
        # TODO: implementa la funzione di predizione DeepFake
        print()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FileLoaderWindow()
    win.show()
    sys.exit(app.exec())
