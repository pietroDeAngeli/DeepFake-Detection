# main.py
# Launch the UI
import UI.ui as ui

if __name__ == "__main__":
    app = ui.QApplication([])
    window = ui.MainWindow()
    window.show()
    app.exec()