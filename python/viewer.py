import sys
import json
import struct
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTreeWidget, 
                            QTreeWidgetItem, QSplitter, QTextEdit, 
                            QFileDialog, QVBoxLayout, QWidget, 
                            QHeaderView)
from PyQt5.QtCore import Qt

class SafetensorsViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Safetensors Viewer")
        self.setGeometry(100, 100, 800, 600)
        
        # 创建主部件
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        # 布局
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)
        
        # 创建分割器
        self.splitter = QSplitter(Qt.Vertical)
        self.layout.addWidget(self.splitter)
        
        # 创建树形视图
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type", "Shape", "Size"])
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tree.itemClicked.connect(self.on_item_clicked)
        
        # 创建文本视图
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        
        # 添加部件到分割器
        self.splitter.addWidget(self.tree)
        self.splitter.addWidget(self.text_view)
        
        # 创建菜单栏
        self.create_menu()
        
        # 初始化变量
        self.file_data = {}
        self.file_path = ""
    
    def create_menu(self):
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("File")
        open_action = file_menu.addAction("Open")
        open_action.triggered.connect(self.open_file)
        
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
    
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Safetensors File", "", "Safetensors Files (*.safetensors)"
        )
        
        if file_path:
            self.file_path = file_path
            self.load_file(file_path)
    
    def load_file(self, file_path):
        self.tree.clear()
        self.text_view.clear()
        
        try:
            with open(file_path, "rb") as f:
                # 读取头部长度 (8字节)
                length_bytes = f.read(8)
                header_length = struct.unpack("<Q", length_bytes)[0]
                
                # 读取头部JSON
                header_bytes = f.read(header_length)
                header = json.loads(header_bytes.decode("utf-8"))
                
                # 存储元数据
                metadata = header.get("__metadata__", {})
                self.file_data = {"metadata": metadata, "tensors": {}}
                
                # 添加元数据到树形视图
                metadata_item = QTreeWidgetItem(["Metadata", "dict", "", ""])
                self.tree.addTopLevelItem(metadata_item)
                
                for key, value in metadata.items():
                    metadata_item.addChild(QTreeWidgetItem([key, str(type(value)), "", str(value)]))
                
                # 添加张量信息
                tensors_item = QTreeWidgetItem(["Tensors", "", "", ""])
                self.tree.addTopLevelItem(tensors_item)
                
                for key, tensor_info in header.items():
                    if key != "__metadata__":
                        dtype = tensor_info["dtype"]
                        shape = tensor_info["shape"]
                        data_offsets = tensor_info["data_offsets"]
                        size = data_offsets[1] - data_offsets[0]
                        
                        tensor_item = QTreeWidgetItem([
                            key, 
                            dtype, 
                            str(shape), 
                            f"{size} bytes"
                        ])
                        tensors_item.addChild(tensor_item)
                        
                        # 存储张量信息
                        self.file_data["tensors"][key] = {
                            "dtype": dtype,
                            "shape": shape,
                            "offsets": data_offsets
                        }
                
                # 展开所有项
                self.tree.expandAll()
                
        except Exception as e:
            self.text_view.setText(f"Error loading file: {str(e)}")
    
    def on_item_clicked(self, item, column):
        if item.parent() is not None and item.parent().text(0) == "Tensors":
            tensor_name = item.text(0)
            tensor_info = self.file_data["tensors"].get(tensor_name)
            
            if tensor_info:
                info_text = f"Tensor: {tensor_name}\n"
                info_text += f"Type: {tensor_info['dtype']}\n"
                info_text += f"Shape: {tensor_info['shape']}\n"
                info_text += f"Data offsets: {tensor_info['offsets']}\n"
                
                # 尝试加载实际张量数据 (仅显示小张量的内容)
                if np.prod(tensor_info["shape"]) <= 100:  # 只显示小张量
                    try:
                        from safetensors import safe_open
                        with safe_open(self.file_path, framework="pt") as f:
                            tensor = f.get_tensor(tensor_name)
                            info_text += f"\nSample data:\n{tensor}"
                    except:
                        info_text += "\n(Unable to load full tensor data)"
                
                self.text_view.setText(info_text)
        elif item.parent() is not None and item.parent().text(0) == "Metadata":
            self.text_view.setText(f"Metadata: {item.text(0)} = {item.text(3)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = SafetensorsViewer()
    viewer.show()
    sys.exit(app.exec_())