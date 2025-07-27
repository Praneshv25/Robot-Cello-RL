import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

reg = re.compile(r'(["\'])(\/[^\'"]+)\1')


def choose_file_cli():
    py_files = [f for f in os.listdir('.') if f.endswith('.py')]
    if not py_files:
        print("No Python files found in current directory.")
        exit(1)

    print("Select a Python file to scan:")
    for i, f in enumerate(py_files):
        print(f"{i + 1}. {f}")
    while True:
        choice = input("Enter number: ")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(py_files):
                return py_files[idx]
        except ValueError:
            pass
        print("Invalid.")


class PathEditor(tk.Tk):
    def __init__(self, file_path, paths):
        super().__init__()
        self.title(f"Editing: {os.path.basename(file_path)}")
        self.geometry("700x400")
        self.file_path = file_path
        # list of (line_index, start_pos, old_path)
        self.original_paths = paths
        self.updated_paths = [p[2] for p in paths]

        self.create_widgets()
        self.populate_paths()

    def create_widgets(self):
        self.tree = ttk.Treeview(self, columns=(
            "path", "action"), show="headings", height=15)
        self.tree.heading("path", text="Original Path")
        self.tree.heading("action", text="Change")
        self.tree.column("path", width=550)
        self.tree.column("action", width=100, anchor='center')
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(
            self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Save and Cancel buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        self.save_btn = tk.Button(
            btn_frame, text="Overwrite the file", command=self.save)
        self.save_btn.pack(side=tk.RIGHT, padx=10)

        self.cancel_btn = tk.Button(
            btn_frame, text="Cancel", command=self.destroy)
        self.cancel_btn.pack(side=tk.RIGHT)

        # Bind double-click on path cell to edit or open file dialog
        self.tree.bind('<Double-1>', self.on_double_click)

    def populate_paths(self):
        for idx, (_, _, path) in enumerate(self.original_paths):
            self.tree.insert("", "end", iid=str(
                idx), values=(path, "Double click to choose..."))

    def on_double_click(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if not row_id or column != '#1':  # Only allow edit on path column
            return

        old_path = self.tree.set(row_id, "path")

        # Open file picker dialog to choose replacement path
        new_path = filedialog.askopenfilename(
            title="Select replacement path",
            initialfile=os.path.basename(old_path),
            initialdir=os.path.dirname(
                old_path) if os.path.dirname(old_path) else ".",
        )
        if new_path:
            self.tree.set(row_id, "path", new_path)
            self.updated_paths[int(row_id)] = new_path

    def save(self):
        # Read file lines
        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        # For each path, replace old path with updated path in the corresponding line
        for i, (line_idx, _, old_path) in enumerate(self.original_paths):
            new_path = self.updated_paths[i]
            if new_path != old_path:
                # Replace all occurrences of old_path in the line (more precise: only the first occurrence)
                lines[line_idx] = lines[line_idx].replace(old_path, new_path)

        # Write back file
        with open(self.file_path, 'w') as f:
            f.writelines(lines)

        messagebox.showinfo("Success", f"Updated {self.file_path}")
        self.destroy()


def find_all_paths(file_path):
    paths = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            for match in reg.finditer(line):
                paths.append((idx, match.start(2), match.group(2)))
    return paths


def main():
    file_path = choose_file_cli()
    if not file_path:
        print("No file selected.")
        return
    paths = find_all_paths(file_path)
    if not paths:
        print("No UNIX paths found.")
        return

    app = PathEditor(file_path, paths)
    app.mainloop()


if __name__ == "__main__":
    main()
