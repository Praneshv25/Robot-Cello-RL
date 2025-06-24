import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


def load_rl_env_from_dot_env():
    """
    Searches the parent directory for .env file and call load_env_file with it.

    Expected to used in another script as following:
    ```python
        from rl_env_manager import load_rl_env_from_dot_env

        # Load environment variables
        env_vars = load_rl_env_from_dot_env()

        trajectory = extract_joint_angles(
            env_vars.get("TRAJ_LOG_CSV_PATH")
        )
    ```

    :return: dictionary of parsed environment variable
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dot_env_path = os.path.abspath(os.path.join(script_dir, '..', '.env'))
    return load_env_file(dot_env_path)


def load_env_file(filepath):
    """
    Internal helper for parsing the `.env` file in the project root directory
    and creating a dictionary to access them.

    :return: dictionary of parsed environment variable
    :raises ValueError: if any of the environment variable is empty
    """

    env_dict = {}
    if not os.path.isfile(filepath):
        return env_dict

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            env_dict[key] = value
    return env_dict


class EnvVarManager:
    def __init__(self, root, env_dict, env_path):
        self.root = root
        self.env_dict = env_dict
        self.env_path = env_path
        self.selected_var = None

        self.root.title("Robot Cello Residual RL Environment Variable Manager")

        self.main_frame = tk.Frame(root, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, sticky="ns")

        tk.Label(self.left_frame, text="Variables", font=(
            "Arial", 10, "bold")).pack(anchor="w")

        self.listbox = tk.Listbox(self.left_frame, width=30)
        self.listbox.pack(fill=tk.Y, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

        self.save_button = tk.Button(
            self.left_frame, text="Save to File",
            command=self.save_to_env_file)
        self.save_button.pack(anchor="w", pady=(10, 0))

        self.add_button = tk.Button(
            self.left_frame, text="Add Variable",
            command=self.add_variable)
        self.add_button.pack(anchor="w", pady=(5, 0))

        self.delete_button = tk.Button(
            self.left_frame, text="Delete Variable",
            command=self.delete_variable)
        self.delete_button.pack(anchor="w", pady=(5, 0))

        # Right panel
        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.grid(row=0, column=1, padx=20, sticky="nsew")
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        # Frame for wrapping label
        self.label_frame = tk.Frame(self.right_frame)
        self.label_frame.pack(fill=tk.BOTH, expand=True, anchor="nw")

        self.value_label = tk.Label(
            self.label_frame, text="Select a variable", anchor="w",
            justify="left"
        )
        self.value_label.pack(fill=tk.BOTH, expand=True)

        # Two buttons: Set manually and select file
        self.button_frame = tk.Frame(self.right_frame)
        self.button_frame.pack(anchor="w", pady=(0, 10))

        self.manual_button = tk.Button(
            self.button_frame, text="Set Manually", command=self.set_manual,
            state=tk.DISABLED)
        self.manual_button.grid(row=0, column=0, padx=(0, 10))

        self.file_button = tk.Button(
            self.button_frame, text="Select File", command=self.select_file,
            state=tk.DISABLED)
        self.file_button.grid(row=0, column=1)

        self.populate_listbox()

        # Handle resizing for label wrapping
        self.label_frame.bind("<Configure>", self.update_wraplength)

    def populate_listbox(self):
        self.listbox.delete(0, tk.END)
        for var in self.env_dict:
            self.listbox.insert(tk.END, var)

    def on_select(self, event):
        selection = event.widget.curselection()
        if not selection:
            return
        index = selection[0]
        var_name = self.listbox.get(index)
        self.selected_var = var_name
        value = self.env_dict[var_name]
        self.value_label.config(text=f"{var_name} = {value}")
        self.manual_button.config(state=tk.NORMAL)
        self.file_button.config(state=tk.NORMAL)

    def set_manual(self):
        if not self.selected_var:
            return
        new_value = simpledialog.askstring("Set Value",
                                           f"Enter value for {
                                               self.selected_var}:",
                                           initialvalue=self.env_dict[
                                               self.selected_var
                                           ])
        if new_value is not None:
            self.env_dict[self.selected_var] = new_value
            self.value_label.config(text=f"{self.selected_var} = {new_value}")
            messagebox.showinfo(
                "Updated", f"{self.selected_var} updated to:\n{new_value}.")

    def select_file(self):
        if not self.selected_var:
            return
        new_path = filedialog.askopenfilename()
        if new_path:
            self.env_dict[self.selected_var] = new_path
            self.value_label.config(text=f"{self.selected_var} = {new_path}")
            messagebox.showinfo(
                "Updated", f"{self.selected_var} updated to:\n{new_path}")

    def save_to_env_file(self):
        try:
            with open(self.env_path, 'w') as f:
                for key, value in self.env_dict.items():
                    f.write(f'{key}="{value}"\n')
            messagebox.showinfo(
                "Saved", f"Environment variables saved to:\n{self.env_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def add_variable(self):
        key = simpledialog.askstring("Add Variable", "Enter variable name:")
        if not key:
            return
        if key in self.env_dict:
            messagebox.showerror("Error", f"Variable '{key}' already exists.")
            return
        value = simpledialog.askstring("Set Value", f"Enter value for {key}:")
        if value is not None:
            self.env_dict[key] = value
            self.populate_listbox()
            messagebox.showinfo("Added", f"Variable '{key}' added.")

    def delete_variable(self):
        if not self.selected_var:
            messagebox.showwarning(
                "No Selection", "No variable selected to delete.")
            return
        confirm = messagebox.askyesno(
            "Confirm Delete", f"Delete variable '{self.selected_var}'?")
        if confirm:
            del self.env_dict[self.selected_var]
            self.selected_var = None
            self.populate_listbox()
            self.value_label.config(text="Select a variable")
            self.manual_button.config(state=tk.DISABLED)
            self.file_button.config(state=tk.DISABLED)

    def update_wraplength(self, event):
        self.value_label.config(wraplength=event.width - 20)


if __name__ == "__main__":
    env_vars = load_rl_env_from_dot_env()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.abspath(os.path.join(script_dir, '..', '.env'))
    root = tk.Tk()
    app = EnvVarManager(root, env_vars, env_path)
    root.mainloop()
