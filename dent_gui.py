import os, json
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np
import tensorflow as tf

# ---------------- CONFIG ----------------
MODEL_DIR = "models"
IMG_SIZE = (224, 224)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# ---------------- MODEL LOADING ----------------
def load_latest_model_and_mapping():
    models = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")])
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, models[-1]))

    with open(os.path.join(MODEL_DIR, "class_mapping.json")) as f:
        mapping = {int(k): v for k, v in json.load(f).items()}

    return model, mapping, models[-1]

# ---------------- APP ----------------
class DentApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Car Dent Detection System")

        # MAXIMIZE
        try:
            self.state("zoomed")
        except:
            self.attributes("-zoomed", True)

        self.minsize(1100, 700)

        self.model, self.mapping, self.model_name = load_latest_model_and_mapping()
        self.img_path = None

        self.build_ui()
        self.animate_header()

    # ---------------- UI ----------------
    def build_ui(self):
        # ===== SIDEBAR =====
        sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        sidebar.pack(side="left", fill="y")

        ctk.CTkLabel(
            sidebar, text="🚗 CAR DAMAGE",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=(30, 5))

        ctk.CTkLabel(
            sidebar, text="Detection System",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(pady=(0, 35))

        ctk.CTkButton(
            sidebar, text="📂 Load Image",
            height=44, command=self.load_image
        ).pack(pady=10, padx=25)

        ctk.CTkButton(
            sidebar, text="🔍 Predict Damage",
            height=44, fg_color="#2563EB",
            command=self.predict
        ).pack(pady=10, padx=25)

        ctk.CTkLabel(
            sidebar,
            text=f"Model Loaded:\n{self.model_name}",
            font=ctk.CTkFont(size=10),
            text_color="gray",
            wraplength=200,
            justify="center"
        ).pack(side="bottom", pady=20)

        # ===== MAIN =====
        self.main = ctk.CTkFrame(self, corner_radius=0)
        self.main.pack(side="right", fill="both", expand=True, padx=25, pady=25)

        # ===== HEADER (animated) =====
        self.header_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        self.header_frame.place(relx=0.5, y=0, anchor="n")

        self.title_lbl = ctk.CTkLabel(
            self.header_frame,
            text="Car Dent Identification",
            font=ctk.CTkFont(size=32, weight="bold")
        )
        self.title_lbl.pack()

        self.subtitle_lbl = ctk.CTkLabel(
            self.header_frame,
            text="Upload a vehicle image to detect dents using deep learning",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.subtitle_lbl.pack(pady=(4, 0))

        # ===== IMAGE CARD =====
        self.image_card = ctk.CTkFrame(self.main, corner_radius=18)
        self.image_card.pack(fill="both", expand=True, pady=(90, 10))

        self.image_label = ctk.CTkLabel(
            self.image_card,
            text="No Image Loaded",
            font=ctk.CTkFont(size=18)
        )
        self.image_label.pack(expand=True)

        # ===== PROGRESS =====
        self.progress = ctk.CTkProgressBar(self.main)
        self.progress.set(0)
        self.progress.pack(fill="x", pady=(18, 8))

        # ===== RESULT =====
        self.result_label = ctk.CTkLabel(
            self.main,
            text="Awaiting Prediction",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.result_label.pack(pady=12)

    # ---------------- HEADER ANIMATION ----------------
    def animate_header(self):
        self.header_y = -40

        def slide():
            if self.header_y < 20:
                self.header_y += 4
                self.header_frame.place(relx=0.5, y=self.header_y, anchor="n")
                self.after(16, slide)

        slide()

    # ---------------- LOGIC ----------------
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")]
        )
        if not path:
            return

        self.img_path = path
        img = Image.open(path).convert("RGB")
        img.thumbnail((900, 500))

        ctk_img = ctk.CTkImage(img, size=img.size)
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img

        self.result_label.configure(text="Image Loaded ✔️", text_color="white")
        self.progress.set(0)

    def predict(self):
        if not self.img_path:
            messagebox.showwarning("Error", "Please load an image first")
            return

        self.progress.set(0.3)
        self.update_idletasks()

        img = Image.open(self.img_path).convert("RGB").resize(IMG_SIZE)
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)

        self.progress.set(0.7)
        self.update_idletasks()

        pred = float(self.model.predict(arr)[0][0])
        dented_index = [k for k, v in self.mapping.items() if "dent" in v.lower()][0]
        prob_dented = pred if dented_index == 1 else (1 - pred)

        self.progress.set(1.0)

        if prob_dented >= 0.5:
            self.result_label.configure(
                text=f"❌ DENTED  |  {prob_dented*100:.2f}%",
                text_color="#EF4444"
            )
        else:
            self.result_label.configure(
                text=f"✅ NOT DENTED  |  {(1-prob_dented)*100:.2f}%",
                text_color="#22C55E"
            )
            

# ---------------- RUN ----------------
if __name__ == "__main__":
    app = DentApp()
    app.mainloop()
