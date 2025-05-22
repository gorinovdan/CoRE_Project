import os
import uuid
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from PIL import Image
import torch
import timm
import numpy as np
from sqlalchemy import create_engine, Column, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import shap
import matplotlib.pyplot as plt

# === Local Paths ===
UPLOAD_DIR = os.path.abspath("uploads")
RESULTS_DIR = os.path.abspath("results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Database ===
Base = declarative_base()
engine = create_engine("sqlite:///db.sqlite3")
SessionLocal = sessionmaker(bind=engine)

class ImageEntry(Base):
    __tablename__ = "images"
    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    path = Column(String)
    rule_center = Column(Float)
    rule_curved = Column(Float)
    rule_diagonal = Column(Float)
    rule_fill_the_frame = Column(Float)
    rule_pattern = Column(Float)
    rule_rule_of_thirds = Column(Float)
    rule_symmetric = Column(Float)
    rule_triangle = Column(Float)
    rule_vanishing_point = Column(Float)
    rule_golden_ratio = Column(Float)
    rule_horizontal = Column(Float)
    rule_radial = Column(Float)
    rule_vertical = Column(Float)
    gradcam_path = Column(String, nullable=True)
    shap_path = Column(String, nullable=True)

Base.metadata.create_all(engine)

# === FastAPI init ===
app = FastAPI(title="Image Composition API (Local)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === MODEL ===
COMPOSITION_RULES = [
    "center", "curved", "diagonal", "fill_the_frame", "pattern",
    "rule_of_thirds", "symmetric", "triangle", "vanishing_point",
    "golden_ratio", "horizontal", "radial", "vertical"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

class ViTModel(torch.nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Remove prefix if checkpoint from Lightning
        if any(k.startswith("backbone.") for k in state_dict.keys()):
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        self.backbone.load_state_dict(state_dict, strict=False)
        self.heads = torch.nn.ModuleList([torch.nn.Linear(self.backbone.num_features, 1) for _ in COMPOSITION_RULES])

    def forward(self, x):
        f = self.backbone(x)
        outs = [head(f).squeeze(1) for head in self.heads]
        return torch.stack(outs, dim=1)

MODEL_PATH = os.path.abspath("models/vit_b16.ckpt")
model = ViTModel(MODEL_PATH).to(device).eval()

# === Preprocess ===
from torchvision import transforms
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def prepare_image(filepath: str) -> torch.Tensor:
    image = Image.open(filepath).convert("RGB")
    return tf(image).unsqueeze(0)

def predict(image_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        preds = model(image_tensor.to(device)).cpu().numpy()[0]
    preds = np.clip(preds, 0, 1) * 10
    return preds

# === API Models ===
class Prediction(BaseModel):
    rule: str
    value: float

class ImageResult(BaseModel):
    id: str
    filename: str
    predictions: List[Prediction]

# === ROUTES ===

@app.post("/upload", response_model=List[ImageResult])
async def upload_images(files: List[UploadFile] = File(...)):
    results = []
    session = SessionLocal()
    for file in files:
        ext = os.path.splitext(file.filename)[1]
        image_id = str(uuid.uuid4())
        save_path = os.path.join(UPLOAD_DIR, f"{image_id}{ext}")
        with open(save_path, "wb") as f_out:
            f_out.write(await file.read())
        img_tensor = prepare_image(save_path)
        preds = predict(img_tensor)
        rule_results = dict(zip(COMPOSITION_RULES, preds))
        entry = ImageEntry(
            id=image_id,
            filename=file.filename,
            path=save_path,
            **{f"rule_{k}": float(v) for k, v in rule_results.items()}
        )
        session.add(entry)
        session.commit()
        results.append(ImageResult(
            id=image_id,
            filename=file.filename,
            predictions=[Prediction(rule=k, value=float(v)) for k, v in rule_results.items()]
        ))
    session.close()
    return results

@app.get("/results", response_model=List[ImageResult])
def get_results():
    session = SessionLocal()
    all_entries = session.query(ImageEntry).all()
    results = []
    for entry in all_entries:
        preds = [Prediction(rule=k, value=getattr(entry, f"rule_{k}")) for k in COMPOSITION_RULES]
        results.append(ImageResult(id=entry.id, filename=entry.filename, predictions=preds))
    session.close()
    return results

@app.get("/logs/export")
def export_csv():
    session = SessionLocal()
    all_entries = session.query(ImageEntry).all()
    data = []
    for e in all_entries:
        row = { "id": e.id, "filename": e.filename }
        for k in COMPOSITION_RULES:
            row[k] = getattr(e, f"rule_{k}")
        data.append(row)
    df = pd.DataFrame(data)
    out_csv = os.path.join(RESULTS_DIR, "results_export.csv")
    df.to_csv(out_csv, index=False)
    session.close()
    return FileResponse(out_csv, media_type='text/csv', filename="results_export.csv")

@app.get("/gradcam/{image_id}")
def gradcam(image_id: str, rule: int = 0):
    session = SessionLocal()
    entry = session.query(ImageEntry).filter(ImageEntry.id == image_id).first()
    session.close()
    if not entry:
        raise HTTPException(404, "Not found")
    img = Image.open(entry.path).convert("RGB")
    rgb = np.array(img.resize((224, 224))) / 255.0
    input_tensor = tf(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out_full = model(input_tensor)
        print(f"[GRADCAM] Model output (all rules): {out_full}")
        print(f"[GRADCAM] Output for rule {rule}: {out_full[:, rule].cpu().numpy()}")
        head_weight = model.heads[rule].weight.data
        head_bias = model.heads[rule].bias.data
        print(f"[GRADCAM] Head #{rule} weights (mean/std/min/max):",
              head_weight.mean().item(), head_weight.std().item(), head_weight.min().item(), head_weight.max().item())
        print(f"[GRADCAM] Head #{rule} bias:", head_bias.item())

    class Wrapper(torch.nn.Module):
        def __init__(self, model, rule_idx):
            super().__init__()
            self.model = model
            self.rule_idx = rule_idx
        def forward(self, x):
            out = self.model(x)
            print(f"[GRADCAM] Wrapper output shape: {out.shape}")
            return out[:, self.rule_idx].unsqueeze(1)

    wrapper = Wrapper(model, rule)
    target_layers = [model.backbone.blocks[-1]]
    def reshape_transform(tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        return result.permute(0, 3, 1, 2)
    cam = GradCAM(wrapper, target_layers=target_layers, reshape_transform=reshape_transform)
    grayscale = cam(input_tensor=input_tensor)[0]
    print(f"[GRADCAM] CAM min/max: {np.min(grayscale):.5f}, {np.max(grayscale):.5f}")
    heatmap = show_cam_on_image(rgb, grayscale, use_rgb=True)
    out_path = os.path.join(UPLOAD_DIR, f"{image_id}_gradcam_{rule}.jpg")
    Image.fromarray(heatmap).save(out_path)
    return FileResponse(out_path, media_type="image/jpeg")

@app.get("/shap/{image_id}")
def shap_vis(image_id: str, rule: int = 0):
    session = SessionLocal()
    entry = session.query(ImageEntry).filter(ImageEntry.id == image_id).first()
    session.close()
    if not entry:
        raise HTTPException(404, "Not found")
    img = Image.open(entry.path).convert("RGB").resize((224,224))
    img_tensor = tf(img).unsqueeze(0).to(device)
    background = img_tensor.repeat(16, 1, 1, 1)
    class Wrapper(torch.nn.Module):
        def __init__(self, model, rule_idx):
            super().__init__()
            self.model = model
            self.rule_idx = rule_idx
        def forward(self, x):
            out = self.model(x)
            # Исправление! detach() перед numpy()
            print(f"[SHAP] Wrapper output shape: {out.shape}")
            print(f"[SHAP] Output for rule {self.rule_idx}: {out[:, self.rule_idx].detach().cpu().numpy()}")
            return out[:, self.rule_idx].unsqueeze(1)
    wrapper = Wrapper(model, rule)
    print("[SHAP] Background tensor shape:", background.shape)
    print("[SHAP] Input tensor shape:", img_tensor.shape)
    e = shap.DeepExplainer(wrapper, background)
    shap_vals = e.shap_values(img_tensor)
    print("[SHAP] shap_vals type:", type(shap_vals))
    print("[SHAP] shap_vals shape:", getattr(shap_vals, "shape", "no shape"))
    plt.clf()
    img_np = np.array(img)
    if img_np.shape == (224, 224, 3):
        pass
    elif img_np.shape == (3, 224, 224):
        img_np = np.transpose(img_np, (1, 2, 0))
    else:
        raise ValueError(f"Unexpected image shape: {img_np.shape}")
    img_np = img_np.reshape(1, 224, 224, 3)
    # Приведение SHAP-значений к (1, 224, 224, 3)
    if isinstance(shap_vals, list):
        shap_arr = shap_vals[0]
    else:
        shap_arr = shap_vals
    print("[SHAP] shap_arr shape before:", shap_arr.shape)
    if shap_arr.shape == (1, 3, 224, 224, 1):
        shap_arr = np.squeeze(shap_arr, axis=-1)        # (1, 3, 224, 224)
        shap_arr = np.transpose(shap_arr, (0, 2, 3, 1)) # (1, 224, 224, 3)
    elif shap_arr.shape == (1, 224, 224, 3):
        pass
    else:
        raise ValueError(f"Unexpected shap_arr shape: {shap_arr.shape}")
    print("[SHAP] shap_arr shape after:", shap_arr.shape)
    print("[SHAP] SHAP min/max:", np.min(shap_arr), np.max(shap_arr))
    shap.image_plot([shap_arr], img_np, show=False)
    out_path = os.path.join(UPLOAD_DIR, f"{image_id}_shap_{rule}.jpg")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    return FileResponse(out_path, media_type="image/jpeg")


# === Static for images ===
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/")
def root():
    return {"message": "Welcome to Image Composition API (Local). See /docs for API usage."}

# === RUNNER ===
if __name__ == "__main__":
    import uvicorn
    print("Running server on http://127.0.0.1:8000 ...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
