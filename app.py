"""
3DGS 训练工作室 — FastAPI 本地 Web 桌面应用
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import shutil
import threading
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ─── 配置 ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train.py")
VIEWER_HTML = os.path.join(SCRIPT_DIR, "viewer.html")
DEFAULT_STEPS = 30_000

app = FastAPI(title="3DGS 训练工作室")

# ─── 运行状态 ────────────────────────────────────────────────────────
class TrainingState:
    def __init__(self):
        self.process = None
        self.running = False
        self.cancelled = False
        self.step = 0
        self.total_steps = DEFAULT_STEPS
        self.loss = 0.0
        self.pts = 0
        self.phase = ""  # colmap, training, exporting, done, error
        self.images_dir = ""
        self.output_dir = ""
        self.error_msg = ""
        self.ws_clients = set()

state = TrainingState()

# ─── WebSocket ───────────────────────────────────────────────────────
async def broadcast(msg: dict):
    """向所有连接的 WebSocket 客户端广播消息。"""
    dead = set()
    for ws in state.ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    state.ws_clients -= dead

async def training_worker(images_dir: str, output_dir: str, steps: int):
    """在后台线程中运行训练，通过 broadcast 发送进度。"""
    state.running = True
    state.cancelled = False
    state.images_dir = images_dir
    state.output_dir = output_dir
    state.step = 0
    state.total_steps = steps
    state.loss = 0.0
    state.pts = 0
    state.error_msg = ""

    try:
        env = os.environ.copy()
        env["TQDM_DISABLE"] = "0"
        env["PYTHONUNBUFFERED"] = "1"

        # 自动查找 COLMAP
        colmap_candidates = [
            os.path.join(SCRIPT_DIR, "colmap", "bin", "colmap.exe"),
        ]
        for c in colmap_candidates:
            if os.path.exists(c):
                env["COLMAP_EXE"] = c
                break

        state.phase = "colmap"
        await broadcast({"type": "phase", "phase": "colmap", "message": "COLMAP 特征提取与匹配中...如果照片分辨率很高，这一步可能需要 5-30 分钟"})

        proc = subprocess.Popen(
            [sys.executable, TRAIN_SCRIPT, images_dir, "--output", output_dir, "--steps", str(steps)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, bufsize=1,
        )
        state.process = proc

        # 解析输出行
        tqdm_pattern = re.compile(
            r"训练 3DGS:\s*(?P<pct>\d+)%\|.*?\|\s*(?P<step>\d+)/(?P<total>\d+)"
            r"\[.*?,\s*(?P<loss>[\d.]+),.*?pts=(?P<pts>\d+)\]"
        )
        colmap_pattern = re.compile(r"\[COLMAP\]")
        phase_pattern = re.compile(r"\[(COLMAP|3DGS|导出|初始化|数据)\]")

        for line in proc.stdout:
            if state.cancelled:
                proc.terminate()
                break

            line = line.rstrip()
            if not line:
                continue

            # 检测阶段
            phase_m = phase_pattern.search(line)
            if phase_m:
                p = phase_m.group(1)
                if p == "COLMAP":
                    state.phase = "colmap"
                elif p in ("3DGS", "训练", "初始化"):
                    state.phase = "training"
                elif p == "导出":
                    state.phase = "exporting"

            # 解析 tqdm 进度
            tqdm_m = tqdm_pattern.search(line)
            if tqdm_m:
                state.step = int(tqdm_m.group("step"))
                state.total_steps = int(tqdm_m.group("total"))
                state.loss = float(tqdm_m.group("loss"))
                state.pts = int(tqdm_m.group("pts"))
                await broadcast({
                    "type": "progress",
                    "step": state.step,
                    "total": state.total_steps,
                    "loss": state.loss,
                    "pts": state.pts,
                    "pct": int(tqdm_m.group("pct")),
                })
            else:
                # 其他日志行
                await broadcast({"type": "log", "text": line})

            # 检测完成
            if "全部完成" in line or "SPLAT 保存" in line:
                state.phase = "done"

        proc.wait()
        state.running = False

        if state.cancelled:
            await broadcast({"type": "cancelled"})
            return

        if proc.returncode != 0:
            state.phase = "error"
            state.error_msg = f"训练进程退出码: {proc.returncode}"
            await broadcast({"type": "error", "message": state.error_msg})
            return

        # 找输出文件
        splat_path = os.path.join(output_dir, "model.splat")
        if os.path.exists(splat_path):
            state.phase = "done"
            await broadcast({
                "type": "done",
                "splat": splat_path,
                "output_dir": output_dir,
                "message": "训练完成！",
            })
        else:
            state.phase = "error"
            state.error_msg = "未找到输出 model.splat 文件"
            await broadcast({"type": "error", "message": state.error_msg})

    except Exception as e:
        state.running = False
        state.phase = "error"
        state.error_msg = str(e)
        await broadcast({"type": "error", "message": str(e)})
    finally:
        state.process = None
        state.running = False


# ─── 路由 ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(SCRIPT_DIR, "templates", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/viewer")
async def viewer(splat: str = ""):
    """返回 viewer.html，可指定 splat 文件路径。"""
    with open(VIEWER_HTML, "r", encoding="utf-8") as f:
        html = f.read()
    if splat:
        # 注入 splat 路径到 URL 参数
        html = html.replace("</body>",
            f'<script>window.BUILTIN_SPLAT="{splat}";</script></body>')
    return HTMLResponse(html)

@app.post("/api/start")
async def start_training(data: dict):
    """开始训练。"""
    if state.running:
        return {"error": "已有训练任务运行中"}

    images_dir = data.get("images_dir", "")
    steps = data.get("steps", DEFAULT_STEPS)

    if not images_dir or not os.path.isdir(images_dir):
        return {"error": "照片文件夹无效"}

    # 创建输出目录
    output_dir = os.path.join(SCRIPT_DIR, "output", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    # 启动后台训练
    thread = threading.Thread(
        target=lambda: asyncio.run(training_worker(images_dir, output_dir, steps)),
        daemon=True,
    )
    thread.start()

    return {"success": True, "output_dir": output_dir}

@app.post("/api/cancel")
async def cancel_training():
    """取消训练。"""
    if state.process and state.running:
        state.cancelled = True
        state.process.terminate()
        return {"success": True}
    return {"error": "没有运行中的训练任务"}

@app.get("/api/status")
async def get_status():
    """获取当前状态。"""
    return {
        "running": state.running,
        "phase": state.phase,
        "step": state.step,
        "total": state.total_steps,
        "loss": state.loss,
        "pts": state.pts,
        "error": state.error_msg,
        "cancelled": state.cancelled,
    }

# ─── WebSocket ───────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        state.ws_clients.discard(ws)

# ─── 静态文件 ────────────────────────────────────────────────────────
static_dir = os.path.join(SCRIPT_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ─── 入口 ────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  3DGS 训练工作室")
    print("=" * 50)
    print()
    print(f"  打开浏览器访问: http://localhost:8080")
    print(f"  按 Ctrl+C 停止服务器")
    print()
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")


if __name__ == "__main__":
    main()
