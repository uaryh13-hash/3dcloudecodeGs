"""
3DGS 训练工作室 — FastAPI 本地 Web 桌面应用
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

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
        self.phase = ""
        self.images_dir = ""
        self.output_dir = ""
        self.error_msg = ""
        self.start_time = 0.0
        self.ws_clients = set()

state = TrainingState()

# ─── WebSocket 广播 ───────────────────────────────────────────────────
async def broadcast(msg: dict):
    dead = set()
    for ws in state.ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    state.ws_clients -= dead


async def training_worker(images_dir: str, output_dir: str, steps: int):
    """纯 async 训练任务 — 使用 asyncio 子进程，不阻塞事件循环。"""
    state.running = True
    state.cancelled = False
    state.images_dir = images_dir
    state.output_dir = output_dir
    state.step = 0
    state.total_steps = steps
    state.loss = 0.0
    state.pts = 0
    state.error_msg = ""
    state.start_time = time.time()

    try:
        env = os.environ.copy()
        env["TQDM_DISABLE"] = "0"
        env["PYTHONUNBUFFERED"] = "1"

        # 自动查找 COLMAP
        colmap_exe = os.path.join(SCRIPT_DIR, "colmap", "bin", "colmap.exe")
        if os.path.exists(colmap_exe):
            env["COLMAP_EXE"] = colmap_exe

        state.phase = "colmap"
        await broadcast({"type": "phase", "phase": "colmap",
                         "message": "COLMAP 特征提取与匹配中...如果照片分辨率很高，这一步可能需要 5-30 分钟"})

        proc = await asyncio.create_subprocess_exec(
            sys.executable, TRAIN_SCRIPT, images_dir,
            "--output", output_dir, "--steps", str(steps),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        state.process = proc

        # 异步逐行读取输出
        tqdm_pattern = re.compile(
            r"训练 3DGS:\s*(?P<pct>\d+)%\|.*?\|\s*(?P<step>\d+)/(?P<total>\d+)"
            r"\[.*?,\s*(?P<loss>[\d.]+),.*?pts=(?P<pts>\d+)\]"
        )
        phase_pattern = re.compile(r"\[(COLMAP|3DGS|导出|初始化|数据)\]")

        while proc.stdout:
            try:
                line = await asyncio.wait_for(
                    proc.stdout.readline(), timeout=3600.0
                )
            except asyncio.TimeoutError:
                continue

            if not line:
                break

            if state.cancelled:
                proc.terminate()
                break

            text = line.decode("utf-8", errors="replace").rstrip()
            if not text:
                continue

            # 检测阶段
            phase_m = phase_pattern.search(text)
            if phase_m:
                p = phase_m.group(1)
                if p == "COLMAP":
                    state.phase = "colmap"
                elif p in ("3DGS", "训练", "初始化"):
                    state.phase = "training"
                elif p == "导出":
                    state.phase = "exporting"

            # 解析 tqdm 进度
            tqdm_m = tqdm_pattern.search(text)
            if tqdm_m:
                state.step = int(tqdm_m.group("step"))
                state.total_steps = int(tqdm_m.group("total"))
                state.loss = float(tqdm_m.group("loss"))
                state.pts = int(tqdm_m.group("pts"))
                elapsed = time.time() - state.start_time
                eta = (elapsed / max(state.step, 1)) * (state.total_steps - state.step)
                await broadcast({
                    "type": "progress",
                    "step": state.step,
                    "total": state.total_steps,
                    "loss": state.loss,
                    "pts": state.pts,
                    "pct": int(tqdm_m.group("pct")),
                    "elapsed": int(elapsed),
                    "eta": int(eta),
                })
            else:
                await broadcast({"type": "log", "text": text})

            if "全部完成" in text or "SPLAT 保存" in text:
                state.phase = "done"

        await proc.wait()
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
    with open(VIEWER_HTML, "r", encoding="utf-8") as f:
        html = f.read()
    if splat:
        html = html.replace("</body>",
            f'<script>window.BUILTIN_SPLAT="{splat}";</script></body>')
    return HTMLResponse(html)


@app.post("/api/start")
async def start_training(data: dict):
    if state.running:
        return {"error": "已有训练任务运行中"}

    images_dir = data.get("images_dir", "")
    steps = data.get("steps", DEFAULT_STEPS)

    if not images_dir or not os.path.isdir(images_dir):
        return {"error": "照片文件夹无效"}

    output_dir = os.path.join(SCRIPT_DIR, "output", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    # 使用 asyncio Task（不再用 threading + asyncio.run 反模式）
    asyncio.create_task(training_worker(images_dir, output_dir, steps))

    return {"success": True, "output_dir": output_dir}


@app.post("/api/cancel")
async def cancel_training():
    if state.process and state.running:
        state.cancelled = True
        state.process.terminate()
        return {"success": True}
    return {"error": "没有运行中的训练任务"}


@app.get("/api/status")
async def get_status():
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
