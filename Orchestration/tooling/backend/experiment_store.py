import json
import os
import shutil
import time
from typing import Optional


class ExperimentStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._counter = 0
        os.makedirs(base_dir, exist_ok=True)

    def create_run(self, name: str, config: dict) -> str:
        self._counter += 1
        run_id = f"{int(time.time() * 1000)}_{self._counter}"
        run_dir = os.path.join(self.base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        config_with_meta = {**config, "name": name, "run_id": run_id}
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config_with_meta, f, indent=2)
        metrics = {"epochs": [], "summary": {}}
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        status = {"status": "pending", "created_at": time.time()}
        with open(os.path.join(run_dir, "status.json"), "w") as f:
            json.dump(status, f, indent=2)
        return run_id

    def list_runs(self, include_deleted: bool = False) -> list:
        if not os.path.exists(self.base_dir):
            return []
        runs = []
        for entry in os.listdir(self.base_dir):
            run_dir = os.path.join(self.base_dir, entry)
            if not os.path.isdir(run_dir):
                continue
            config_path = os.path.join(run_dir, "config.json")
            status_path = os.path.join(run_dir, "status.json")
            metrics_path = os.path.join(run_dir, "metrics.json")
            if not os.path.exists(config_path):
                continue
            with open(config_path) as f:
                config = json.load(f)
            with open(status_path) as f:
                status_data = json.load(f)
            status = status_data.get("status", "pending")
            if include_deleted and status != "deleted":
                continue
            if not include_deleted and status == "deleted":
                continue
            summary = {}
            epochs = []
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
                    summary = metrics.get("summary", {})
                    epochs = metrics.get("epochs", [])
            total_epochs = sum(
                s.get("epochs", 0) for s in config.get("stages", [])
            )
            runs.append({
                "id": entry,
                "name": config.get("name", ""),
                "model_type": config.get("model_type", "lbeads"),
                "status": status,
                "created_at": status_data.get("created_at", 0),
                "epoch_count": len(epochs),
                "total_epochs": total_epochs,
                "summary": summary,
            })
        runs.sort(key=lambda r: r["created_at"], reverse=True)
        return runs

    def get_run(self, run_id: str) -> Optional[dict]:
        run_dir = os.path.join(self.base_dir, run_id)
        if not os.path.isdir(run_dir):
            return None
        config_path = os.path.join(run_dir, "config.json")
        status_path = os.path.join(run_dir, "status.json")
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(config_path) as f:
            config = json.load(f)
        with open(status_path) as f:
            status_data = json.load(f)
        with open(metrics_path) as f:
            metrics = json.load(f)
        files = []
        for root, dirs, filenames in os.walk(run_dir):
            for fname in filenames:
                if fname.endswith(('.png', '.pth')):
                    rel = os.path.relpath(os.path.join(root, fname), run_dir)
                    files.append(rel)
        return {
            "id": run_id,
            "name": config.get("name", ""),
            "config": config,
            "status": status_data.get("status", "pending"),
            "created_at": status_data.get("created_at", 0),
            "metrics": metrics,
            "files": files,
        }

    def update_status(self, run_id: str, status: str):
        status_path = os.path.join(self.base_dir, run_id, "status.json")
        with open(status_path) as f:
            data = json.load(f)
        data["status"] = status
        data["updated_at"] = time.time()
        with open(status_path, "w") as f:
            json.dump(data, f, indent=2)

    def append_epoch(self, run_id: str, epoch_data: dict):
        metrics_path = os.path.join(self.base_dir, run_id, "metrics.json")
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["epochs"].append(epoch_data)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def set_summary(self, run_id: str, summary: dict):
        metrics_path = os.path.join(self.base_dir, run_id, "metrics.json")
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["summary"] = summary
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def append_error(self, run_id: str, error: dict):
        metrics_path = os.path.join(self.base_dir, run_id, "metrics.json")
        with open(metrics_path) as f:
            metrics = json.load(f)
        if "errors" not in metrics:
            metrics["errors"] = []
        metrics["errors"].append(error)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def soft_delete_run(self, run_id: str):
        status_path = os.path.join(self.base_dir, run_id, "status.json")
        if not os.path.exists(status_path):
            return False
        with open(status_path) as f:
            data = json.load(f)
        data["previous_status"] = data.get("status", "pending")
        data["status"] = "deleted"
        data["deleted_at"] = time.time()
        with open(status_path, "w") as f:
            json.dump(data, f, indent=2)
        return True

    def restore_run(self, run_id: str):
        status_path = os.path.join(self.base_dir, run_id, "status.json")
        if not os.path.exists(status_path):
            return False
        with open(status_path) as f:
            data = json.load(f)
        if data.get("status") != "deleted":
            return False
        data["status"] = data.pop("previous_status", "failed")
        data.pop("deleted_at", None)
        data["updated_at"] = time.time()
        with open(status_path, "w") as f:
            json.dump(data, f, indent=2)
        return True

    def permanently_delete_run(self, run_id: str):
        run_dir = os.path.join(self.base_dir, run_id)
        if not os.path.isdir(run_dir):
            return False
        shutil.rmtree(run_dir)
        return True
