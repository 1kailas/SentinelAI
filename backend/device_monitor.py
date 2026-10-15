"""Best-effort Windows Plug and Play watcher for locally enumerated devices."""

from __future__ import annotations

import json
import platform
import subprocess
import threading


class WindowsDeviceMonitor:
	def __init__(self, callback, user_provider, interval: float = 3.0):
		self.callback = callback
		self.user_provider = user_provider
		self.interval = interval
		self._stop = threading.Event()
		self._thread = None
		self._devices = {}
		self._owners = {}

	@property
	def supported(self):
		return platform.system() == "Windows"

	def start(self):
		if not self.supported or self._thread and self._thread.is_alive():
			return
		self._thread = threading.Thread(target=self._run, name="sentinel-device-monitor", daemon=True)
		self._thread.start()

	def stop(self):
		self._stop.set()
		if self._thread:
			self._thread.join(timeout=2)

	def _run(self):
		self._devices = self._snapshot()
		while not self._stop.wait(self.interval):
			current = self._snapshot()
			for device_id, device in current.items():
				if device_id not in self._devices:
					user_id = self.user_provider()
					if user_id:
						self._owners[device_id] = user_id
						self.callback(user_id, "DEVICE_CONNECT", device)
			for device_id in self._devices.keys() - current.keys():
				user_id = self._owners.pop(device_id, None) or self.user_provider()
				if user_id:
					self.callback(user_id, "DEVICE_DISCONNECT", self._devices[device_id])
			self._devices = current

	def _snapshot(self):
		if not self.supported:
			return {}
		command = ["powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-PnpDevice -PresentOnly | Select-Object InstanceId,FriendlyName,Class,Status | ConvertTo-Json -Compress"]
		try:
			result = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
			if result.returncode != 0 or not result.stdout.strip():
				return {}
			payload = json.loads(result.stdout)
			items = payload if isinstance(payload, list) else [payload]
			return {item["InstanceId"]: {"device_id": item["InstanceId"], "name": item.get("FriendlyName") or item["InstanceId"], "device_type": item.get("Class") or "Unknown", "status": item.get("Status") or "Unknown"} for item in items if item.get("InstanceId")}
		except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TypeError):
			return {}