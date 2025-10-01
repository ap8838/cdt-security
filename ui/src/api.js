import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export const fetchAlerts = (limit = 100, since_id = 0) =>
  axios.get(`${API_BASE}/alerts?limit=${limit}&since_id=${since_id}`).then(r => r.data);

export const fetchAssets = () =>
  axios.get(`${API_BASE}/assets`).then(r => r.data);

export const postScore = (payload) =>
  axios.post(`${API_BASE}/score`, payload).then(r => r.data);
