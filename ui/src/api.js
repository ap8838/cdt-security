import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export const fetchAlerts = (limit = 200, since_id = 0, dataset = "") =>
  axios
    .get(
      `${API_BASE}/alerts?limit=${limit}&since_id=${since_id}${
        dataset ? `&dataset=${dataset}` : ""
      }`
    )
    .then((r) => r.data);


export const fetchAssets = () =>
  axios.get(`${API_BASE}/assets`).then(r => r.data);

export const fetchDatasets = () =>
  axios.get(`${API_BASE}/datasets`).then(r => r.data.datasets);

export const postScore = (payload, dataset = null) => {
  const url = dataset
    ? `${API_BASE}/score/${dataset}`
    : `${API_BASE}/score`;
  return axios.post(url, payload).then(r => r.data);
};

export const fetchBlockRecord = (tx_hash) =>
  axios.get(`${API_BASE}/blockchain/verify?tx_hash=${encodeURIComponent(tx_hash)}`).then(r => r.data);

export const generateSamples = (payload) =>
  axios.post(`${API_BASE}/adversarial/generate`, payload).then(r => r.data);

export const fetchMetrics = () =>
  axios.get(`${API_BASE}/metrics`).then(r => r.data);
