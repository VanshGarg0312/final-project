import axios from "axios";

export const predictNews = (text) => {
  return axios.post("http://localhost:8080/news/predict", { text });
};