import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import Navbar from "../components/Navbar";
import styles from "./StatCard.module.css";
import AboutCard from "./AboutCard";
import AboutAll from "./AboutAll";

const StatCard = ({ img, desc }) => {
  return (
    <div className={styles.statCard}>
      <img src={img} alt="statistic" className={styles.statImg} />
      <p className={styles.statDesc}>{desc}</p>
    </div>
  );
};

export default StatCard;
