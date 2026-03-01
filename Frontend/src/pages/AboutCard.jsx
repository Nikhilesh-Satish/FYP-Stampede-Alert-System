import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import Navbar from "../components/Navbar";
import styles from "./AboutCard.module.css";
import { FaLinkedin } from "react-icons/fa";

const AboutCard = ({ photo, name, description, socials }) => {
  return (
    <div className={styles.aboutCard}>
      <img src={photo} alt={name} className={styles.aboutPhoto} />
      <h2 className={styles.aboutName}>{name}</h2>
      <p className={styles.aboutDesc}>{description}</p>
      <a href={socials} target="_blank">
        <FaLinkedin className={styles.linkedinIcon} />
      </a>
    </div>
  );
};

export default AboutCard;
