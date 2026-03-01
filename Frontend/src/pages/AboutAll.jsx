import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import Navbar from "../components/Navbar";
import styles from "./AboutAll.module.css";

const AboutAll = () => {
  return (
    <div className={styles.about}>
      <div>
        <h1 className={styles.title}>A little about us </h1>
      </div>
      <h4>
        We are all Final year students at JSS Science And Technology University,
        Mysore. We built this project as part of our Final Year Project. We
        wanted to create an application that could contribute massively to
        society. Right during this time, there were multiple instances of
        stampedes across our country and the whole world. Most of these could
        have been prevented with proper monitoring and management of crowds.
        Hence, we decided to come up with a project that could help acheive this
        using AI models. Our project helps in predicting a stampede much before
        it actually occurs.
      </h4>
    </div>
  );
};

export default AboutAll;
