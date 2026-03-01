import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import Navbar from "../components/Navbar";
import styles from "./About.module.css";
import AboutCard from "./AboutCard";
import AboutAll from "./AboutAll";

const About = () => {
  return (
    <div>
      <Navbar />
      <div className={styles.about}>
        <AboutAll />

        <AboutCard
          photo={"./src/assets/1.jpg"}
          name="Praagna L Vaishnavi"
          description="Praagna is an Intern at GoTo. She leads the team and is responsible for the overall project management, coordination, and communication. With her strong leadership skills and dedication, Praagna ensures that the team stays on track and achieves its goals effectively. Responsible for evaluating various models. "
          socials="https://www.linkedin.com/in/praagna-vaishnavi/"
        />
        <AboutCard
          photo={"./src/assets/2.svg"}
          name="Nikhilesh Satish"
          description="Nikhilesh is an Intern at HPE. He is responsible for the development of the project including model development, backend and frontend development"
          socials="https://www.linkedin.com/in/nikhilesh-satish-50b315273/"
        />
        <AboutCard
          photo={"./src/assets/2.svg"}
          name="Chethan S Narayan"
          description="Chethan is an Intern at Infosys. He is responsible for the design of the frontend of the project and also a key part of documentation"
          socials="https://www.linkedin.com/in/chethan-s-narayan-920ab8261/"
        />
        <AboutCard
          photo={"./src/assets/2.svg"}
          name="Ngangom Sudarshan"
          description="Sudarshan is a key member of the team. He worked alongside Chethan to get the design of the frontend and documentation ready. "
          socials="https://www.linkedin.com/in/sudarshan-ngangom-2502b8375/"
        />
      </div>
    </div>
  );
};

export default About;
