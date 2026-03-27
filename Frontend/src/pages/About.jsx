import Navbar from "../components/Navbar";
import styles from "./About.module.css";
import AboutCard from "./AboutCard";
import AboutAll from "./AboutAll";
import Reveal from "../components/Reveal";

const About = () => {
  return (
    <div className={styles.page}>
      <Navbar />
      <div className={styles.about}>
        <Reveal>
          <AboutAll />
        </Reveal>

        <div className={styles.teamGrid}>
          <Reveal delay={0}>
            <AboutCard
              photo={"./src/assets/1.jpg"}
              name="Praagna L Vaishnavi"
              description="Praagna is an Intern at GoTo. She leads the team and drives project management, coordination, model evaluation, and full-stack execution."
              socials="https://www.linkedin.com/in/praagna-vaishnavi/"
            />
          </Reveal>
          <Reveal delay={80}>
            <AboutCard
              photo={"./src/assets/2.svg"}
              name="Nikhilesh Satish"
              description="Nikhilesh is an Intern at HPE. He contributes across model development, backend systems, and frontend implementation."
              socials="https://www.linkedin.com/in/nikhilesh-satish-50b315273/"
            />
          </Reveal>
          <Reveal delay={160}>
            <AboutCard
              photo={"./src/assets/2.svg"}
              name="Chethan S Narayan"
              description="Chethan is an Intern at Infosys. He contributed to the interface design direction and project documentation."
              socials="https://www.linkedin.com/in/chethan-s-narayan-920ab8261/"
            />
          </Reveal>
          <Reveal delay={240}>
            <AboutCard
              photo={"./src/assets/2.svg"}
              name="Ngangom Sudarshan"
              description="Sudarshan supported frontend design and documentation, helping shape the communication layer of the project."
              socials="https://www.linkedin.com/in/sudarshan-ngangom-2502b8375/"
            />
          </Reveal>
        </div>
      </div>
    </div>
  );
};

export default About;
