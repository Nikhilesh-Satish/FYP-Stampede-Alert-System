import styles from "./AboutAll.module.css";

const AboutAll = () => {
  return (
    <div className={styles.about}>
      <p className={styles.kicker}>The team</p>
      <h1 className={styles.title}>Built as a final year project with public safety in mind.</h1>
      <p className={styles.body}>
        We are final year students at JSS Science and Technology University,
        Mysore. The project grew from a simple observation: many crowd disasters
        are preceded by visible operational warning signs, but those signals are
        often missed in time. We set out to build a system that combines AI,
        monitoring, and prediction so responders can intervene earlier.
      </p>
    </div>
  );
};

export default AboutAll;
