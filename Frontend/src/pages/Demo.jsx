import styles from "./Demo.module.css";

const Demo = () => {
  return (
    <form className={styles.demoForm}>
      <div className={styles.formHeader}>
        <span className={styles.kicker}>Direct line</span>
        <h2 className={styles.title}>Tell us about your venue or event</h2>
        <p className={styles.subtitle}>
          Fill in the details below and the team can follow up on deployment,
          operations, or research queries.
        </p>
      </div>

      <div className={styles.formGrid}>
        <input
          className={styles.inputField}
          type="text"
          id="fname"
          name="fname"
          placeholder="Your name"
        />

        <input
          className={styles.inputField}
          type="text"
          id="orgname"
          name="orgname"
          placeholder="Organisation"
        />

        <input
          className={styles.inputField}
          type="email"
          id="email"
          name="email"
          placeholder="Email address"
        />
        <input
          className={styles.inputField}
          id="contact"
          name="contact"
          placeholder="Contact number"
        />
      </div>

      <textarea
        className={`${styles.inputField} ${styles.textarea}`}
        id="message"
        name="message"
        placeholder="Describe your query, venue scale, and expected crowd conditions"
      />
      <button className={styles.submitButton} type="submit">
        Send Enquiry
      </button>
    </form>
  );
};

export default Demo;
