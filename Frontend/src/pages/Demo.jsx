import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import Navbar from "../components/Navbar";
import styles from "./Demo.module.css";

const Demo = () => {
  return (
    <center>
      <form className={styles.demoForm}>
        <h2>
          Fill the form below in case of any queries! Our team will be happy to
          reach out to you and help you out!
        </h2>

        <input
          className={styles.inputField}
          type="text"
          id="fname"
          name="fname"
          placeholder="Enter your name"
        ></input>

        <input
          className={styles.inputField}
          type="text"
          id="orgname"
          name="orgname"
          placeholder="Enter your organisation name"
        ></input>

        <input
          className={styles.inputField}
          type="email"
          id="email"
          name="email"
          placeholder="Enter your email"
        ></input>
        <input
          className={styles.inputField}
          id="contact"
          name="contact"
          placeholder="Enter your contact number"
        ></input>
        <textarea
          className={styles.inputField}
          id="message"
          name="message"
          placeholder="Enter your query"
        ></textarea>
        <button className={styles.submitButton} type="submit">
          Submit
        </button>
      </form>
    </center>
  );
};

export default Demo;
