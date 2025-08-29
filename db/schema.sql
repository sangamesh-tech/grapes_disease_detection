CREATE DATABASE IF NOT EXISTS grape_disease;
USE grape_disease;

CREATE TABLE history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_path VARCHAR(255),
    result VARCHAR(100),
    confidence FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
